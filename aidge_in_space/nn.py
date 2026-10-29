"""! @brief Modèles de Ballé et al. : 2017 (notebook) et 2018 (factorisé, hyperprior).

Les noms de couches reprennent ceux du notebook (Balle2017) et de CompressAI (modèles 2018), ce
qui permet de charger leurs checkpoints tels quels, sans dépendre de CompressAI.

Contrat commun des modèles (utilisé par aidge_in_space.codecs.base.LearnedCodec) :
- @c in_ch, @c N, @c M : canaux de l'image, des couches cachées et du latent ;
- @c pad_multiple : l'image est complétée à un multiple de cette taille ;
- @c downsampling : facteur de réduction spatiale de g_a (latent = image / downsampling) ;
- @c g_a, @c g_s : transformées d'analyse et de synthèse.
"""
import copy
import math
from statistics import NormalDist

import torch
import torch.nn as nn
import torch.nn.functional as F


class _LowerBound(torch.autograd.Function):
    """! @brief max(x, borne) dont le gradient passe quand x >= borne ou quand il pousse x vers le haut."""

    @staticmethod
    def forward(ctx, x, bound):
        """! @brief max(x, bound)."""
        ctx.save_for_backward(x)
        ctx.bound = bound
        return x.clamp(min=bound)

    @staticmethod
    def backward(ctx, grad):
        """! @brief Gradient transmis là où x >= borne ou là où il fait remonter x."""
        (x,) = ctx.saved_tensors
        pass_through = (x >= ctx.bound) | (grad < 0)
        return grad * pass_through.to(grad.dtype), None


def lower_bound(x, bound):
    """! @brief Borne inférieure qui laisse passer le gradient (voir _LowerBound) pendant l'entraînement.

    @param x      Tenseur.
    @param bound  Borne inférieure (scalaire).
    @return max(x, bound).
    """
    if torch.is_grad_enabled() and x.requires_grad:
        return _LowerBound.apply(x, bound)
    return x.clamp(min=bound)


class GDN(nn.Module):
    """! @brief Generalized Divisive Normalization (Ballé et al. 2016).

    y_i = x_i / sqrt(beta_i + sum_j gamma_ij x_j^2), ou x_i * sqrt(...) pour la GDN inverse (IGDN).
    beta et gamma sont re-paramétrés (racine carrée + piédestal) pour rester positifs.
    """

    def __init__(self, ch, inverse=False, beta_min=1e-6, gamma_init=0.1,
                 reparam_offset=2 ** -18, symmetric=False):
        """! @brief Crée une GDN initialisée à beta = 1, gamma = gamma_init * identité.

        @param ch              Nombre de canaux.
        @param inverse         True pour l'IGDN (multiplication au lieu de division).
        @param beta_min        Valeur minimale de beta.
        @param gamma_init      Valeur initiale de la diagonale de gamma.
        @param reparam_offset  Décalage de la re-paramétrisation (piédestal = offset²).
        @param symmetric       True pour symétriser gamma (variante du notebook).
        """
        super().__init__()
        ## Nombre de canaux, GDN inverse (IGDN), gamma symétrisé.
        self.ch, self.inverse, self.symmetric = ch, inverse, symmetric
        ## Piédestal de la re-paramétrisation (offset²).
        self.pedestal = reparam_offset ** 2
        ## Borne inférieure du paramètre brut beta.
        self.beta_bound = (beta_min + self.pedestal) ** 0.5
        ## Borne inférieure du paramètre brut gamma.
        self.gamma_bound = reparam_offset
        ## Paramètre brut de beta (forme C) ; valeur effective : effective_params().
        self.beta = nn.Parameter(torch.sqrt(torch.ones(ch) + self.pedestal))
        ## Paramètre brut de gamma (forme C x C) ; valeur effective : effective_params().
        self.gamma = nn.Parameter(torch.sqrt(gamma_init * torch.eye(ch) + self.pedestal))

    def effective_params(self):
        """! @brief Paramètres effectifs après re-paramétrisation.

        @return Couple (beta de forme C, gamma de forme C x C).
        """
        beta = lower_bound(self.beta, self.beta_bound) ** 2 - self.pedestal
        gamma = lower_bound(self.gamma, self.gamma_bound) ** 2 - self.pedestal
        if self.symmetric:
            gamma = 0.5 * (gamma + gamma.t())
        return beta, gamma

    def forward(self, x):
        """! @brief Applique la (I)GDN à un tenseur B x C x H x W."""
        beta, gamma = self.effective_params()
        norm = torch.sqrt(F.conv2d(x * x, gamma.view(self.ch, self.ch, 1, 1), beta))
        return x * norm if self.inverse else x / norm

    def extra_repr(self):
        """! @brief Paramètres affichés par print(module)."""
        return f"ch={self.ch}, inverse={self.inverse}"


class FrozenGDN(nn.Module):
    """! @brief GDN figée pour le déploiement : conv 1x1 (poids gamma, biais beta), racine, division.

    N'utilise que des opérateurs ONNX simples (Mul, Conv, Sqrt, Div), pris en charge par Aidge.
    """

    def __init__(self, gdn):
        """! @brief Fige les paramètres effectifs d'une GDN.

        @param gdn  GDN entraînée.
        """
        super().__init__()
        beta, gamma = gdn.effective_params()
        ## Nombre de canaux et GDN inverse, repris de la GDN d'origine.
        self.ch, self.inverse = gdn.ch, gdn.inverse
        ## Conv 1x1 qui calcule beta + gamma * x² (poids gamma, biais beta).
        self.norm = nn.Conv2d(gdn.ch, gdn.ch, 1)
        self.norm.weight.data = gamma.detach().clone().view(gdn.ch, gdn.ch, 1, 1)
        self.norm.bias.data = beta.detach().clone()

    def forward(self, x):
        """! @brief Même calcul que GDN.forward()."""
        n = torch.sqrt(self.norm(x * x))
        return x * n if self.inverse else x / n


def freeze(module):
    """! @brief Copie d'un module où chaque GDN devient une FrozenGDN.

    @param module  Module quelconque (non modifié).
    @return Copie profonde en mode évaluation, exportable en ONNX avec des opérateurs simples.
    """
    m = copy.deepcopy(module).eval()

    def swap(parent):
        for name, child in parent.named_children():
            if isinstance(child, GDN):
                setattr(parent, name, FrozenGDN(child))
            else:
                swap(child)

    swap(m)
    return m


# ---------------------------------------------------------------- modèles d'entropie factorisés

class _FactorizedPrior(nn.Module):
    """! @brief Densité apprise indépendante par canal (base de FactorizedDensity et EntropyBottleneck).

    Les sous-classes définissent @c channels et _logits(v), la fonction de répartition (avant
    sigmoïde) évaluée en v de forme C x n.
    """

    def prob(self, v):
        """! @brief Probabilité que le latent quantifié vaille v : P(v - 0,5 < y < v + 0,5).

        Calculée du côté où la différence de sigmoïdes est numériquement précise.
        @param v  Tenseur C x n.
        @return Tenseur C x n de probabilités.
        """
        lower, upper = self._logits(v - 0.5), self._logits(v + 0.5)
        s = -torch.sign(lower + upper)
        s = torch.where(s == 0, torch.ones_like(s), s)
        return torch.abs(torch.sigmoid(s * upper) - torch.sigmoid(s * lower))

    def likelihood(self, y):
        """! @brief Probabilité de chaque valeur d'un tenseur quantifié.

        @param y  Tenseur B x C x H x W.
        @return Tenseur de même forme.
        """
        B, C = y.shape[:2]
        v = y.transpose(0, 1).reshape(C, -1)
        return self.prob(v).reshape(C, B, *y.shape[2:]).transpose(0, 1)

    def medians(self):
        """! @brief Médiane de chaque canal, centre de la quantification (0 par défaut).

        @return Tenseur de forme C.
        """
        return torch.zeros(self.channels)


class FactorizedDensity(_FactorizedPrior):
    """! @brief Prior factorisé simplifié du notebook : un scalaire par couche et par canal."""

    def __init__(self, ch, n_layers=4):
        """! @brief Initialise des paramètres aléatoires proches de 0.

        @param ch        Nombre de canaux.
        @param n_layers  Nombre de couches de la fonction de répartition.
        """
        super().__init__()
        ## Nombre de canaux.
        self.channels = ch
        init = lambda: nn.Parameter(torch.randn(1, ch, 1, 1) * 0.01)
        ## Pentes des couches (passées par softplus), un scalaire par canal.
        self.h = nn.ParameterList([init() for _ in range(n_layers)])
        ## Biais des couches, un scalaire par canal.
        self.b = nn.ParameterList([init() for _ in range(n_layers)])
        ## Facteurs des non-linéarités tanh, un scalaire par canal.
        self.a = nn.ParameterList([init() for _ in range(n_layers - 1)])

    def _logits(self, v):
        """! @brief Fonction de répartition avant sigmoïde, v de forme C x n."""
        for k in range(len(self.h)):
            v = v * F.softplus(self.h[k].view(-1, 1)) + self.b[k].view(-1, 1)
            if k < len(self.a):
                v = v + torch.tanh(v) * torch.tanh(self.a[k].view(-1, 1))
        return v


class EntropyBottleneck(_FactorizedPrior):
    """! @brief Entropy bottleneck de Ballé et al. 2018, avec les noms de paramètres de CompressAI."""

    def __init__(self, channels, filters=(3, 3, 3, 3), init_scale=10.0):
        """! @brief Initialisation de CompressAI.

        @param channels    Nombre de canaux.
        @param filters     Largeur des couches cachées de la fonction de répartition.
        @param init_scale  Échelle initiale de la densité.
        """
        super().__init__()
        ## Nombre de canaux.
        self.channels = channels
        dims = (1,) + tuple(filters) + (1,)
        scale = init_scale ** (1 / (len(filters) + 1))
        ## Matrices, biais et facteurs de chaque couche de la fonction de répartition (noms de CompressAI).
        self._matrices, self._biases, self._factors = nn.ParameterList(), nn.ParameterList(), nn.ParameterList()
        for i in range(len(filters) + 1):
            init = math.log(math.expm1(1 / scale / dims[i + 1]))
            self._matrices.append(nn.Parameter(torch.full((channels, dims[i + 1], dims[i]), init)))
            self._biases.append(nn.Parameter(torch.empty(channels, dims[i + 1], 1).uniform_(-0.5, 0.5)))
            if i < len(filters):
                self._factors.append(nn.Parameter(torch.zeros(channels, dims[i + 1], 1)))
        ## Quantiles appris de chaque canal (bas, médiane, haut).
        self.quantiles = nn.Parameter(torch.tensor([-init_scale, 0.0, init_scale]).repeat(channels, 1, 1))

    def _logits(self, v):
        """! @brief Fonction de répartition avant sigmoïde, v de forme C x n."""
        x = v.unsqueeze(1)                                   # C x 1 x n
        for i in range(len(self._matrices)):
            x = torch.matmul(F.softplus(self._matrices[i]), x) + self._biases[i]
            if i < len(self._factors):
                x = x + torch.tanh(self._factors[i]) * torch.tanh(x)
        return x.squeeze(1)

    def medians(self):
        """! @brief Médianes apprises (quantile central de @c quantiles)."""
        return self.quantiles[:, 0, 1].detach()


# ---------------------------------------------------------------- prior gaussien (hyperprior)

## Plus petite échelle de gaussienne utilisée (en dessous, la loi est trop piquée pour être codée).
SCALE_BOUND = 0.11
## Les 64 échelles quantifiées de CompressAI, de 0,11 à 256 en progression géométrique.
SCALE_TABLE = torch.exp(torch.linspace(math.log(0.11), math.log(256), 64))


def _std_cdf(x):
    """! @brief Fonction de répartition de la loi normale centrée réduite."""
    return 0.5 * torch.erfc(-(2 ** -0.5) * x)


def gaussian_likelihood(y_hat, scales):
    """! @brief Probabilité de chaque latent quantifié sous une gaussienne centrée.

    @param y_hat   Latents quantifiés (entiers).
    @param scales  Écarts-types prédits par h_s, même forme.
    @return Tenseur de probabilités de même forme.
    """
    s = scales.clamp(min=SCALE_BOUND)
    v = y_hat.abs()
    return _std_cdf((0.5 - v) / s) - _std_cdf((-0.5 - v) / s)


def scale_indexes(scales):
    """! @brief Indice, dans SCALE_TABLE, de la plus petite échelle supérieure ou égale à l'échelle prédite.

    @param scales  Écarts-types prédits par h_s.
    @return Tenseur d'entiers de même forme (indices de table pour le codeur arithmétique).
    """
    s = scales.clamp(min=SCALE_BOUND).contiguous()
    return torch.searchsorted(SCALE_TABLE[:-1].contiguous(), s)


def gaussian_pmfs(tail_mass=1e-9):
    """! @brief Lois discrètes centrées, une par niveau de SCALE_TABLE.

    @param tail_mass  Masse des queues coupées de part et d'autre.
    @return Couple (pmfs, los) : probabilités de chaque support et première valeur de chacun,
            au format de aidge_in_space.entropy.make_tables().
    """
    mult = -NormalDist().inv_cdf(tail_mass / 2)
    pmfs, los = [], []
    for s in SCALE_TABLE.double():
        c = int(math.ceil(float(s) * mult))
        k = torch.arange(-c, c + 1, dtype=torch.float64).abs()
        pmfs.append((_std_cdf((0.5 - k) / s) - _std_cdf((-0.5 - k) / s)).numpy())
        los.append(-c)
    return pmfs, los


# ---------------------------------------------------------------- architectures

def _conv(cin, cout, k=5, s=2):
    """! @brief Convolution k x k de pas s avec padding « same »."""
    return nn.Conv2d(cin, cout, k, stride=s, padding=k // 2)


def _deconv(cin, cout, k=5, s=2):
    """! @brief Convolution transposée k x k qui multiplie la taille par s."""
    return nn.ConvTranspose2d(cin, cout, k, stride=s, padding=k // 2, output_padding=s - 1)


class Analysis2017(nn.Module):
    """! @brief Transformée d'analyse de Ballé 2017 : conv 9x9/4, 5x5/2, 5x5/2, chacune suivie d'une GDN."""

    def __init__(self, in_ch=1, N=128, gdn_last=True):
        """! @brief Crée les couches (poids non entraînés).
        @param in_ch     Canaux de l'image (1 en gris, 3 en RGB).
        @param N         Canaux des couches et du latent.
        @param gdn_last  False pour supprimer la dernière GDN.
        """
        super().__init__()
        ## Conv 9x9 de pas 4.
        self.conv1 = nn.Conv2d(in_ch, N, 9, stride=4, padding=4)
        ## GDN après conv1.
        self.gdn1 = GDN(N, symmetric=True)
        ## Conv 5x5 de pas 2.
        self.conv2 = nn.Conv2d(N, N, 5, stride=2, padding=2)
        ## GDN après conv2.
        self.gdn2 = GDN(N, symmetric=True)
        ## Conv 5x5 de pas 2.
        self.conv3 = nn.Conv2d(N, N, 5, stride=2, padding=2)
        ## GDN après conv3 (identité si gdn_last est faux).
        self.gdn3 = GDN(N, symmetric=True) if gdn_last else nn.Identity()

    def forward(self, x):
        """! @brief Image 1 x in_ch x H x W -> latent 1 x N x H/16 x W/16."""
        x = self.gdn1(self.conv1(x))
        x = self.gdn2(self.conv2(x))
        return self.gdn3(self.conv3(x))


class Synthesis2017(nn.Module):
    """! @brief Transformée de synthèse de Ballé 2017, symétrique de Analysis2017 (IGDN + conv transposées)."""

    def __init__(self, in_ch=1, N=128, gdn_last=True):
        """! @brief Mêmes paramètres que Analysis2017."""
        super().__init__()
        ## IGDN d'entrée (identité si gdn_last est faux).
        self.igdn1 = GDN(N, inverse=True, symmetric=True) if gdn_last else nn.Identity()
        ## Conv transposée 5x5 de pas 2.
        self.deconv1 = nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1)
        ## IGDN après deconv1.
        self.igdn2 = GDN(N, inverse=True, symmetric=True)
        ## Conv transposée 5x5 de pas 2.
        self.deconv2 = nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1)
        ## IGDN après deconv2.
        self.igdn3 = GDN(N, inverse=True, symmetric=True)
        ## Conv transposée 9x9 de pas 4, vers les canaux de l'image.
        self.deconv3 = nn.ConvTranspose2d(N, in_ch, 9, stride=4, padding=4, output_padding=3)

    def forward(self, y):
        """! @brief Latent 1 x N x h x w -> image 1 x in_ch x 16h x 16w."""
        y = self.deconv1(self.igdn1(y))
        y = self.deconv2(self.igdn2(y))
        return self.deconv3(self.igdn3(y))


class Balle2017(nn.Module):
    """! @brief Modèle du notebook (E2ECompressor) : mêmes noms de couches, checkpoints compatibles."""

    ## L'image est complétée à un multiple de cette taille.
    pad_multiple = 16
    ## Facteur de réduction spatiale de g_a.
    downsampling = 16

    def __init__(self, in_ch=1, N=128, gdn_last=True):
        """! @brief Mêmes paramètres que Analysis2017 ; le latent a N canaux (M = N)."""
        super().__init__()
        ## Canaux de l'image, des couches cachées et du latent.
        self.in_ch, self.N, self.M = in_ch, N, N
        ## Transformée d'analyse (encodeur, à bord).
        self.g_a = Analysis2017(in_ch, N, gdn_last)
        ## Transformée de synthèse (décodeur, au sol).
        self.g_s = Synthesis2017(in_ch, N, gdn_last)
        ## Prior factorisé des latents (nom du notebook).
        self.entropy_model = FactorizedDensity(N)

    @property
    def prior(self):
        """! @brief Prior factorisé des latents (FactorizedDensity)."""
        return self.entropy_model


class FactorizedPrior2018(nn.Module):
    """! @brief bmshj2018-factorized : 4 conv 5x5 de pas 2 avec GDN entre chacune, prior factorisé."""

    ## L'image est complétée à un multiple de cette taille.
    pad_multiple = 16
    ## Facteur de réduction spatiale de g_a.
    downsampling = 16

    def __init__(self, N=128, M=192):
        """! @brief Crée les couches (poids non entraînés).
        @param N  Canaux des couches cachées.
        @param M  Canaux du latent.
        """
        super().__init__()
        ## Canaux de l'image, des couches cachées et du latent.
        self.in_ch, self.N, self.M = 3, N, M
        ## Transformée d'analyse (encodeur, à bord) : 4 conv 5x5 de pas 2 et 3 GDN.
        self.g_a = nn.Sequential(_conv(3, N), GDN(N), _conv(N, N), GDN(N), _conv(N, N), GDN(N), _conv(N, M))
        ## Transformée de synthèse (décodeur, au sol).
        self.g_s = nn.Sequential(_deconv(M, N), GDN(N, inverse=True), _deconv(N, N), GDN(N, inverse=True),
                                 _deconv(N, N), GDN(N, inverse=True), _deconv(N, 3))
        ## Prior factorisé des latents (nom de CompressAI).
        self.entropy_bottleneck = EntropyBottleneck(M)

    @property
    def prior(self):
        """! @brief Prior factorisé des latents (EntropyBottleneck)."""
        return self.entropy_bottleneck


class ScaleHyperprior2018(nn.Module):
    """! @brief bmshj2018-hyperprior : un hyper-latent z, codé avec un prior factorisé, prédit l'échelle de chaque latent.

    h_a réduit encore la taille par 4 (latent / 4), d'où un padding à un multiple de 64.
    """

    ## L'image est complétée à un multiple de cette taille (g_a puis h_a divisent par 64).
    pad_multiple = 64
    ## Facteur de réduction spatiale de g_a.
    downsampling = 16

    def __init__(self, N=128, M=192):
        """! @brief Crée les couches (poids non entraînés).
        @param N  Canaux des couches cachées et de z.
        @param M  Canaux du latent y.
        """
        super().__init__()
        ## Canaux de l'image, des couches cachées et du latent.
        self.in_ch, self.N, self.M = 3, N, M
        ## Transformée d'analyse (encodeur, à bord) : 4 conv 5x5 de pas 2 et 3 GDN.
        self.g_a = nn.Sequential(_conv(3, N), GDN(N), _conv(N, N), GDN(N), _conv(N, N), GDN(N), _conv(N, M))
        ## Transformée de synthèse (décodeur, au sol).
        self.g_s = nn.Sequential(_deconv(M, N), GDN(N, inverse=True), _deconv(N, N), GDN(N, inverse=True),
                                 _deconv(N, N), GDN(N, inverse=True), _deconv(N, 3))
        ## Hyper-analyse : latent |y| -> hyper-latent z (taille divisée par 4).
        self.h_a = nn.Sequential(_conv(M, N, 3, 1), nn.ReLU(), _conv(N, N), nn.ReLU(), _conv(N, N))
        ## Hyper-synthèse : z -> écart-type prédit de chaque latent.
        self.h_s = nn.Sequential(_deconv(N, N), nn.ReLU(), _deconv(N, N), nn.ReLU(), _conv(N, M, 3, 1), nn.ReLU())
        ## Prior factorisé de l'hyper-latent z.
        self.entropy_bottleneck = EntropyBottleneck(N)


class HyperpriorOnboard(nn.Module):
    """! @brief Partie embarquée d'un hyperprior : g_a, h_a et h_s (pour le profilage et l'export)."""

    def __init__(self, model):
        """! @brief Regroupe les transformées embarquées d'un hyperprior.
        @param model  ScaleHyperprior2018 dont les sous-modules sont repris (non copiés).
        """
        super().__init__()
        ## Transformées d'analyse et d'hyper-analyse / hyper-synthèse du modèle.
        self.g_a, self.h_a, self.h_s = model.g_a, model.h_a, model.h_s

    def forward(self, x):
        """! @brief Image -> échelles prédites des latents (z arrondi sans médiane : coût identique)."""
        y = self.g_a(x)
        return self.h_s(torch.round(self.h_a(torch.abs(y))))


def load_compatible(model, state_dict):
    """! @brief Charge un checkpoint en ignorant les tampons propres à CompressAI.

    @param model       Modèle à remplir.
    @param state_dict  Dictionnaire de poids (par exemple un checkpoint CompressAI).
    @return Le modèle, en mode évaluation.
    @throws KeyError si une clé du modèle manque dans le checkpoint.
    """
    own = model.state_dict()
    missing = [k for k in own if k not in state_dict]
    if missing:
        raise KeyError(f"checkpoint incompatible, clés manquantes : {missing[:5]}")
    model.load_state_dict({k: state_dict[k] for k in own})
    return model.eval()


def n_params(module):
    """! @brief Nombre de paramètres d'un module (sous-modules compris)."""
    return sum(p.numel() for p in module.parameters())


def bits_from_likelihood(lik):
    """! @brief Débit estimé, en bits, d'un tenseur de probabilités : somme de -log2(p).

    @param lik  Probabilités (bornées à 1e-9 pour éviter log(0)).
    @return Nombre de bits (float).
    """
    return float(-torch.log2(lik.clamp(min=1e-9)).sum())

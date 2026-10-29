"""! @brief Boucle d'entraînement de Ballé 2017, portée des cellules 4, 15 et 17 du notebook.

Banque de patchs uint8 pré-extraits sur le device, perte λ·MSE + bpp, Adam avec pas divisé par 10 en
fin d'entraînement, évaluation avec vraie quantification, checkpoints atomiques et reprise. Le contrat
est dans docs/conception_entrainement.md (§ 2.3).
"""
import math
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .. import config as paths
from ..images import convert, download_kodak, list_images, to_tensor
from ..nn import Balle2017
from .config import ConfigError, TrainConfig, resolve_path
from .imaging import to_8bit

## Borne inférieure des probabilités (comme le notebook) avant -log2.
LIKELIHOOD_MIN = 1e-9
## Clés de l'historique, celles du notebook.
HISTORY_KEYS = ("step", "loss", "bpp", "psnr", "eval_step", "eval_psnr", "eval_bpp")
## Nouvelles tentatives de os.replace sous Windows (fichier ouvert en lecture) et délai entre deux, en secondes.
REPLACE_RETRIES, REPLACE_DELAY_S = 10, 0.5
## Les paramètres de l'architecture doivent être identiques à ceux du checkpoint repris.
ARCH_KEYS = ("in_ch", "N", "gdn_last", "lmbda")


@dataclass
class TrainResult:
    """! @brief Résultat d'un appel à train()."""

    ## "done" (dernier pas atteint, ou checkpoint déjà complet) ou "cancelled" (should_stop vrai).
    status: str
    ## Dernier pas effectué, égal au pas du checkpoint sauvegardé.
    step: int
    ## Chemin du checkpoint.
    ckpt_path: Path
    ## Historique (mêmes clés que le notebook).
    history: dict
    ## Dernière évaluation {"psnr": float, "bpp": float, "source": str}, ou None.
    final_eval: dict | None


def _say(*args) -> None:
    """! @brief print() immédiat (la sortie est redirigée vers train.log et lue pendant l'entraînement)."""
    print(*args, flush=True)


def seed_everything(seed: int) -> None:
    """! @brief Fixe la graine de random, numpy et torch (CUDA compris).

    @param seed  Graine.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(prefer: str = "auto") -> torch.device:
    """! @brief Choisit le device et applique les réglages de performance.

    Sur CUDA : cudnn.benchmark et TF32. Sur CPU : tous les cœurs sauf un (pour laisser l'interface réactive).
    @param prefer  @c "auto" (CUDA si disponible), @c "cpu" ou @c "cuda".
    @return Le device.
    @throws ConfigError si CUDA est demandé mais absent, ou si @p prefer est inconnu.
    """
    if prefer not in ("auto", "cpu", "cuda"):
        raise ConfigError(f"device : auto, cpu ou cuda attendu (reçu {prefer!r})")
    if prefer == "cuda" and not torch.cuda.is_available():
        raise ConfigError("device : CUDA demandé mais aucun GPU CUDA n'est disponible")
    use_cuda = torch.cuda.is_available() and prefer != "cpu"
    if use_cuda:
        torch.backends.cudnn.benchmark = True          # cuDNN choisit l'algorithme de convolution le plus rapide
        torch.backends.cuda.matmul.allow_tf32 = True   # TF32 sur les GPU Ampere et plus récents
        torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda")
    if prefer == "auto":
        _say("ATTENTION : aucun GPU CUDA détecté, entraînement sur CPU (beaucoup plus lent).")
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    return torch.device("cpu")


def init_reference_weights(model: Balle2017) -> Balle2017:
    """! @brief Initialisation du notebook : xavier_normal_ de gain √2 et biais 0,01 sur les conv et deconv.

    La GDN et le prior gardent leur initialisation.
    @param model  Modèle, modifié en place.
    @return Le même modèle.
    """
    convs = [model.g_a.conv1, model.g_a.conv2, model.g_a.conv3,
             model.g_s.deconv1, model.g_s.deconv2, model.g_s.deconv3]
    for m in convs:
        nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
        nn.init.constant_(m.bias, 0.01)
    return model


def forward_train(model: Balle2017, x: torch.Tensor) -> dict:
    """! @brief Passe avant du notebook : bruit uniforme en entraînement, vrai arrondi en évaluation.

    @param model  Modèle (son mode train/eval choisit la quantification).
    @param x      Lot B x C x H x W dans [0, 1].
    @return Dictionnaire @c y, @c y_hat, @c x_hat et @c likelihoods (probabilités non bornées).
    """
    y = model.g_a(x)
    if model.training:
        y_hat = y + torch.empty_like(y).uniform_(-0.5, 0.5)   # proxy dérivable de l'arrondi
    else:
        y_hat = torch.round(y)                                # vraie quantification (médianes nulles)
    x_hat = model.g_s(y_hat)
    return {"y": y, "y_hat": y_hat, "x_hat": x_hat, "likelihoods": model.entropy_model.likelihood(y_hat)}


def rd_loss(out: dict, x: torch.Tensor, lmbda: float):
    """! @brief Perte débit-distorsion λ·MSE + bpp, sans synchronisation GPU.

    @param out    Sortie de forward_train().
    @param x      Lot d'entrée.
    @param lmbda  Poids de la distorsion.
    @return Triplet de scalaires (perte, MSE, bpp).
    """
    n_pix = x.shape[0] * x.shape[2] * x.shape[3]
    bpp = (-torch.log2(out["likelihoods"].clamp(min=LIKELIHOOD_MIN))).sum() / n_pix
    mse = F.mse_loss(out["x_hat"], x)
    return lmbda * mse + bpp, mse, bpp


@torch.no_grad()
def evaluate(model: Balle2017, images: list, device) -> tuple:
    """! @brief PSNR et débit moyens sur des images, avec vraie quantification.

    Chaque image est complétée par réplication à un multiple de 16 ; l'image reconstruite est
    rognée, écrêtée puis ré-arrondie sur 8 bits avant le PSNR. Le modèle reste en mode évaluation.
    @param model   Modèle.
    @param images  Tenseurs 1 x C x H x W dans [0, 1].
    @param device  Device du modèle.
    @return Couple (PSNR moyen en dB, débit moyen en bpp).
    """
    model.eval()
    psnrs, bpps = [], []
    for x in images:
        H, W = x.shape[-2:]
        m = Balle2017.pad_multiple
        Hp, Wp = math.ceil(H / m) * m, math.ceil(W / m) * m
        xp = F.pad(x, (0, Wp - W, 0, Hp - H), mode="replicate").to(device)
        out = forward_train(model, xp)
        x_hat = out["x_hat"][..., :H, :W].clamp(0, 1).cpu()
        x_hat = torch.round(x_hat * 255) / 255
        mse = F.mse_loss(x_hat, x).item()
        psnrs.append(100.0 if mse == 0 else 10 * math.log10(1.0 / mse))
        bits = (-torch.log2(out["likelihoods"].clamp(min=LIKELIHOOD_MIN))).sum().item()
        bpps.append(bits / (H * W))
    return float(np.mean(psnrs)), float(np.mean(bpps))


def _warn_16bit(n: int) -> None:
    """! @brief Avertit que @p n images à plus de 8 bits ont été mises à l'échelle (rien si @p n est nul)."""
    if n:
        _say(f"ATTENTION : {n} image(s) 16 bits ramenée(s) à 8 bits par mise à l'échelle linéaire "
             f"([0, max] vers [0, 255]).")


def build_patch_bank(paths_: list, n_patches: int, crop: int, in_ch: int, device, seed=None,
                     stats: dict | None = None) -> torch.Tensor:
    """! @brief Extrait une fois pour toutes des patchs uint8 (avec le sous-échantillonnage de l'article).

    Par image : un facteur d'échelle aléatoire entre max(crop / min(w, h), 0,3) et 0,75, puis plusieurs recadrages.
    Si le jeu compte plus d'images que de patchs, @p n_patches chemins sont tirés au hasard (reproductible
    avec @p seed) avant tout décodage : la banque ne dépasse jamais @p n_patches patchs. Les images à plus
    de 8 bits sont mises à l'échelle (to_8bit).
    @param paths_      Chemins des images.
    @param n_patches   Nombre de patchs visé (borne haute de la banque).
    @param crop        Côté des patchs.
    @param in_ch       1 (gris) ou 3 (RGB).
    @param device      Device de la banque.
    @param seed        Graine du tirage des chemins (défaut : générateur global de random).
    @param stats       Si fourni, reçoit le nombre d'images converties dans @c stats["converted"] et
                       l'avertissement 16 bits est laissé à l'appelant ; sinon il est écrit ici.
    @return Tenseur uint8 (n, C, crop, crop) sur @p device.
    @throws ConfigError si aucune image n'est fournie ou lisible.
    """
    if not paths_:
        raise ConfigError("aucune image d'entraînement")
    if len(paths_) > n_patches:
        paths_ = (random.Random(seed) if seed is not None else random).sample(list(paths_), n_patches)
    base, extra = divmod(n_patches, len(paths_))     # répartition exacte : n_patches au total, jamais plus
    converted = 0
    patches, t0 = [], time.time()
    for i, p in enumerate(paths_):
        try:
            img, was16 = to_8bit(Image.open(p))
            img = img.convert("L" if in_ch == 1 else "RGB")
            img.load()
            converted += was16
        except Exception as e:
            _say(f"ATTENTION : image illisible ignorée ({Path(p).name}) : {e}")
            continue
        w, h = img.size
        f_min = crop / min(w, h)
        if f_min <= 0.75:                       # un facteur aléatoire par image, puis plusieurs recadrages
            f = random.uniform(max(f_min, 0.3), 0.75)
            img = img.resize((max(crop, round(w * f)), max(crop, round(h * f))), Image.BICUBIC)
        elif f_min > 1:
            img = img.resize((max(w, crop), max(h, crop)), Image.BICUBIC)
        arr = np.asarray(img)
        arr = arr[None] if arr.ndim == 2 else arr.transpose(2, 0, 1)
        H, W = arr.shape[1:]
        for _ in range(base + (i < extra)):
            y0, x0 = random.randint(0, H - crop), random.randint(0, W - crop)
            patches.append(arr[:, y0:y0 + crop, x0:x0 + crop].copy())
    if not patches:
        raise ConfigError("aucune image d'entraînement lisible")
    if stats is not None:
        stats["converted"] = stats.get("converted", 0) + converted
    else:
        _warn_16bit(converted)
    bank = torch.from_numpy(np.stack(patches)).to(device)
    _say(f"{len(bank)} patchs {tuple(bank.shape[1:])} en {time.time() - t0:.0f} s, "
         f"{bank.numel() / 2 ** 20:.0f} Mo sur {bank.device}")
    return bank


def sample_batch(bank: torch.Tensor, batch_size: int) -> torch.Tensor:
    """! @brief Lot tiré avec remise dans la banque, converti en [0, 1], avec retournements aléatoires.

    @param bank        Banque uint8 de build_patch_bank().
    @param batch_size  Taille du lot.
    @return Tenseur float sur le device de la banque.
    """
    idx = torch.randint(len(bank), (batch_size,), device=bank.device)
    x = bank[idx].float().div_(255)
    if random.random() < 0.5:
        x = x.flip(-1)
    if random.random() < 0.5:
        x = x.flip(-2)
    return x


def checkpoint_path(cfg: TrainConfig, job_id=None, ckpt_dir=None) -> Path:
    """! @brief Chemin du checkpoint d'une configuration (motif reconnu par le banc d'essai).

    @param cfg       Configuration.
    @param job_id    Numéro du job (suffixe @c _run{N}), ou None (nom du notebook).
    @param ckpt_dir  Dossier (défaut config.CKPT_DIR, lu à l'appel).
    @return Chemin @c bls2017_c{in_ch}_N{N}_lambda{λ}[_run{job}].pt.
    """
    ckpt_dir = Path(ckpt_dir) if ckpt_dir is not None else paths.CKPT_DIR
    suffix = "" if job_id is None else f"_run{job_id}"
    return ckpt_dir / f"bls2017_c{cfg.in_ch}_N{cfg.N}_lambda{cfg.lmbda:g}{suffix}.pt"


def save_checkpoint(path: Path, model, optimizer, cfg: TrainConfig, history: dict, step: int,
                    run_info: dict) -> Path:
    """! @brief Écrit un checkpoint de façon atomique (fichier temporaire puis os.replace).

    Le contenu est relisible avec torch.load(weights_only=True). Sous Windows, os.replace est
    retenté si le fichier est ouvert en lecture (le banc d'essai).
    @param path       Fichier cible.
    @param model      Modèle.
    @param optimizer  Optimiseur.
    @param cfg        Configuration.
    @param history    Historique (clés du notebook).
    @param step       Pas atteint.
    @param run_info   Clé @c run (sans @c date, ajoutée ici).
    @return @p path.
    @throws PermissionError si le remplacement échoue après toutes les tentatives.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    run = {"job_id": run_info.get("job_id"), "name": str(run_info.get("name", "")),
           "author": str(run_info.get("author", "")), "commit": str(run_info.get("commit", "")),
           "date": datetime.now().isoformat(timespec="seconds")}
    data = {"state_dict": model.state_dict(), "cfg": cfg.to_cfg_dict(),
            "history": {k: list(v) for k, v in history.items()}, "step": int(step),
            "optimizer": optimizer.state_dict(), "run": run}
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        torch.save(data, tmp)
        for attempt in range(REPLACE_RETRIES + 1):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == REPLACE_RETRIES:
                    raise
                time.sleep(REPLACE_DELAY_S)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def load_checkpoint(path: Path, map_location="cpu") -> dict:
    """! @brief Lit un checkpoint avec torch.load(weights_only=True).

    @param path          Fichier.
    @param map_location  Device de destination des tenseurs.
    @return Le dictionnaire du checkpoint.
    @throws FileNotFoundError si le fichier est absent.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"checkpoint introuvable : {path}")
    return torch.load(path, map_location=map_location, weights_only=True)


def _read_ref(gitdir: Path, ref: str) -> str:
    """! @brief Empreinte d'une référence Git (fichier de la référence, puis packed-refs), ou chaîne vide."""
    dirs = [gitdir]
    common = gitdir / "commondir"
    if common.is_file():
        dirs.append((gitdir / common.read_text().strip()).resolve())
    for d in dirs:
        f = d / ref
        if f.is_file():
            return f.read_text().strip()
    for d in dirs:
        packed = d / "packed-refs"
        if packed.is_file():
            for line in packed.read_text().splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0]
    return ""


def git_commit() -> str:
    """! @brief Empreinte (12 caractères) du commit courant, lue sans sous-processus.

    Gère le dossier @c .git, le fichier @c .git d'un worktree, HEAD détaché et packed-refs.
    @return 12 caractères hexadécimaux, ou chaîne vide en cas d'échec (l'image Docker n'a pas git).
    """
    try:
        git = paths.ROOT / ".git"
        if git.is_file():
            text = git.read_text().strip()
            if not text.startswith("gitdir:"):
                return ""
            git = (paths.ROOT / text[len("gitdir:"):].strip()).resolve()
        head = (git / "HEAD").read_text().strip()
        sha = _read_ref(git, head[5:].strip()) if head.startswith("ref:") else head
        return sha[:12] if re.fullmatch(r"[0-9a-f]{40,64}", sha) else ""
    except Exception:
        return ""


def _warn_callback(step: int, exc: Exception) -> None:
    """! @brief Avertit qu'un rappel de métriques a échoué (l'entraînement continue)."""
    _say(f"ATTENTION : métriques non transmises au pas {step} : {exc}")


def _check_resume(ck: dict, cfg: TrainConfig, run_info: dict, ckpt_path: Path) -> None:
    """! @brief Refuse de reprendre un checkpoint qui n'est pas celui de ce run.

    @throws ConfigError si la clé @c run manque, si le job ou le nom diffère, ou si l'architecture ou λ diffère.
    """
    hint = f" Déplacez ou supprimez le fichier pour repartir de zéro : {ckpt_path}"
    run = ck.get("run")
    if not isinstance(run, dict):
        raise ConfigError(f"checkpoint existant sans clé « run » (créé par le notebook ?) : reprise refusée.{hint}")
    if run.get("job_id") != run_info["job_id"] or run.get("name") != run_info["name"]:
        raise ConfigError(f"checkpoint existant d'un autre run (job {run.get('job_id')}, « {run.get('name')} ») : "
                          f"reprise refusée.{hint}")
    old, new = ck.get("cfg", {}), cfg.to_cfg_dict()
    for k in ARCH_KEYS:
        if k == "gdn_last":
            same = old.get(k, True) == new[k]
        else:
            same = k in old and float(old[k]) == float(new[k])
        if not same:
            raise ConfigError(f"checkpoint existant incompatible ({k} : {old.get(k)!r} contre {new[k]!r}) : "
                              f"reprise refusée.{hint}")


def _pick_data(cfg: TrainConfig) -> tuple:
    """! @brief Images d'entraînement et d'évaluation, avec les replis de la conception (§ 2.3.1).

    @return Triplet (chemins d'entraînement, chemins d'évaluation, source de l'évaluation).
    @throws ConfigError si le dossier d'entraînement désigné est vide, ou si les deux jeux sont vides.
    """
    train_dir, eval_dir = resolve_path(cfg.train_dir), resolve_path(cfg.eval_dir)
    train_paths = list_images(train_dir) if train_dir.exists() else []
    eval_paths = list_images(eval_dir) if eval_dir.exists() else []
    eval_is_kodak = eval_dir.resolve() == paths.KODAK_DIR.resolve()
    if not eval_paths and eval_is_kodak:
        eval_paths = download_kodak(paths.KODAK_DIR)
    if not train_paths:
        if train_dir.resolve() != paths.TRAIN_DIR.resolve():
            raise ConfigError(f"aucune image dans {train_dir}")
        if not eval_paths:
            raise ConfigError(f"aucune image dans {train_dir} ni dans {eval_dir}")
        _say("ATTENTION : entraînement sur les images d'évaluation : scores optimistes.")
        train_paths = list(eval_paths)
    if eval_paths:
        return train_paths, eval_paths, "kodak" if eval_is_kodak else "eval"
    _say(f"ATTENTION : images d'évaluation absentes ({eval_dir}) : évaluation sur "
         f"{min(cfg.n_eval_images, len(train_paths))} images du jeu d'entraînement (scores optimistes).")
    return train_paths, train_paths, "train"


def train(cfg: TrainConfig, ckpt_path: Path, *, run_info: dict | None = None, device=None,
          should_stop: Callable[[], bool] | None = None,
          on_metrics: Callable[[int, dict], None] | None = None) -> TrainResult:
    """! @brief Entraîne (ou reprend) un modèle Ballé 2017 et sauvegarde ses checkpoints.

    Déroulement (docs/conception_entrainement.md, § 2.3.1) : données, reprise éventuelle, boucle
    Adam, puis journal, évaluation et sauvegarde périodiques. @p should_stop est appelé à la fin de
    chaque pas ; s'il renvoie vrai, le checkpoint est sauvegardé au pas courant.
    @param cfg          Configuration.
    @param ckpt_path    Checkpoint à écrire (et à reprendre s'il existe).
    @param run_info     Clé @c run du checkpoint sans @c date (défaut : job None, nom et auteur de @p cfg, commit courant).
    @param device       Device (défaut : select_device("auto")).
    @param should_stop  Appelable sans argument, vrai pour demander l'arrêt.
    @param on_metrics   Appelable (pas, dictionnaire @c kind=train ou @c kind=eval).
    @return Le résultat, de statut @c "done" ou @c "cancelled".
    @throws ConfigError si les données manquent ou si le checkpoint existant est incompatible.
    """
    ckpt_path = Path(ckpt_path)
    run_info = {"job_id": None, "name": cfg.name, "author": cfg.author, "commit": git_commit(),
                **(run_info or {})}
    device = device or select_device("auto")

    ck = None
    if ckpt_path.exists():
        ck = load_checkpoint(ckpt_path)
        _check_resume(ck, cfg, run_info, ckpt_path)
        if ck["step"] >= cfg.steps:
            _say(f"Checkpoint déjà complet (pas {ck['step']}) : {ckpt_path}")
            return TrainResult("done", int(ck["step"]), ckpt_path,
                               {k: list(ck["history"].get(k, [])) for k in HISTORY_KEYS}, None)

    train_paths, eval_paths, source = _pick_data(cfg)
    eval_paths = sorted(eval_paths)[:cfg.n_eval_images]
    stats16 = {"converted": 0}
    eval_images = []
    for p in eval_paths:
        img, was16 = to_8bit(Image.open(p))
        stats16["converted"] += was16
        eval_images.append(to_tensor(convert(img, cfg.mode)))
    _say(f"Entraînement : {cfg.train_dir} ({len(train_paths)} images) ; "
         f"évaluation : {cfg.eval_dir} ({len(eval_images)} images, source {source})")

    start = 1
    if ck is None:
        seed_everything(cfg.seed)
    else:
        start = int(ck["step"]) + 1
        seed_everything(cfg.seed + int(ck["step"]))
    bank = build_patch_bank(train_paths, cfg.n_patches, cfg.crop, cfg.in_ch, device, seed=cfg.seed, stats=stats16)
    _warn_16bit(stats16["converted"])

    model = Balle2017(cfg.in_ch, cfg.N, cfg.gdn_last)
    if ck is None:
        init_reference_weights(model)
    else:
        model.load_state_dict(ck["state_dict"])
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    hist = {k: [] for k in HISTORY_KEYS}
    if ck is not None:
        opt.load_state_dict(ck["optimizer"])
        hist = {k: list(ck["history"].get(k, [])) for k in HISTORY_KEYS}
        _say(f"Reprise au pas {ck['step']} : {ckpt_path}")

    def emit(step, data):
        """! @brief Transmet des métriques sans jamais interrompre l'entraînement."""
        if on_metrics is not None:
            try:
                on_metrics(step, data)
            except Exception as e:
                _warn_callback(step, e)

    def save(step, fatal=True):
        """! @brief Sauvegarde le checkpoint au pas donné ; si @p fatal est faux, un échec est signalé puis ignoré."""
        try:
            save_checkpoint(ckpt_path, model, opt, cfg, hist, step, run_info)
        except OSError as e:
            if fatal:
                raise
            _say(f"ATTENTION : sauvegarde périodique impossible au pas {step} ({e}) : "
                 f"l'entraînement continue, nouvelle tentative à la prochaine sauvegarde.")
            return
        _say(f"Checkpoint sauvegardé (pas {step}) : {ckpt_path}")

    decay_step = int(cfg.lr_decay_at * cfg.steps)
    acc = torch.zeros(3, device=device)      # perte, MSE, bpp cumulés SUR le device : pas de synchro à chaque pas
    n, t0, t_log = 0, time.time(), time.time()
    final_eval, status, step = None, "done", start - 1
    for step in range(start, cfg.steps + 1):
        model.train()
        if step == decay_step:
            for g in opt.param_groups:
                g["lr"] *= 0.1
        x = sample_batch(bank, cfg.batch_size)
        out = forward_train(model, x)
        loss, mse, bpp = rd_loss(out, x, cfg.lmbda)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), cfg.grad_clip)
        opt.step()
        acc += torch.stack([loss.detach(), mse.detach(), bpp.detach()])
        n += 1

        last = step == cfg.steps
        if not last and should_stop is not None and should_stop():
            save(step)
            status = "cancelled"
            break

        if step % cfg.log_every == 0 or last:
            l, m, b = (acc / n).tolist()           # une seule synchro tous les log_every pas
            acc.zero_()
            now = time.time()
            it_s = n / max(now - t_log, 1e-9)
            t_log, n = now, 0
            eta = (cfg.steps - step) / max(it_s, 1e-9)
            psnr = 10 * math.log10(1 / max(m, 1e-10))
            hist["step"].append(step)
            hist["loss"].append(l)
            hist["bpp"].append(b)
            hist["psnr"].append(psnr)
            _say(f"step {step:>7}/{cfg.steps} | {it_s:5.1f} it/s | reste ~{eta / 60:4.1f} min | "
                 f"loss {l:.4f} | {b:.3f} bpp | {psnr:.2f} dB")
            emit(step, {"kind": "train", "loss": l, "mse": m, "bpp": b, "psnr": psnr, "it_s": it_s,
                        "eta_s": eta, "lr": float(opt.param_groups[0]["lr"]), "steps": cfg.steps})
        if step % cfg.eval_every == 0 or last:
            p_ev, b_ev = evaluate(model, eval_images, device)
            hist["eval_step"].append(step)
            hist["eval_psnr"].append(p_ev)
            hist["eval_bpp"].append(b_ev)
            final_eval = {"psnr": p_ev, "bpp": b_ev, "source": source}
            _say(f"   -> {source} (arrondi) : {b_ev:.3f} bpp, {p_ev:.2f} dB")
            emit(step, {"kind": "eval", "bpp": b_ev, "psnr": p_ev, "source": source,
                        "n_images": len(eval_images)})
        if step % cfg.save_every == 0 or last:
            save(step, fatal=last)
    if status == "done":
        _say(f"Terminé en {(time.time() - t0) / 60:.1f} min : {ckpt_path}")
    return TrainResult(status, step, ckpt_path, hist, final_eval)

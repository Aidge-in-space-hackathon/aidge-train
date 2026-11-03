"""! @brief Codage entropique : codeur arithmétique 32 bits (repris du notebook) et tables de fréquences.

Chaque symbole est codé avec sa propre table (un canal pour un prior factorisé, un niveau
d'échelle pour un prior gaussien). Les valeurs hors du support d'une table passent par un symbole
d'échappement suivi de BYPASS_BITS bits bruts (codage zigzag), si bien que toute valeur entière
de moins de 2^19 en valeur absolue reste codable.

encode_symbols() et decode_symbols() passent par des noyaux compilés par numba (un seul thread), qui
produisent exactement le même flux que les classes Python ci-dessous. Ces classes restent la référence
et servent de repli si numba n'est pas installé.
"""
import bisect
from dataclasses import dataclass, field

import numpy as np

try:
    import numba
except ImportError:                     # repli : codeur Python pur, même flux
    numba = None

## Précision des fréquences, en bits : chaque table somme à 2^PRECISION.
PRECISION = 16
## Nombre de bits bruts écrits après un symbole d'échappement.
BYPASS_BITS = 20
## Taille des registres du codeur arithmétique, en bits.
STATE_BITS = 32
## Bornes de l'intervalle du codeur : 2^STATE_BITS, sa moitié et son quart.
FULL, HALF, QUARTER = 1 << STATE_BITS, 1 << (STATE_BITS - 1), 1 << (STATE_BITS - 2)
## Masque des STATE_BITS bits de poids faible.
MASK = FULL - 1
## Une valeur échappée v est codable si -ZIGZAG_LIMIT <= v < ZIGZAG_LIMIT (zigzag sur BYPASS_BITS bits).
ZIGZAG_LIMIT = 1 << (BYPASS_BITS - 1)
## True si numba est importable : encode_symbols() et decode_symbols() utilisent alors les noyaux compilés.
NUMBA = numba is not None
## Plus grand total de fréquences des noyaux : rng * cum (rng <= 2^STATE_BITS) doit tenir sur 63 bits.
NUMBA_MAX_TOTAL = 1 << 30


class BitWriter:
    """! @brief Écriture bit à bit dans un tampon d'octets (bit de poids fort en premier)."""

    def __init__(self):
        """! @brief Crée un tampon vide."""
        ## Octets complets, octet en cours de remplissage et nombre de bits qu'il contient.
        self.buf, self.cur, self.n = bytearray(), 0, 0

    def write(self, bit):
        """! @brief Ajoute un bit (0 ou 1)."""
        self.cur = (self.cur << 1) | bit
        self.n += 1
        if self.n == 8:
            self.buf.append(self.cur)
            self.cur, self.n = 0, 0

    def getvalue(self):
        """! @brief Octets écrits, le dernier complété par des zéros."""
        if self.n:
            return bytes(self.buf) + bytes([self.cur << (8 - self.n)])
        return bytes(self.buf)


class BitReader:
    """! @brief Lecture bit à bit d'un tampon d'octets ; renvoie des zéros au-delà de la fin."""

    def __init__(self, data):
        """! @brief Commence la lecture au premier bit.

        @param data  Octets à lire.
        """
        ## Octets à lire et position du prochain bit.
        self.data, self.pos = data, 0

    def read(self):
        """! @brief Bit suivant (0 ou 1)."""
        byte = self.pos >> 3
        if byte >= len(self.data):
            return 0
        bit = (self.data[byte] >> (7 - (self.pos & 7))) & 1
        self.pos += 1
        return bit


class ArithmeticEncoder:
    """! @brief Codeur arithmétique entier sur STATE_BITS bits (méthode des bits en attente)."""

    def __init__(self):
        """! @brief Intervalle initial [0, MASK]."""
        ## Bornes de l'intervalle courant, bits en attente (intervalle autour de 1/2) et sortie.
        self.low, self.high, self.pending, self.out = 0, MASK, 0, BitWriter()

    def encode(self, cum_lo, cum_hi, total):
        """! @brief Code un symbole d'intervalle [cum_lo, cum_hi) dans [0, total).

        @param cum_lo  Fréquence cumulée avant le symbole.
        @param cum_hi  Fréquence cumulée après le symbole (> cum_lo).
        @param total   Somme des fréquences de la table.
        """
        rng = self.high - self.low + 1
        self.high = self.low + (rng * cum_hi) // total - 1
        self.low = self.low + (rng * cum_lo) // total
        while ((self.low ^ self.high) & HALF) == 0:          # bit de tête décidé : on l'écrit
            bit = self.low >> (STATE_BITS - 1)
            self.out.write(bit)
            for _ in range(self.pending):
                self.out.write(bit ^ 1)
            self.pending = 0
            self.low = (self.low << 1) & MASK
            self.high = ((self.high << 1) & MASK) | 1
        while (self.low & ~self.high & QUARTER) != 0:        # intervalle coincé autour de 1/2
            self.pending += 1
            self.low = (self.low << 1) ^ HALF
            self.high = ((self.high ^ HALF) << 1) | HALF | 1

    def finish(self):
        """! @brief Termine le flux.

        @return Flux binaire complet (bytes).
        """
        self.out.write(1)
        return self.out.getvalue()


class ArithmeticDecoder:
    """! @brief Décodeur symétrique de ArithmeticEncoder."""

    def __init__(self, data):
        """! @brief Charge les STATE_BITS premiers bits du flux.

        @param data  Flux produit par ArithmeticEncoder.finish().
        """
        ## Lecteur du flux.
        self.inp = BitReader(data)
        ## Bornes de l'intervalle courant et fenêtre de STATE_BITS bits lue dans le flux.
        self.low, self.high, self.code = 0, MASK, 0
        for _ in range(STATE_BITS):
            self.code = (self.code << 1) | self.inp.read()

    def decode(self, cum, total):
        """! @brief Décode un symbole.

        @param cum    Fréquences cumulées de la table (cum[0] = 0, cum[-1] = total).
        @param total  Somme des fréquences.
        @return Indice s du symbole, tel que cum[s] <= valeur < cum[s + 1].
        """
        rng = self.high - self.low + 1
        value = ((self.code - self.low + 1) * total - 1) // rng
        s = bisect.bisect_right(cum, value) - 1
        self.high = self.low + (rng * cum[s + 1]) // total - 1
        self.low = self.low + (rng * cum[s]) // total
        while ((self.low ^ self.high) & HALF) == 0:
            self.code = ((self.code << 1) & MASK) | self.inp.read()
            self.low = (self.low << 1) & MASK
            self.high = ((self.high << 1) & MASK) | 1
        while (self.low & ~self.high & QUARTER) != 0:
            self.code = (self.code & HALF) | ((self.code << 1) & (MASK >> 1)) | self.inp.read()
            self.low = (self.low << 1) ^ HALF
            self.high = ((self.high ^ HALF) << 1) | HALF | 1
        return s


@dataclass
class CdfTables:
    """! @brief Ensemble de tables de fréquences cumulées, prêtes pour le codeur arithmétique."""

    ## Valeur du premier symbole de chaque table.
    lo: list
    ## Fréquences cumulées de chaque table (listes d'entiers) ; le dernier symbole est l'échappement.
    cum: list
    ## Somme des fréquences, commune à toutes les tables (2^PRECISION).
    total: int
    ## Fréquences cumulées de toutes les tables mises bout à bout (int64), pour les noyaux numba.
    flat_cum: np.ndarray = field(init=False, repr=False, compare=False)
    ## Début de chaque table dans flat_cum, suivi de la fin de la dernière (int64, len(lo) + 1 valeurs).
    offsets: np.ndarray = field(init=False, repr=False, compare=False)
    ## lo en tableau int64, pour les noyaux numba.
    flat_lo: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        """! @brief Aplatit les tables une fois pour toutes, au format des noyaux numba."""
        sizes = np.array([len(c) for c in self.cum], dtype=np.int64)
        self.offsets = np.zeros(len(sizes) + 1, dtype=np.int64)
        self.offsets[1:] = np.cumsum(sizes)
        self.flat_cum = np.array([f for c in self.cum for f in c], dtype=np.int64)
        self.flat_lo = np.asarray(self.lo, dtype=np.int64).reshape(-1)

    def rom_bytes(self):
        """! @brief Taille mémoire des tables à embarquer : fréquences sur 16 bits + une borne par table.

        @return Nombre d'octets.
        """
        return 2 * sum(len(c) for c in self.cum) + 4 * len(self.lo)


def make_tables(pmfs, los, precision=PRECISION, tail_mass=1e-9):
    """! @brief Convertit des lois discrètes en tables de fréquences entières.

    Les queues de probabilité inférieure à @p tail_mass sont coupées et renvoyées vers le symbole
    d'échappement ; chaque symbole conservé reçoit une fréquence d'au moins 1.
    @param pmfs       Liste de lois : pmfs[i][j] = P(valeur = los[i] + j).
    @param los        Première valeur du support de chaque loi.
    @param precision  Les fréquences de chaque table somment à 2^precision.
    @param tail_mass  Masse en dessous de laquelle une queue est coupée.
    @return CdfTables.
    @throws ValueError si une table a trop de symboles pour la précision demandée.
    """
    total = 1 << precision
    lo_out, cums = [], []
    for p, lo in zip(pmfs, los):
        p = np.asarray(p, dtype=np.float64)
        left, right = np.cumsum(p), np.cumsum(p[::-1])[::-1]
        i0 = int(np.argmax(left > tail_mass))
        i1 = len(p) - 1 - int(np.argmax(right[::-1] > tail_mass))
        support = p[i0:i1 + 1]
        escape = max(1.0 - support.sum(), 1e-12)
        probs = np.append(support, escape)
        n = len(probs)
        if n >= total:
            raise ValueError(f"table de {n} symboles trop grande pour une précision de {precision} bits")
        freq = np.floor(probs / probs.sum() * (total - n)).astype(np.int64) + 1   # au moins 1
        freq[np.argmax(freq)] += total - freq.sum()                              # somme exacte
        lo_out.append(int(lo) + i0)
        cums.append([0] + np.cumsum(freq).tolist())
    return CdfTables(lo_out, cums, total)


def encode_symbols(values, table_idx, tables):
    """! @brief Code une suite d'entiers, chacun avec sa table.

    Passe par le noyau numba quand il est disponible, par encode_symbols_python() sinon : le flux est
    le même octet par octet.
    @param values     Tableau d'entiers (aplati dans l'ordre de numpy).
    @param table_idx  Indice de table de chaque valeur, même nombre d'éléments.
    @param tables     CdfTables.
    @return Couple (flux binaire, nombre de valeurs échappées).
    @throws ValueError si une valeur échappée dépasse BYPASS_BITS bits après codage zigzag.
    @throws IndexError si un indice de table est hors de [0, nombre de tables) (noyau numba).
    """
    values, table_idx = np.asarray(values).ravel(), np.asarray(table_idx).ravel()
    if not (_numba_ready(tables, values, table_idx) and len(values) == len(table_idx)):
        return encode_symbols_python(values, table_idx, tables)
    values = np.ascontiguousarray(values, dtype=np.int64)
    buf = np.empty(_stream_capacity(len(values), tables.total), dtype=np.uint8)
    n, n_esc, bad = _encode_kernel(values, _checked_index(table_idx, tables), tables.flat_cum, tables.offsets,
                                   tables.flat_lo, int(tables.total), buf)
    if bad >= 0:
        raise ValueError(f"valeur {values[bad]} hors de portée du codage brut sur {BYPASS_BITS} bits")
    return buf[:n].tobytes(), int(n_esc)


def decode_symbols(data, table_idx, tables):
    """! @brief Décode un flux de encode_symbols().

    Passe par le noyau numba quand il est disponible, par decode_symbols_python() sinon.
    @param data       Flux binaire.
    @param table_idx  Les mêmes indices de table qu'à l'encodage.
    @param tables     Les mêmes CdfTables qu'à l'encodage.
    @return Tableau int64 aplati des valeurs décodées.
    @throws IndexError si un indice de table est hors de [0, nombre de tables) (noyau numba).
    """
    table_idx = np.asarray(table_idx).ravel()
    if not _numba_ready(tables, table_idx):
        return decode_symbols_python(data, table_idx, tables)
    stream = np.frombuffer(bytes(data), dtype=np.uint8)        # toujours en lecture seule : une seule signature
    return _decode_kernel(stream, _checked_index(table_idx, tables), tables.flat_cum, tables.offsets,
                          tables.flat_lo, int(tables.total))


## True une fois les noyaux compilés (ou chargés du cache disque de numba) par warmup().
_WARM = False


def warmup():
    """! @brief Compile les noyaux numba, ou les charge du cache disque, avant toute mesure de temps.

    Appelée par aidge_in_space.coding.LatentCoding.tables(), hors chronométrage. Ne fait rien sans
    numba ni après le premier appel. Les types des arguments sont ceux des vrais appels (int64,
    flux en lecture seule), si bien qu'aucune compilation n'a lieu ensuite.
    """
    global _WARM
    if _WARM or not NUMBA:
        return
    tables = make_tables([[0.25, 0.5, 0.25]], [-1])
    values = np.array([0, 1, -1, 7], dtype=np.int64)             # 7 : hors support, échappé
    idx = np.zeros(len(values), dtype=np.int64)
    decode_symbols(encode_symbols(values, idx, tables)[0], idx, tables)
    _WARM = True


def _numba_ready(tables, *arrays):
    """! @brief Indique si les noyaux numba peuvent traiter ces tables et ces tableaux.

    @param tables  CdfTables.
    @param arrays  Tableaux numpy d'entrée, qui doivent être entiers (sinon : codeur Python).
    @return True si numba est disponible, le total assez petit pour 63 bits et les tableaux entiers.
    """
    return (NUMBA and 0 < tables.total <= NUMBA_MAX_TOTAL
            and all(a.dtype.kind in "iu" for a in arrays))


def _checked_index(table_idx, tables):
    """! @brief Indices de table en int64 contigus, vérifiés (les noyaux ne contrôlent pas les bornes).

    @param table_idx  Tableau d'indices aplati.
    @param tables     CdfTables.
    @return Tableau int64 contigu.
    @throws IndexError si un indice est hors de [0, nombre de tables).
    """
    idx = np.ascontiguousarray(table_idx, dtype=np.int64)
    if idx.size and (idx.min() < 0 or idx.max() >= len(tables.offsets) - 1):
        raise IndexError(f"indice de table hors de [0, {len(tables.offsets) - 1})")
    return idx


def _stream_capacity(n, total):
    """! @brief Taille de tampon suffisante pour le flux de n valeurs.

    Après renormalisation, l'intervalle couvre plus d'un quart de 2^STATE_BITS : un symbole de
    fréquence 1/total fait sortir au plus bit_length(total) + 2 bits, un bit brut au plus 3.
    @param n      Nombre de valeurs à coder.
    @param total  Somme des fréquences des tables.
    @return Nombre d'octets.
    """
    bits_per_value = int(total).bit_length() + 3 + 3 * BYPASS_BITS
    return (n * bits_per_value + 1) // 8 + 2


def _jit(fn):
    """! @brief Compile @p fn avec numba (mode nopython, cache disque, un seul thread).

    Sans numba, rend @p fn telle quelle : les noyaux ne sont alors jamais appelés.
    @param fn  Fonction à compiler.
    @return Fonction compilée, ou @p fn.
    """
    return numba.njit(cache=True)(fn) if numba is not None else fn


@_jit
def _put_bit(buf, w, bit):
    """! @brief BitWriter.write() dans un noyau.

    @param buf  Tampon de sortie (uint8), dimensionné par _stream_capacity().
    @param w    État de l'écriture : [octets complets, octet en cours, nombre de bits de cet octet].
    @param bit  Bit à écrire (0 ou 1).
    """
    w[1] = (w[1] << 1) | bit
    w[2] += 1
    if w[2] == 8:
        buf[w[0]] = w[1]
        w[0] += 1
        w[1] = 0
        w[2] = 0


@_jit
def _encode_step(buf, w, low, high, pending, cum_lo, cum_hi, total):
    """! @brief ArithmeticEncoder.encode() dans un noyau.

    @param buf      Tampon de sortie.
    @param w        État de l'écriture (voir _put_bit()).
    @param low      Borne basse de l'intervalle.
    @param high     Borne haute de l'intervalle.
    @param pending  Bits en attente.
    @param cum_lo   Fréquence cumulée avant le symbole.
    @param cum_hi   Fréquence cumulée après le symbole.
    @param total    Somme des fréquences.
    @return Nouvel état (low, high, pending).
    """
    rng = high - low + 1
    high = low + (rng * cum_hi) // total - 1
    low = low + (rng * cum_lo) // total
    while ((low ^ high) & HALF) == 0:
        bit = low >> (STATE_BITS - 1)
        _put_bit(buf, w, bit)
        for _ in range(pending):
            _put_bit(buf, w, bit ^ 1)
        pending = 0
        low = (low << 1) & MASK
        high = ((high << 1) & MASK) | 1
    while (low & ~high & QUARTER) != 0:
        pending += 1
        low = (low << 1) ^ HALF
        high = ((high ^ HALF) << 1) | HALF | 1
    return low, high, pending


@_jit
def _encode_kernel(values, table_idx, cum, offsets, lo, total, buf):
    """! @brief Boucle de encode_symbols_python() sur les tables aplaties de CdfTables.

    @param values     Valeurs (int64).
    @param table_idx  Indice de table de chaque valeur (int64, vérifiés).
    @param cum        CdfTables.flat_cum.
    @param offsets    CdfTables.offsets.
    @param lo         CdfTables.flat_lo.
    @param total      Somme des fréquences.
    @param buf        Tampon de sortie, dimensionné par _stream_capacity().
    @return Triplet (octets écrits dans buf, valeurs échappées, indice de la première valeur hors de
            portée du codage brut ou -1).
    """
    w = np.zeros(3, dtype=np.int64)
    low, high, pending = 0, MASK, 0
    n_esc = 0
    for i in range(values.shape[0]):
        t = table_idx[i]
        base = offsets[t]
        esc = offsets[t + 1] - base - 2
        v = values[i]
        s = v - lo[t]
        if s >= 0 and s < esc:
            low, high, pending = _encode_step(buf, w, low, high, pending, cum[base + s], cum[base + s + 1], total)
        else:
            if v < -ZIGZAG_LIMIT or v >= ZIGZAG_LIMIT:
                return 0, n_esc, i
            low, high, pending = _encode_step(buf, w, low, high, pending, cum[base + esc], cum[base + esc + 1],
                                              total)
            n_esc += 1
            z = 2 * v if v >= 0 else -2 * v - 1                 # zigzag
            for b in range(BYPASS_BITS - 1, -1, -1):
                bit = (z >> b) & 1
                low, high, pending = _encode_step(buf, w, low, high, pending, bit, bit + 1, 2)
    _put_bit(buf, w, 1)                                         # ArithmeticEncoder.finish()
    n = w[0]
    if w[2]:
        buf[n] = w[1] << (8 - w[2])
        n += 1
    return n, n_esc, -1


@_jit
def _read_bit(data, pos):
    """! @brief BitReader.read() dans un noyau.

    @param data  Flux (uint8).
    @param pos   Position du bit.
    @return Bit lu, 0 au-delà de la fin.
    """
    byte = pos >> 3
    if byte >= data.shape[0]:
        return 0
    return (data[byte] >> (7 - (pos & 7))) & 1


@_jit
def _decode_step(data, low, high, code, pos, cum_lo, cum_hi, total):
    """! @brief Fin de ArithmeticDecoder.decode() dans un noyau, une fois le symbole trouvé.

    @param data    Flux (uint8).
    @param low     Borne basse de l'intervalle.
    @param high    Borne haute de l'intervalle.
    @param code    Fenêtre de STATE_BITS bits lue dans le flux.
    @param pos     Position du prochain bit à lire.
    @param cum_lo  Fréquence cumulée avant le symbole décodé.
    @param cum_hi  Fréquence cumulée après le symbole décodé.
    @param total   Somme des fréquences.
    @return Nouvel état (low, high, code, pos).
    """
    rng = high - low + 1
    high = low + (rng * cum_hi) // total - 1
    low = low + (rng * cum_lo) // total
    while ((low ^ high) & HALF) == 0:
        code = ((code << 1) & MASK) | _read_bit(data, pos)
        pos += 1
        low = (low << 1) & MASK
        high = ((high << 1) & MASK) | 1
    while (low & ~high & QUARTER) != 0:
        code = (code & HALF) | ((code << 1) & (MASK >> 1)) | _read_bit(data, pos)
        pos += 1
        low = (low << 1) ^ HALF
        high = ((high ^ HALF) << 1) | HALF | 1
    return low, high, code, pos


@_jit
def _decode_kernel(data, table_idx, cum, offsets, lo, total):
    """! @brief Boucle de decode_symbols_python() sur les tables aplaties de CdfTables.

    @param data       Flux (uint8).
    @param table_idx  Indice de table de chaque valeur (int64, vérifiés).
    @param cum        CdfTables.flat_cum.
    @param offsets    CdfTables.offsets.
    @param lo         CdfTables.flat_lo.
    @param total      Somme des fréquences.
    @return Tableau int64 des valeurs décodées.
    """
    out = np.empty(table_idx.shape[0], dtype=np.int64)
    low, high, code, pos = 0, MASK, 0, 0
    for _ in range(STATE_BITS):
        code = (code << 1) | _read_bit(data, pos)
        pos += 1
    for i in range(table_idx.shape[0]):
        t = table_idx[i]
        base = offsets[t]
        size = offsets[t + 1] - base
        value = ((code - low + 1) * total - 1) // (high - low + 1)
        a, b = 0, size - 1                      # bisect_right(cum, value) - 1 : cum[a] <= value < cum[b]
        while b - a > 1:
            m = (a + b) >> 1
            if cum[base + m] <= value:
                a = m
            else:
                b = m
        low, high, code, pos = _decode_step(data, low, high, code, pos, cum[base + a], cum[base + a + 1], total)
        if a < size - 2:
            out[i] = a + lo[t]
        else:
            z = 0
            for _ in range(BYPASS_BITS):
                bit = 1 if ((code - low + 1) * 2 - 1) // (high - low + 1) >= 1 else 0
                low, high, code, pos = _decode_step(data, low, high, code, pos, bit, bit + 1, 2)
                z = (z << 1) | bit
            out[i] = z // 2 if z % 2 == 0 else -(z + 1) // 2
    return out


def encode_symbols_python(values, table_idx, tables):
    """! @brief Codeur de référence en Python pur, repli de encode_symbols() sans numba.

    @param values     Tableau d'entiers (aplati dans l'ordre de numpy).
    @param table_idx  Indice de table de chaque valeur, même nombre d'éléments.
    @param tables     CdfTables.
    @return Couple (flux binaire, nombre de valeurs échappées).
    @throws ValueError si une valeur échappée dépasse BYPASS_BITS bits après codage zigzag.
    """
    enc = ArithmeticEncoder()
    total, los, cums = tables.total, tables.lo, tables.cum
    n_esc = 0
    for v, t in zip(np.asarray(values).ravel().tolist(), np.asarray(table_idx).ravel().tolist()):
        cum = cums[t]
        esc = len(cum) - 2
        s = v - los[t]
        if 0 <= s < esc:
            enc.encode(cum[s], cum[s + 1], total)
        else:
            enc.encode(cum[esc], cum[esc + 1], total)
            n_esc += 1
            z = 2 * v if v >= 0 else -2 * v - 1                 # zigzag
            if z >= 1 << BYPASS_BITS:
                raise ValueError(f"valeur {v} hors de portée du codage brut sur {BYPASS_BITS} bits")
            for b in range(BYPASS_BITS - 1, -1, -1):
                bit = (z >> b) & 1
                enc.encode(bit, bit + 1, 2)
    return enc.finish(), n_esc


def decode_symbols_python(data, table_idx, tables):
    """! @brief Décodeur de référence en Python pur, repli de decode_symbols() sans numba.

    @param data       Flux binaire.
    @param table_idx  Les mêmes indices de table qu'à l'encodage.
    @param tables     Les mêmes CdfTables qu'à l'encodage.
    @return Tableau int64 aplati des valeurs décodées.
    """
    dec = ArithmeticDecoder(data)
    total, los, cums = tables.total, tables.lo, tables.cum
    idx = np.asarray(table_idx).ravel().tolist()
    out = np.empty(len(idx), dtype=np.int64)
    for i, t in enumerate(idx):
        cum = cums[t]
        s = dec.decode(cum, total)
        if s < len(cum) - 2:
            out[i] = s + los[t]
        else:
            z = 0
            for _ in range(BYPASS_BITS):
                z = (z << 1) | dec.decode([0, 1, 2], 2)
            out[i] = z // 2 if z % 2 == 0 else -(z + 1) // 2
    return out

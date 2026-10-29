"""! @brief Codage entropique : codeur arithmétique 32 bits (repris du notebook) et tables de fréquences.

Chaque symbole est codé avec sa propre table (un canal pour un prior factorisé, un niveau
d'échelle pour un prior gaussien). Les valeurs hors du support d'une table passent par un symbole
d'échappement suivi de BYPASS_BITS bits bruts (codage zigzag), si bien que toute valeur entière
de moins de 2^19 en valeur absolue reste codable.
"""
import bisect
from dataclasses import dataclass

import numpy as np

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


def decode_symbols(data, table_idx, tables):
    """! @brief Décode un flux de encode_symbols().

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

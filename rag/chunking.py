"""! @brief Découpage des documents en passages : par section, environ 800 tokens, 100 de chevauchement.

Chaque format est d'abord lu en blocs (section, page, texte) : titres Markdown, titres reStructuredText soulignés,
table des matières du PDF (PyMuPDF). Les blocs d'une même section sont ensuite regroupés en passages, sans jamais
mélanger deux sections. Le nombre de tokens est estimé (caractères / 4) : aucun tokenizer à télécharger.
"""
import hashlib
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

from rag import config

## Section des lignes qui précèdent le premier titre.
PREAMBLE = "Début du document"
## Taille minimale d'un passage (tokens) : en dessous, c'est un débris d'extraction, pas un contenu.
MIN_TOKENS = 12
## Caractères de soulignement des titres reStructuredText.
_RST_ADORNMENT = re.compile(r"^([=\-~^\"'`#*+:.])\1{2,}\s*$")
## Titre Markdown ATX (# à ######).
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
## Début ou fin de bloc de code Markdown.
_MD_FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass(frozen=True)
class Block:
    """! @brief Morceau de texte continu d'un document, dans une seule section et une seule page.

    @param section  Titre de la section (chemin « Titre > Sous-titre »).
    @param page     Page (PDF, à partir de 1) ; None pour un texte.
    @param text     Texte.
    """

    section: str
    page: int | None
    text: str


@dataclass(frozen=True)
class Chunk:
    """! @brief Passage indexé.

    @param chunk_id    Identifiant stable (empreinte du contenu et de sa position).
    @param source      Nom du document affiché dans les citations.
    @param source_id   Identifiant de la source dans le catalogue.
    @param section     Section du passage.
    @param page        Première page (PDF) ou None.
    @param page_end    Dernière page (PDF) ou None.
    @param text        Texte du passage.
    @param visibility  "public" ou "private".
    """

    chunk_id: str
    source: str
    source_id: str
    section: str
    page: int | None
    page_end: int | None
    text: str
    visibility: str = "public"

    def embed_text(self):
        """! @brief Texte envoyé au modèle d'embeddings : le document et la section en tête aident la recherche.

        @return str.
        """
        return f"{self.source} | {self.section}\n{self.text}"

    def citation(self):
        """! @brief Référence lisible : fichier, page(s), section.

        @return str, ex. « balle2017_end2end_1611.01704v3.pdf, p. 3 (2 Choice of forward...) ».
        """
        if self.page is None:
            return f"{self.source} ({self.section})"
        pages = f"p. {self.page}" if self.page_end in (None, self.page) else f"p. {self.page}-{self.page_end}"
        return f"{self.source}, {pages} ({self.section})"

    def to_dict(self):
        """! @brief Dictionnaire sérialisable (JSON).

        @return dict.
        """
        return asdict(self)


def estimate_tokens(text):
    """! @brief Estimation du nombre de tokens, sans tokenizer.

    @param text  Texte.
    @return int >= 1 pour un texte non vide, 0 sinon.
    """
    text = text.strip()
    return 0 if not text else max(1, round(len(text) / config.CHARS_PER_TOKEN))


def _clean(text):
    """! @brief Normalise un texte extrait : Unicode NFC, ligatures, espaces de fin, lignes vides multiples.

    @param text  Texte brut.
    @return Texte nettoyé.
    """
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # césure de fin de ligne (PDF)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# --------------------------------------------------------------------------- lecture par format

def parse_markdown(text):
    """! @brief Découpe un Markdown selon ses titres (#...), en ignorant les blocs de code.

    @param text  Contenu du fichier.
    @return Liste de Block (page None).
    """
    blocks, path, lines, in_code = [], [], [], False

    def flush():
        """! @brief Termine le bloc en cours."""
        body = _clean("\n".join(lines))
        if body:
            blocks.append(Block(" > ".join(t for _, t in path) or PREAMBLE, None, body))
        lines.clear()

    for line in text.splitlines():
        if _MD_FENCE.match(line):
            in_code = not in_code
        m = None if in_code else _MD_HEADING.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            path = [(lv, t) for lv, t in path if lv < level] + [(level, title)]
        else:
            lines.append(line)
    flush()
    return blocks


def parse_rst(text):
    """! @brief Découpe un reStructuredText selon ses titres soulignés (le niveau suit l'ordre d'apparition des styles).

    @param text  Contenu du fichier.
    @return Liste de Block (page None).
    """
    raw = text.splitlines()
    styles, path, blocks, lines = [], [], [], []
    i = 0

    def flush():
        """! @brief Termine le bloc en cours."""
        body = _clean("\n".join(lines))
        if body:
            blocks.append(Block(" > ".join(t for _, t in path) or PREAMBLE, None, body))
        lines.clear()

    while i < len(raw):
        line = raw[i]
        nxt = raw[i + 1] if i + 1 < len(raw) else ""
        over = _RST_ADORNMENT.match(line)
        title = None
        if over and i + 2 < len(raw) and raw[i + 1].strip() and _RST_ADORNMENT.match(raw[i + 2]) \
                and raw[i + 2].strip()[0] == line.strip()[0]:
            title, style, step = raw[i + 1].strip(), "over" + line.strip()[0], 3
        elif line.strip() and not over and _RST_ADORNMENT.match(nxt) and len(nxt.strip()) >= len(line.strip()):
            title, style, step = line.strip(), nxt.strip()[0], 2
        if title:
            flush()
            if style not in styles:
                styles.append(style)
            level = styles.index(style)
            path = [(lv, t) for lv, t in path if lv < level] + [(level, title)]
            i += step
            continue
        lines.append(line)
        i += 1
    flush()
    return blocks


def _norm(text):
    """! @brief Forme de comparaison : minuscules, sans accents ni ponctuation, espaces simples.

    @param text  Texte.
    @return str.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    return " ".join(re.findall(r"\w+", text))


def pdf_blocks(pages, toc):
    """! @brief Découpe le texte des pages d'un PDF selon sa table des matières.

    Un titre est cherché dans le texte de sa page (sans son numéro, sans tenir compte de la casse) : le bloc est
    coupé là ; s'il n'est pas trouvé, la section commence en haut de la page.
    @param pages  Liste des textes des pages (page 1 en premier).
    @param toc    Table des matières PyMuPDF : [[niveau, titre, page], ...] ; vide : une section par document.
    @return Liste de Block.
    """
    starts = {}                                      # page -> [(niveau, titre)]
    for level, title, page in toc:
        if 1 <= page <= len(pages):
            starts.setdefault(page, []).append((level, title.strip()))
    blocks, path = [], []
    for pno, text in enumerate(pages, start=1):
        text = _clean(text)
        cuts = []                                    # (position, niveau, titre)
        for level, title in starts.get(pno, []):
            core = re.sub(r"^\s*\d+(\.\d+)*\.?\s+", "", title)
            pos = _find_heading(text, core)
            cuts.append((pos, level, title))
        cuts.sort(key=lambda c: c[0])
        prev = 0
        for pos, level, title in cuts:
            body = text[prev:pos].strip()
            if body:
                blocks.append(Block(" > ".join(t for _, t in path) or PREAMBLE, pno, body))
            path = [(lv, t) for lv, t in path if lv < level] + [(level, title)]
            prev = pos
        body = text[prev:].strip()
        if body:
            blocks.append(Block(" > ".join(t for _, t in path) or PREAMBLE, pno, body))
    return blocks


def _find_heading(text, title):
    """! @brief Position du titre dans le texte de sa page (début de la ligne qui le contient), ou 0.

    @param text   Texte de la page.
    @param title  Titre sans numéro.
    @return Indice de caractère.
    """
    target = _norm(title)
    if not target:
        return 0
    offset = 0
    lines = text.split("\n")
    for k, line in enumerate(lines):
        # le titre peut tenir sur deux lignes (« 2 » puis « CHOICE OF ... »), on teste la ligne et la suivante
        window = _norm(line + " " + (lines[k + 1] if k + 1 < len(lines) else ""))
        window = re.sub(r"^(\d+ )+", "", window)        # numéro de section, éventuellement seul sur sa ligne
        if window.startswith(target[:40]) or (_norm(line) and target.startswith(_norm(line)) and len(_norm(line)) > 8):
            return offset
        offset += len(line) + 1
    return 0


def parse_pdf(path):
    """! @brief Lit un PDF avec PyMuPDF : texte par page et table des matières.

    @param path  Fichier PDF.
    @return Liste de Block.
    """
    import pymupdf
    with pymupdf.open(path) as doc:
        pages = [page.get_text("text") for page in doc]
        toc = doc.get_toc(simple=True)
    return pdf_blocks(pages, toc)


def parse_file(path):
    """! @brief Lit un document selon son extension.

    @param path  Fichier .pdf, .md, .rst ou .txt.
    @return Liste de Block.
    @throws ValueError pour une extension inconnue.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".md":
        return parse_markdown(text)
    if suffix == ".rst":
        return parse_rst(text)
    if suffix == ".txt":
        return [Block(PREAMBLE, None, _clean(text))] if text.strip() else []
    raise ValueError(f"format non pris en charge : {path.name}")


# --------------------------------------------------------------------------- passages

def _split_long(text, max_tokens):
    """! @brief Coupe un paragraphe trop long en morceaux de mots d'au plus max_tokens.

    @param text        Paragraphe.
    @param max_tokens  Taille maximale.
    @return Liste de str.
    """
    words, parts, cur = text.split(), [], []
    for w in words:
        if cur and estimate_tokens(" ".join(cur + [w])) > max_tokens:
            parts.append(" ".join(cur))
            cur = []
        cur.append(w)
    if cur:
        parts.append(" ".join(cur))
    return parts


def _tail(text, tokens):
    """! @brief Fin d'un texte d'environ @p tokens tokens, coupée sur un mot.

    @param text    Texte.
    @param tokens  Taille voulue.
    @return str.
    """
    if tokens <= 0:
        return ""
    words, out = text.split(), []
    for w in reversed(words):
        if estimate_tokens(" ".join([w] + out)) > tokens:
            break
        out.insert(0, w)
    return " ".join(out)


def chunk_blocks(blocks, source, source_id, visibility="public", size=None, overlap=None):
    """! @brief Regroupe les blocs en passages d'environ @p size tokens, sans mélanger deux sections.

    Les paragraphes sont gardés entiers quand c'est possible ; chaque passage après le premier d'une section reprend
    la fin du précédent (@p overlap tokens) pour ne pas couper une idée en deux.
    @param blocks      Liste de Block.
    @param source      Nom du document affiché dans les citations.
    @param source_id   Identifiant de la source.
    @param visibility  "public" ou "private".
    @param size        Taille visée (tokens) ; None : config.CHUNK_TOKENS.
    @param overlap     Chevauchement (tokens) ; None : config.CHUNK_OVERLAP.
    @return Liste de Chunk.
    """
    size = size or config.CHUNK_TOKENS
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    if overlap >= size:
        raise ValueError("le chevauchement doit être plus petit que la taille des passages")
    # (section, [(page, paragraphe)]) en gardant l'ordre ; une section reprise plus loin forme un nouveau groupe
    groups = []
    for b in blocks:
        paras = [p.strip() for p in re.split(r"\n\s*\n", b.text) if p.strip()]
        if not paras:
            continue
        if groups and groups[-1][0] == b.section:
            groups[-1][1].extend((b.page, p) for p in paras)
        else:
            groups.append((b.section, [(b.page, p) for p in paras]))

    chunks = []
    for section, paras in groups:
        units = []                                   # (page, morceau) d'au plus size - overlap tokens
        for page, p in paras:
            units += [(page, part) for part in _split_long(p, size - overlap)]
        cur, cur_pages, prefix = [], [], ""
        for page, unit in units:
            candidate = "\n\n".join(([prefix] if prefix else []) + cur + [unit])
            if cur and estimate_tokens(candidate) > size:
                text = "\n\n".join(([prefix] if prefix else []) + cur)
                chunks.append(_make_chunk(text, source, source_id, section, cur_pages, visibility))
                prefix = _tail("\n\n".join(cur), overlap)
                cur, cur_pages = [], []
            cur.append(unit)
            cur_pages.append(page)
        if cur:
            text = "\n\n".join(([prefix] if prefix else []) + cur)
            chunks.append(_make_chunk(text, source, source_id, section, cur_pages, visibility))
    return [c for c in chunks if c is not None]


def _make_chunk(text, source, source_id, section, pages, visibility):
    """! @brief Crée un Chunk et son identifiant stable.

    @param text        Texte du passage.
    @param source      Nom du document.
    @param source_id   Identifiant de la source.
    @param section     Section.
    @param pages       Pages des morceaux (None pour un texte).
    @param visibility  "public" ou "private".
    @return Chunk, ou None pour un débris (numéro de page, légende isolée) de moins de MIN_TOKENS.
    """
    if estimate_tokens(text) < MIN_TOKENS:
        return None
    known = [p for p in pages if p is not None]
    page, page_end = (min(known), max(known)) if known else (None, None)
    digest = hashlib.sha256(f"{source}\0{section}\0{page}\0{text}".encode()).hexdigest()[:20]
    return Chunk(digest, source, source_id, section, page, page_end, text, visibility)


def chunk_document(doc):
    """! @brief Lit et découpe un document du catalogue.

    @param doc  sources.Document.
    @return Liste de Chunk.
    """
    return chunk_blocks(parse_file(doc.path), doc.name, doc.source_id, doc.visibility)

"""Découpage : sections, taille, chevauchement, métadonnées (fichier, page, section)."""
import pytest

from rag import config
from rag.chunking import (
    PREAMBLE,
    Block,
    chunk_blocks,
    estimate_tokens,
    parse_file,
    parse_markdown,
    parse_rst,
    pdf_blocks,
)


def words(n, start=0):
    return " ".join(f"mot{i}" for i in range(start, start + n))


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("   ") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 4000) == 4000 // config.CHARS_PER_TOKEN


def test_markdown_sections_follow_headings_and_ignore_code():
    text = "intro\n\n# Titre\n\ncorps 1\n\n## Sous-titre\n\ncorps 2\n\n```\n# pas un titre\n```\n\n# Autre\n\nfin"
    blocks = parse_markdown(text)
    assert [b.section for b in blocks] == [PREAMBLE, "Titre", "Titre > Sous-titre", "Autre"]
    assert "# pas un titre" in blocks[2].text
    assert all(b.page is None for b in blocks)


def test_rst_sections_levels_follow_adornment_order():
    text = ("=====\nAidge\n=====\n\nintro\n\nInstall\n-------\n\npip install\n\nDétails\n~~~~~~~\n\nplus\n\n"
            "Usage\n-----\n\nexemple\n")
    blocks = parse_rst(text)
    assert [b.section for b in blocks] == ["Aidge", "Aidge > Install", "Aidge > Install > Détails", "Aidge > Usage"]


def test_pdf_blocks_split_pages_on_toc_titles():
    pages = ["Titre de l'article\nRésumé ici.\n1\nINTRODUCTION\nTexte de l'intro.",
             "suite de l'intro\n2 MÉTHODE\nLa GDN normalise.\n2.1 Détail\nprécisions"]
    toc = [[1, "1 Introduction", 1], [1, "2 Méthode", 2], [2, "2.1 Détail", 2]]
    blocks = pdf_blocks(pages, toc)
    got = [(b.section, b.page, b.text.split("\n")[0]) for b in blocks]
    assert got == [(PREAMBLE, 1, "Titre de l'article"),
                   ("1 Introduction", 1, "1"),
                   ("1 Introduction", 2, "suite de l'intro"),
                   ("2 Méthode", 2, "2 MÉTHODE"),
                   ("2 Méthode > 2.1 Détail", 2, "2.1 Détail")]


def test_pdf_without_toc_is_one_section():
    blocks = pdf_blocks(["page un", "page deux"], [])
    assert [(b.section, b.page) for b in blocks] == [(PREAMBLE, 1), (PREAMBLE, 2)]


def test_chunks_respect_size_overlap_and_sections():
    paras = [words(50, i * 50) for i in range(30)]                 # ~30 x 85 tokens
    blocks = [Block("A", 3, "\n\n".join(paras[:20])), Block("A", 4, "\n\n".join(paras[20:25])),
              Block("B", 4, "\n\n".join(paras[25:]))]
    chunks = chunk_blocks(blocks, "doc.pdf", "src", size=400, overlap=60)
    assert len(chunks) > 3
    assert all(estimate_tokens(c.text) <= 400 for c in chunks)
    a = [c for c in chunks if c.section == "A"]
    b = [c for c in chunks if c.section == "B"]
    assert a and b and not any("mot1250" in c.text for c in a)    # aucune section mélangée
    # chevauchement : chaque passage après le premier d'une section reprend la fin du précédent
    for prev, cur in zip(a, a[1:], strict=False):
        tail = prev.text.split()[-5:]
        assert " ".join(tail) in cur.text
    # pages : premier passage page 3, un passage à cheval sur 3 et 4
    assert a[0].page == 3
    assert any(c.page == 3 and c.page_end == 4 for c in a) or a[-1].page == 4
    assert b[0].page == 4 and b[0].page_end == 4
    # identifiants stables et uniques
    again = chunk_blocks(blocks, "doc.pdf", "src", size=400, overlap=60)
    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in again]
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_long_paragraph_is_split_and_tiny_debris_dropped():
    blocks = [Block("S", None, words(2000)), Block("T", None, "12")]
    chunks = chunk_blocks(blocks, "x.md", "src", size=200, overlap=20)
    assert all(estimate_tokens(c.text) <= 200 for c in chunks)
    assert {c.section for c in chunks} == {"S"}                    # « 12 » seul est un débris
    assert "mot1999" in chunks[-1].text


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_blocks([Block("S", None, "texte")], "x", "s", size=100, overlap=100)


def test_metadata_and_citation():
    c = chunk_blocks([Block("2 Méthode", 3, words(40))], "balle.pdf", "balle2017", "public")[0]
    assert (c.source, c.source_id, c.section, c.page, c.page_end, c.visibility) == \
        ("balle.pdf", "balle2017", "2 Méthode", 3, 3, "public")
    assert c.citation() == "balle.pdf, p. 3 (2 Méthode)"
    assert c.embed_text().startswith("balle.pdf | 2 Méthode\n")
    md = chunk_blocks([Block("Install", None, words(40))], "docs/guide.md", "projet")[0]
    assert md.citation() == "docs/guide.md (Install)" and md.page is None


def test_parse_real_pdf_with_toc(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    for text in ("1 Introduction\nLa compression apprise.", "2 Normalisation\nLa GDN divise chaque canal."):
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.set_toc([[1, "1 Introduction", 1], [1, "2 Normalisation", 2]])
    path = tmp_path / "papier.pdf"
    doc.save(path)
    blocks = parse_file(path)
    assert [(b.section, b.page) for b in blocks] == [("1 Introduction", 1), ("2 Normalisation", 2)]
    assert "GDN" in blocks[1].text


def test_unknown_extension(tmp_path):
    p = tmp_path / "x.docx"
    p.write_text("x")
    with pytest.raises(ValueError):
        parse_file(p)

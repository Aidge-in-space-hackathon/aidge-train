"""! @brief Évaluation manuelle de l'assistant sur rag/eval/questions.yaml (hors CI : demande l'index, et le proxy
avec --llm).

Usage :
    python -m rag.eval.run              recherche seule : la bonne source sort-elle dans le top 5 ? (taux de réussite)
    python -m rag.eval.run --llm        affiche aussi les réponses du LLM, à relire à côté du champ « attendu »
Avec --llm, chaque question consomme des requêtes gratuites (Voyage, NVIDIA NIM ou Groq) : 11 questions restent
sous les quotas, mais pas en boucle.
"""
import argparse
import sys
from pathlib import Path

import yaml

## Questions de référence.
QUESTIONS = Path(__file__).with_name("questions.yaml")


def hit(passages, item):
    """! @brief La recherche a-t-elle trouvé une source attendue ?

    @param passages  Liste de rag.retrieval.Passage.
    @param item      Entrée de questions.yaml.
    @return True / False ; None pour une question hors périmètre (aucune source attendue).
    """
    expected = item.get("sources") or []
    if not expected:
        return None
    files = item.get("files") or []
    for p in passages:
        if p.chunk.source_id in expected and (not files or any(f in p.chunk.source for f in files)):
            return True
    return False


def main(argv=None):
    """! @brief Point d'entrée.

    @param argv  Arguments ; ceux de la ligne de commande si None.
    @return Code de sortie (0, ou 1 si l'index manque).
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Évalue l'assistant RAG sur les questions de référence")
    ap.add_argument("--llm", action="store_true", help="interroge aussi le LLM (consomme des quotas)")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args(argv)
    from rag.retrieval import IndexMissing, get_retriever
    items = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    ok = total = 0
    for i, item in enumerate(items, 1):
        try:
            result = get_retriever().search(item["question"], args.k)
        except IndexMissing as e:
            print(e, file=sys.stderr)
            return 1
        found = hit(result.passages, item)
        if found is not None:
            total += 1
            ok += found
        mark = {True: "OK ", False: "RATÉ", None: "hors"}[found]
        print(f"\n[{i:2}] {mark}  {item['question']}   ({' + '.join(result.methode)})")
        for p in result.passages:
            print(f"       - {p.chunk.citation()[:110]}")
        if args.llm:
            from rag.agent import Agent, cited
            answer, sources = "", []
            for event in Agent().run(item["question"]):
                if event.kind == "text":
                    answer += event.data
                elif event.kind == "sources":
                    sources = cited(answer, event.data)
                elif event.kind == "error":
                    answer += f"\n[erreur] {event.data}"
            print(f"     Réponse : {answer.strip()}")
            print(f"     Sources : {'; '.join(p['source'] for p in sources) or 'aucune'}")
            print(f"     Attendu : {item.get('attendu', '')}")
    print(f"\nRecherche : {ok}/{total} questions avec une source attendue dans le top {args.k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

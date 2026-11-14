"""! @brief Évaluation de l'assistant sur rag/eval/questions.yaml (demande l'index, et le proxy avec --llm).

Usage :
    python -m rag.eval.run              recherche seule : la bonne source sort-elle dans le top 5 ? (taux de réussite)
    python -m rag.eval.run --wait       idem, en attendant les créneaux Voyage plutôt que se replier (CI)
    python -m rag.eval.run --llm        affiche aussi les réponses du LLM, à relire à côté du champ « attendu »
Sans --wait, une question qui attendrait plus de config.QUERY_MAX_WAIT_S passe en local, comme dans la bulle : le
score mesure alors le repli. Avec --wait, comptez environ 40 s par question (2 requêtes, 3 par minute sans moyen de
paiement). Avec --llm, chaque question consomme des requêtes gratuites (Voyage, NVIDIA NIM ou Groq) : les questions
de référence restent sous les quotas, mais pas en boucle.
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
    @return Code de sortie : 0, 1 si l'index manque, 2 si une question s'est repliée alors que --wait l'interdit.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Évalue l'assistant RAG sur les questions de référence")
    ap.add_argument("--llm", action="store_true", help="interroge aussi le LLM (consomme des quotas)")
    ap.add_argument("--wait", action="store_true",
                    help="attend les créneaux Voyage au lieu de se replier : mesure la chaîne complète (CI) ; "
                         "une question repliée fait échouer l'évaluation")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args(argv)
    if args.wait:
        from rag import config
        config.QUERY_MAX_WAIT_S = None               # None : attente sans limite, nouvelles tentatives sur 429
    from rag.retrieval import IndexMissing, get_retriever
    items = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    ok = total = degraded = 0
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
        for note in result.notes:
            print(f"     ! {note}")
        degraded += bool(result.notes)
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
    if degraded:
        print(f"{degraded} question(s) en mode dégradé (repli local, BM25 seul ou ordre RRF)")
        if args.wait:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

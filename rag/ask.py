"""! @brief Question à l'assistant en ligne de commande.

Usage :
    python -m rag.ask "Comment fonctionne la GDN ?"
    python -m rag.ask --search-only "GDN"        passages trouvés, sans LLM (diagnostic de la recherche)
Demande l'index (python -m rag.ingest) et, sauf --search-only, le proxy LiteLLM (LITELLM_BASE_URL,
LITELLM_MASTER_KEY).
"""
import argparse
import sys

from rag import config


def main(argv=None):
    """! @brief Point d'entrée en ligne de commande.

    @param argv  Arguments ; ceux de la ligne de commande si None.
    @return Code de sortie (0 réponse obtenue, 1 erreur).
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Pose une question à l'assistant RAG")
    ap.add_argument("question")
    ap.add_argument("--search-only", action="store_true", help="affiche les passages trouvés, sans LLM")
    ap.add_argument("-k", type=int, default=config.TOP_K, help="passages (recherche seule)")
    args = ap.parse_args(argv)

    if args.search_only:
        from rag.retrieval import IndexMissing, get_retriever
        try:
            result = get_retriever().search(args.question, args.k)
        except IndexMissing as e:
            print(e, file=sys.stderr)
            return 1
        print(f"Méthode : {' + '.join(result.methode)}")
        for note in result.notes:
            print(f"Remarque : {note}")
        for i, p in enumerate(result.passages, 1):
            print(f"\n[{i}] {p.chunk.citation()}  (score {p.score:.3f})\n{p.chunk.text[:600]}")
        return 0

    from rag.agent import Agent, cited
    try:
        agent = Agent()
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1
    answer, code = "", 0
    for event in agent.run(args.question):
        if event.kind == "text":
            answer += event.data
            print(event.data, end="", flush=True)
        elif event.kind == "tool":
            print(f"[outil {event.data['name']} {event.data['arguments']}]", file=sys.stderr)
        elif event.kind == "error":
            print(f"\n{event.data}", file=sys.stderr)
            code = 1
        elif event.kind == "sources" and event.data:
            print("\n\nSources :")
            for p in cited(answer, event.data):
                print(f"- {p['source']}")
    print()
    return code


if __name__ == "__main__":
    raise SystemExit(main())

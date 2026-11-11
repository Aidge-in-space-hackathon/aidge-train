"""! @brief Registre d'outils de l'agent : schéma JSON (format OpenAI) + fonction Python, appelés par leur nom.

Pour ajouter un outil, décorer une fonction avec @c REGISTRY.tool(...) dans ce module ou un autre importé par
rag.agent. Prévus plus tard (non implémentés) :
- search_code(query, k) : recherche dans le code du dépôt ;
- get_runs(ids) : métriques des entraînements et optimisations (file SQLite de la plateforme) ;
- suggest_config(...) : configuration proposée, avec un bouton « Appliquer au formulaire » dans la bulle.
Un outil renvoie un objet sérialisable en JSON ; une exception devient {"erreur": ...} pour le LLM, sans arrêter
la conversation.
"""
import json
from collections.abc import Callable
from dataclasses import dataclass

from rag import config


@dataclass(frozen=True)
class Tool:
    """! @brief Outil appelable par le LLM.

    @param name         Nom (lettres, chiffres, _).
    @param description  Ce que fait l'outil et quand l'appeler (lu par le LLM).
    @param parameters   Schéma JSON des arguments.
    @param func         Fonction appelée avec les arguments nommés.
    """

    name: str
    description: str
    parameters: dict
    func: Callable

    def schema(self):
        """! @brief Déclaration de l'outil pour l'API chat.completions.

        @return dict.
        """
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


class ToolRegistry:
    """! @brief Ensemble d'outils disponibles pour l'agent."""

    def __init__(self):
        """! @brief Registre vide."""
        self._tools = {}

    def register(self, tool):
        """! @brief Ajoute un outil.

        @param tool  Tool.
        @return tool.
        @throws ValueError si le nom est déjà pris ou invalide.
        """
        if not tool.name.replace("_", "").isalnum():
            raise ValueError(f"nom d'outil invalide : {tool.name!r}")
        if tool.name in self._tools:
            raise ValueError(f"outil déjà enregistré : {tool.name}")
        if tool.parameters.get("type") != "object":
            raise ValueError(f"{tool.name} : le schéma des paramètres doit être de type object")
        self._tools[tool.name] = tool
        return tool

    def tool(self, name, description, parameters):
        """! @brief Décorateur : enregistre la fonction comme outil.

        @param name         Nom de l'outil.
        @param description  Description pour le LLM.
        @param parameters   Schéma JSON des arguments.
        @return Décorateur qui rend la fonction inchangée.
        """
        def decorate(func):
            """! @brief Enregistre @p func.

            @param func  Fonction de l'outil.
            @return func.
            """
            self.register(Tool(name, description, parameters, func))
            return func
        return decorate

    def names(self):
        """! @brief Noms des outils, dans l'ordre d'enregistrement.

        @return Liste de str.
        """
        return list(self._tools)

    def get(self, name):
        """! @brief Outil par son nom.

        @param name  Nom.
        @return Tool ou None.
        """
        return self._tools.get(name)

    def schemas(self):
        """! @brief Déclarations de tous les outils pour l'API chat.completions.

        @return Liste de dict.
        """
        return [t.schema() for t in self._tools.values()]

    def call(self, name, arguments):
        """! @brief Exécute un outil demandé par le LLM.

        @param name       Nom de l'outil.
        @param arguments  Arguments : chaîne JSON (telle que renvoyée par le LLM) ou dict.
        @return Résultat sérialisable ; {"erreur": ...} si l'outil est inconnu, les arguments invalides ou l'appel
                en échec.
        """
        tool = self._tools.get(name)
        if tool is None:
            return {"erreur": f"outil inconnu : {name}. Outils disponibles : {', '.join(self._tools)}"}
        try:
            args = json.loads(arguments or "{}") if isinstance(arguments, str) else dict(arguments or {})
        except json.JSONDecodeError:
            return {"erreur": "arguments JSON invalides"}
        if not isinstance(args, dict):
            return {"erreur": "les arguments doivent être un objet JSON"}
        props = tool.parameters.get("properties", {})
        missing = [p for p in tool.parameters.get("required", []) if p not in args]
        if missing:
            return {"erreur": f"arguments manquants : {', '.join(missing)}"}
        args = {k: v for k, v in args.items() if k in props}       # arguments inventés ignorés
        try:
            return tool.func(**args)
        except Exception as e:                                       # l'erreur est rendue au LLM, pas à l'utilisateur
            return {"erreur": f"{type(e).__name__}: {e}"}


## Registre utilisé par l'agent.
REGISTRY = ToolRegistry()


@REGISTRY.tool(
    "search_docs",
    "Cherche dans la documentation du projet : papier de Ballé et al. (ICLR 2017, compression d'image apprise, GDN), "
    "article ERTS 2026 sur Aidge, documentation Eclipse Aidge 0.10.1, documentation LoCod, docs et README du dépôt "
    "aidge-train. Renvoie les passages les plus pertinents avec leur source (fichier, page, section) à citer. "
    "Appelle-le avant de répondre à toute question technique ; reformule la requête si les passages ne "
    "conviennent pas.",
    {"type": "object",
     "properties": {"query": {"type": "string", "description": "Ce qu'il faut chercher (mots-clés ou question)."},
                    "k": {"type": "integer", "minimum": 1, "maximum": 8,
                          "description": "Nombre de passages (5 par défaut)."}},
     "required": ["query"]})
def search_docs(query, k=None):
    """! @brief Outil search_docs : passages pertinents et leur source.

    @param query  Requête.
    @param k      Nombre de passages (1 à 8) ; None : config.TOP_K.
    @return dict {"passages": [...], "methode": [...]} ; {"passages": [], "remarque": ...} si rien n'est trouvé.
    """
    from rag.retrieval import get_retriever
    query = str(query).strip()
    if not query:
        return {"erreur": "requête vide"}
    try:
        k = max(1, min(8, int(k))) if k is not None else config.TOP_K
    except (TypeError, ValueError):
        k = config.TOP_K
    result = get_retriever().search(query, k)
    out = {"passages": [p.to_tool() for p in result.passages], "methode": result.methode}
    if not result.passages:
        out["remarque"] = "aucun passage trouvé"
    return out

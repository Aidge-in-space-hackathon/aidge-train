"""! @brief Assistant RAG de la plateforme : recherche dans le papier de Ballé, les docs d'Aidge, de LoCod et du projet.

Ingestion : @c python -m rag.ingest ; question en ligne de commande : @c python -m rag.ask "question".
Sources : rag/sources.yaml (source unique de vérité). Index : rag/index/ (non versionné).
Le LLM passe par le proxy LiteLLM (litellm/config.yaml) ; l'interface l'affiche dans une bulle
(aidge_in_space/ui_assistant.py).
"""

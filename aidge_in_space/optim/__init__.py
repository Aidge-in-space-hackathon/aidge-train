"""! @brief Optimisation d'un encodeur avec Aidge : quantification, plan mémoire, évaluation et exports C.

Paquet sans import lourd : optim.config reste importable sans torch ni Aidge. Les autres modules chargent ces
bibliothèques et ne s'importent qu'au moment de les utiliser. Contrat : docs/conception_optimisation.md.
"""

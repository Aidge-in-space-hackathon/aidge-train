"""Réglages communs des tests."""


def pytest_configure(config):
    """Déclare le marqueur des tests lents : python -m pytest tests -m "not slow" les exclut."""
    config.addinivalue_line("markers", "slow: test lent (quantification Aidge, compilation d'un export)")

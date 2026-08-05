"""Tiara package with lazy legacy exports.

Keeping package import light lets schema/config tooling run outside the full
scientific environment. Legacy symbols are loaded on first access.
"""
from __future__ import annotations
__all__=["TfidfWeighter","oligofreq","single_oligofreq","multiple_oligofreq"]
def __getattr__(name):
    if name=="TfidfWeighter":
        from tiara.src.transformations import TfidfWeighter
        return TfidfWeighter
    if name in ("oligofreq","single_oligofreq","multiple_oligofreq"):
        from tiara.src import bow
        return getattr(bow,name)
    raise AttributeError(name)

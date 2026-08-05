"""Tiara2 v2.3 hierarchical classification package (light imports)."""
from .schema import schema, profile, scaled_priors, HierarchySchema
__all__=["schema","profile","scaled_priors","HierarchySchema","HierarchicalClassifier","MaskedHierarchicalLoss"]
def __getattr__(name):
    if name in ("HierarchicalClassifier","MaskedHierarchicalLoss"):
        from . import model
        return getattr(model,name)
    raise AttributeError(name)

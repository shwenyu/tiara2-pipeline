# Hotfix 1

Fixes the missing metadata bootstrap step:

- adds `python -m tiara.hierarchical init-metadata`;
- keeps init/audit independent of the heavy ML imports;
- missing TSV now returns structured guidance instead of a traceback;
- audit reports unresolved euk labels and leakage counts;
- README now uses the correct command order.

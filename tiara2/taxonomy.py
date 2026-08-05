"""NCBI taxonomy: taxid -> lineage -> rank -> curation clade.

Why this module exists
----------------------
The curation strategy is expressed in TAXONOMIC terms ("one representative
genome per genus", "Fungi get 20 percent of the eukaryotic budget"), but every
upstream artifact -- assembly_summary.txt, the unified index, the download
manifest -- only carries a bare taxid. Something has to turn that integer into
(genus, family, clade). That is all this module does.

Two backends, same interface, chosen automatically:

1. taxdump  -- the real thing: nodes.dmp + names.dmp. Exact ranks and parents.
2. lineage  -- fallback: a pre-rendered semicolon lineage string of the kind
   Tiara's Supplementary Table S1 carries. Ranks are unknown, so genus is the
   last element and clade membership is matched by NAME.

The fallback exists on purpose: the download is still running, and a curation
plan must be computable from metadata already on disk. It is strictly less
precise, and Taxonomy.backend records which one was used so the manifest never
hides it.

NOTHING here does network I/O. Fetching taxdump belongs to the acquire stage.

Why names are NEVER used for identity
-------------------------------------
Tiara's own S1 table lists "Saccharomyces kluyveri" and "Lachancea kluyveri"
as two rows pointing at ONE accession (GCA_000149225.1), and likewise
Chlorella / Auxenochlorella pyrenoidosa. Grouping by organism-name string
would have silently produced two genera from one genome. Grouping is therefore
keyed on taxid whenever a taxdump is available.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Ranks we ever need to resolve. Kept small: every extra rank is another dict
# lookup per genome across ~10^5 genomes.
WANTED_RANKS = ("genus", "family", "order", "class", "phylum", "kingdom",
                "superkingdom")

# --- Curation clades ------------------------------------------------------
# Ordered MOST SPECIFIC FIRST: the first matching ancestor wins, so Vertebrata
# must be tested before Metazoa or every vertebrate would land in the
# invertebrate bucket.
#
# These are the buckets the eukaryotic fragment budget is split across. The
# protist/fungal/algal entries reproduce Tiara1's in-domain "microbial
# eukaryote" scope; vertebrate / invertebrate / land plant are the deliberate
# widening of Tiara2's target range.
DEFAULT_CLADES = (
    # (clade name, taxid of the clade root, name aliases for the fallback)
    ("metazoa_vertebrate", 7742, ("Vertebrata", "Craniata")),
    ("metazoa_invertebrate", 33208, ("Metazoa",)),
    ("land_plant", 3193, ("Embryophyta", "Tracheophyta", "Streptophyta")),
    ("fungi", 4751, ("Fungi",)),
    ("alveolata", 33630, ("Alveolata", "Apicomplexa", "Ciliophora")),
    ("stramenopiles", 33634, ("Stramenopiles", "Bacillariophyta",
                              "Phaeophyceae", "Eustigmatophyceae")),
    ("algae", 3041, ("Chlorophyta",)),
    ("algae", 2763, ("Rhodophyta", "Bangiophyceae")),
    ("other_protist", 2611352, ("Discoba", "Euglenozoa", "Kinetoplastida")),
    ("other_protist", 554915, ("Amoebozoa", "Evosea")),
    ("other_protist", 543769, ("Rhizaria", "Cercozoa")),
    ("other_protist", 2683617, ("Haptista", "Haptophyta")),
    ("other_protist", 2686027, ("Cryptista", "Cryptophyceae")),
    ("eukarya_other", 2759, ("Eukaryota",)),
    ("bacteria", 2, ("Bacteria",)),
    ("archaea", 2157, ("Archaea",)),
)


@dataclass
class TaxonInfo:
    """Everything curation needs to know about one genome's taxid."""
    taxid: int = 0
    ok: bool = False
    clade: str = "unknown"
    #: key for the "one genome per genus" grouping -- a taxid-derived string
    #: when a real genus node exists, else a synthetic per-genome key.
    genus_key: str = ""
    genus_name: str = ""
    family_key: str = ""
    family_name: str = ""
    ranks: dict = field(default_factory=dict)
    lineage_names: tuple = ()

    def as_dict(self) -> dict:
        return {"taxid": self.taxid, "ok": self.ok, "clade": self.clade,
                "genus_key": self.genus_key, "genus_name": self.genus_name,
                "family_key": self.family_key,
                "family_name": self.family_name}


def _iter_dmp(path):
    """Yield the pipe-separated, whitespace-padded fields of a .dmp line."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            yield [f.strip() for f in line.rstrip("\n").rstrip("\t|").split("|")]


def load_nodes(path) -> dict:
    """nodes.dmp -> {taxid: (parent_taxid, rank)}."""
    out = {}
    for fields in _iter_dmp(path):
        if len(fields) < 3:
            continue
        try:
            out[int(fields[0])] = (int(fields[1]), fields[2])
        except ValueError:
            continue
    return out


def load_names(path) -> dict:
    """names.dmp -> {taxid: scientific name} (other name classes ignored)."""
    out = {}
    for fields in _iter_dmp(path):
        if len(fields) < 4 or fields[3] != "scientific name":
            continue
        try:
            out[int(fields[0])] = fields[1]
        except ValueError:
            continue
    return out


def find_taxdump(*candidates):
    """First directory among candidates holding nodes.dmp + names.dmp.

    Accepts the directory itself or a parent containing a taxdump/taxonomy
    subdirectory, which is how ncbi_pipeline.py lays it out.
    """
    for cand in candidates:
        if not cand:
            continue
        root = Path(str(cand)).expanduser()
        for probe in (root, root / "taxdump", root / "taxonomy",
                      root / "taxonomy" / "taxdump"):
            if (probe / "nodes.dmp").exists() and (probe / "names.dmp").exists():
                return probe
    return None


class Taxonomy:
    """Resolve taxids to curation clades and grouping keys.

    Build with from_taxdump (exact) or from_lineages (fallback). ``backend``
    says which, and every consumer records it in its manifest.
    """

    def __init__(self, *, nodes=None, names=None, clades=DEFAULT_CLADES,
                 backend="none"):
        self.nodes = nodes or {}
        self.names = names or {}
        self.clades = tuple(clades)
        self.backend = backend
        self._cache = {}
        self._clade_roots = {root: name for name, root, _a in self.clades}
        self._clade_order = [root for _n, root, _a in self.clades]
        self._alias_to_clade = {}
        for name, _root, aliases in self.clades:
            for alias in aliases:
                self._alias_to_clade.setdefault(alias.lower(), name)

    # ---- constructors ----
    @classmethod
    def from_taxdump(cls, taxdump_dir, *, clades=DEFAULT_CLADES):
        d = Path(str(taxdump_dir))
        return cls(nodes=load_nodes(d / "nodes.dmp"),
                   names=load_names(d / "names.dmp"),
                   clades=clades, backend="taxdump")

    @classmethod
    def from_lineages(cls, *, clades=DEFAULT_CLADES):
        """Name-string backend: no taxdump on disk yet."""
        return cls(clades=clades, backend="lineage")

    # ---- resolution ----
    def ancestors(self, taxid):
        """Yield (taxid, rank) from the node up to the root.

        Bounded and cycle-aware: stale taxdumps do contain malformed parent
        chains, and an infinite loop here would hang a 100k-genome plan.
        """
        seen = set()
        node = int(taxid)
        for _ in range(200):
            if node in seen or node not in self.nodes:
                return
            seen.add(node)
            parent, rank = self.nodes[node]
            yield node, rank
            if parent == node:
                return
            node = parent

    def info(self, taxid, *, lineage=None, organism=None) -> TaxonInfo:
        """Resolve one genome. lineage/organism only feed the fallback."""
        try:
            tid = int(taxid)
        except (TypeError, ValueError):
            tid = 0
        if self.backend == "taxdump" and tid in self._cache:
            return self._cache[tid]
        if self.backend == "taxdump" and tid:
            out = self._info_taxdump(tid)
            if out.ok:
                self._cache[tid] = out
                return out
            # Unknown taxid (index older than the taxdump, or a suppressed
            # node). Degrade for THIS genome instead of dropping it.
            out = self._info_lineage(tid, lineage, organism)
            self._cache[tid] = out
            return out
        return self._info_lineage(tid, lineage, organism)

    def _info_taxdump(self, tid) -> TaxonInfo:
        out = TaxonInfo(taxid=tid)
        chain = list(self.ancestors(tid))
        if not chain:
            return out
        out.ok = True
        ids = [n for n, _r in chain]
        out.ranks = {r: n for n, r in reversed(chain) if r in WANTED_RANKS}
        out.lineage_names = tuple(self.names.get(n, str(n)) for n in reversed(ids))
        idset = set(ids)
        for root in self._clade_order:
            if root in idset:
                out.clade = self._clade_roots[root]
                break
        genus = out.ranks.get("genus")
        if genus:
            out.genus_key = "t%d" % genus
            out.genus_name = self.names.get(genus, "taxid:%d" % genus)
        else:
            # No genus node (common for "unclassified X" and for taxids that
            # sit above genus). Keeping the genome as its OWN group is the
            # conservative choice: merging every rank-less organism into one
            # bucket would delete real diversity.
            out.genus_key = "n%d" % tid
            out.genus_name = self.names.get(tid, "taxid:%d" % tid)
        family = out.ranks.get("family")
        if family:
            out.family_key = "t%d" % family
            out.family_name = self.names.get(family, "taxid:%d" % family)
        else:
            out.family_key = out.genus_key
            out.family_name = out.genus_name
        return out

    def _info_lineage(self, tid, lineage, organism) -> TaxonInfo:
        """Fallback: derive clade/genus from a semicolon lineage string."""
        out = TaxonInfo(taxid=tid)
        parts = [p.strip() for p in str(lineage or "").split(";") if p.strip()]
        if parts:
            out.ok = True
            out.lineage_names = tuple(parts)
            for part in reversed(parts):   # deepest match wins
                hit = self._alias_to_clade.get(part.lower())
                if hit:
                    out.clade = hit
                    break
            out.genus_name = parts[-1]
            out.genus_key = "s" + parts[-1].lower()
            out.family_name = parts[-2] if len(parts) > 1 else parts[-1]
            out.family_key = "s" + out.family_name.lower()
            return out
        if organism:
            # Weakest possible signal: first token of the organism name. Left
            # ok=False so the report can count how much of the plan rests on it.
            genus = str(organism).split()[0]
            out.genus_name = genus
            out.genus_key = "s" + genus.lower()
            out.family_key = out.genus_key
            out.family_name = genus
        return out


def build(cfg: dict) -> Taxonomy:
    """Best Taxonomy available for this config. Never raises.

    Looks at the configured taxdump location, then the usual acquire metadata
    directories, and only then degrades to the lineage backend.
    """
    cur = (cfg.get("curate", {}) or {})
    found = find_taxdump(
        cur.get("taxdump_dir"),
        os.path.join(str(cfg.get("base", ".")), "metadata"),
        os.path.join(str(cfg.get("data_home", ".")), "metadata"),
        cfg.get("base"), cfg.get("data_home"),
    )
    if found:
        try:
            return Taxonomy.from_taxdump(found)
        except OSError:
            pass
    return Taxonomy.from_lineages()

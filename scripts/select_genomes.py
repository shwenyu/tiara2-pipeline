#!/usr/bin/env python3
"""Genus-aware genome selection and per-genome sampling budget.

This is the pure, testable core of the ``curate`` stage. It takes a table of
candidate genomes (one row per assembly, the columns of
``assembly_summary.txt`` plus a taxid) and produces:

  * a SELECTION -- which assemblies to keep, and
  * a BUDGET    -- how many base pairs to draw from each kept assembly.

Why both, and why the second one matters more than it looks
-----------------------------------------------------------
Tiara1 trained on 69 eukaryotic nuclear genomes and still beats our v2.0 by
28 F1 points. Reading Supplementary Table S3 explains why: eukaryotes are
0.84 percent of its genomes but 31.67 percent of its 5 kb training fragments,
and the per-genome depth works out at ~23.7 Mb -- roughly 0.79 genome
equivalents each. Prokaryotes are held to ~1.2 Mb (0.25x) and archaea to
~0.46 Mb (0.12x). Tiara did not "pick fewer genomes"; it capped what each
genome may contribute and spent the saved budget on class balance.

Selection alone therefore does NOT fix our problem. Adding one human genome
at Tiara's 0.79x depth would yield ~620k fragments -- more than twice Tiara's
ENTIRE eukaryotic class -- and would reintroduce exactly the prior skew that
collapsed v2.0's recall. So every selected genome also gets a hard bp cap.

The three knobs, in the order they are applied
----------------------------------------------
1. QUALITY  -- rank assemblies inside a genus, keep the best.
2. BREADTH  -- one genome per genus by default; more only where the genus is
   demonstrably heterogeneous (see ``intra_genus_quota``).
3. DEPTH    -- split each clade's fragment budget across its kept genomes,
   clamped to [min_bp, max_bp] and to a fraction of the genome size.

When a clade's budget cannot give every genome the minimum depth, we DROP
GENOMES rather than lower the depth. Thin sampling of many genomes produces
fragments that are individually uninformative; Tiara's numbers say depth per
genome is what carries the signal.

No I/O policy: everything here is deterministic and side-effect free apart
from the optional __main__ CLI. That is what makes it unit-testable without a
corpus on disk.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import OrderedDict, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tiara2 import taxonomy as taxmod
except Exception:      # pragma: no cover - standalone use without the package
    taxmod = None

# --- Quality ordering ------------------------------------------------------
# Lower is better in every component, so the whole thing sorts ascending.
# These are the only quality signals actually present in assembly_summary.txt;
# BUSCO and N50 are NOT in it, which is why they are optional extra columns
# rather than assumed inputs.
REFSEQ_CATEGORY_RANK = {"reference genome": 0, "representative genome": 1}
ASSEMBLY_LEVEL_RANK = {"complete genome": 0, "chromosome": 1,
                       "scaffold": 2, "contig": 3}


def _num(value, default=0.0):
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return default if out != out else out       # reject NaN


def _text(row, *keys, default=""):
    for key in keys:
        val = row.get(key)
        if val not in (None, "", "na", "NA", "n/a"):
            return str(val).strip()
    return default


def quality_key(row):
    """Sort key ranking assemblies best-first inside a genus.

    Order of precedence, and why:

    1. ``refseq_category`` -- NCBI's own curated pick. A reference genome is
       the community's chosen exemplar; nothing we compute beats that.
    2. ``assembly_level``  -- complete > chromosome > scaffold > contig. A
       contig-level assembly fragments the very k-mer context the model reads.
    3. full representation and a clean ``excluded_from_refseq`` field --
       partial/anomalous assemblies carry annotation-grade defects.
    4. BUSCO completeness, if the caller supplied it (never present in
       assembly_summary.txt itself).
    5. ``contig_count`` ascending -- our N50 proxy, since N50 is absent.
    6. newer ``seq_rel_date`` wins remaining ties, then accession for total
       determinism (two runs must produce the same plan).
    """
    cat = _text(row, "refseq_category").lower()
    level = _text(row, "assembly_level").lower()
    genome_rep = _text(row, "genome_rep").lower()
    excluded = _text(row, "excluded_from_refseq")
    busco = row.get("busco_complete")
    busco_rank = -_num(busco, -1.0) if busco not in (None, "") else 0.0
    contigs = _num(row.get("contig_count"), 1e9)
    date = _text(row, "seq_rel_date", "annotation_date")
    return (
        REFSEQ_CATEGORY_RANK.get(cat, 2),
        ASSEMBLY_LEVEL_RANK.get(level, 4),
        0 if genome_rep in ("full", "") else 1,
        0 if not excluded else 1,
        busco_rank,
        contigs,
        _neg_date(date),
        _text(row, "assembly_accession", "accession"),
    )


def _neg_date(text):
    """Map a date string to a value that sorts NEWEST FIRST."""
    digits = "".join(ch for ch in str(text) if ch.isdigit())[:8]
    return -int(digits) if digits else 0


def passes_gates(row, gates):
    """Hard quality floor applied before any ranking.

    Anything rejected here is never eligible, however empty its genus is. The
    gates are deliberately few: each one is a defect that would inject wrong
    LABELS into training, not merely a lower-quality genome.
    """
    reasons = []
    size = _num(row.get("genome_size"))
    if gates.get("min_genome_size") and size and size < gates["min_genome_size"]:
        reasons.append("genome_too_small")
    if gates.get("max_genome_size") and size > gates["max_genome_size"] > 0:
        reasons.append("genome_too_large")
    level = _text(row, "assembly_level").lower()
    allowed = [str(x).lower() for x in gates.get("allowed_levels", [])]
    if allowed and level and level not in allowed:
        reasons.append("assembly_level")
    if gates.get("require_full_genome_rep", True):
        if _text(row, "genome_rep").lower() not in ("full", ""):
            reasons.append("partial_genome_rep")
    max_contigs = gates.get("max_contig_count") or 0
    if max_contigs and _num(row.get("contig_count")) > max_contigs:
        reasons.append("too_many_contigs")
    if gates.get("exclude_flagged", True) and _text(row, "excluded_from_refseq"):
        reasons.append("excluded_from_refseq")
    if _text(row, "version_status").lower() in ("replaced", "suppressed"):
        reasons.append("version_status")
    return (not reasons), reasons


# --- Breadth: how many genomes may one genus contribute? -------------------
def intra_genus_quota(rows, cfg):
    """Genomes allowed from one genus. Default 1; more only when earned.

    A flat "1 per genus" rule looks clean and is wrong at the edges. Tiara
    itself kept 10 Plasmodium and 8 Leishmania genomes out of 69 -- not
    sloppiness, but coverage of genuine within-genus divergence in AT content
    and genome architecture. Collapsing those to one genome each would have
    deleted signal the classifier depends on.

    So the quota rises above 1 only when the genus shows measurable
    heterogeneity in the two properties a k-mer model can actually see:

      * genome size spread beyond ``size_ratio`` (default 3x), or
      * GC spread beyond ``gc_spread_pp`` (default 8 percentage points).

    The ceiling is ``min(max_per_genus, ceil(n_species ** exponent))``, so even
    a 400-species genus contributes a handful, never a flood.
    """
    if not rows:
        return 0
    max_per = int(cfg.get("max_per_genus", 4))
    if max_per <= 1:
        return 1
    sizes = [_num(r.get("genome_size")) for r in rows]
    sizes = [s for s in sizes if s > 0]
    gcs = [_num(r.get("gc_percent"), -1.0) for r in rows]
    gcs = [g for g in gcs if g >= 0]
    heterogeneous = False
    if sizes and min(sizes) > 0:
        heterogeneous |= (max(sizes) / min(sizes)) >= float(
            cfg.get("size_ratio", 3.0))
    if len(gcs) >= 2:
        heterogeneous |= (max(gcs) - min(gcs)) >= float(
            cfg.get("gc_spread_pp", 8.0))
    if not heterogeneous:
        return 1
    species = {_text(r, "species_taxid", "taxid", default=str(i))
               for i, r in enumerate(rows)}
    scaled = math.ceil(len(species) ** float(cfg.get("exponent", 0.4)))
    return max(1, min(max_per, scaled, len(rows)))


def group_by_genus(rows, taxonomy, *, clade_of_row=None):
    """Bucket rows into {clade: {genus_key: [rows]}}, annotating each row.

    Each row gains ``_clade``, ``_genus_key``, ``_genus``, ``_family`` and
    ``_tax_ok`` in place, so downstream reporting never has to re-resolve a
    taxid.
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        info = taxonomy.info(row.get("taxid") or row.get("species_taxid"),
                             lineage=row.get("lineage"),
                             organism=row.get("organism_name"))
        clade = info.clade
        if clade_of_row is not None:
            clade = clade_of_row(row, info) or clade
        row["_clade"] = clade
        row["_genus_key"] = info.genus_key or ("acc:" + _text(
            row, "assembly_accession", "accession"))
        row["_genus"] = info.genus_name
        row["_family"] = info.family_key or row["_genus_key"]
        row["_family_name"] = info.family_name
        row["_tax_ok"] = info.ok
        grouped[clade][row["_genus_key"]].append(row)
    return grouped


def _family_round_robin(rows, limit):
    """Trim ``rows`` to ``limit``, spreading the loss across families.

    Truncating a quality-sorted list would silently delete whole families --
    typically the least-studied ones, which are exactly the ones widening the
    model's range. Round-robin over families takes the best genome from each
    family first, then the second best, and so on.
    """
    if limit >= len(rows):
        return list(rows)
    by_family = OrderedDict()
    for row in rows:
        by_family.setdefault(row.get("_family", "?"), []).append(row)
    kept, exhausted = [], False
    while len(kept) < limit and not exhausted:
        exhausted = True
        for fam_rows in by_family.values():
            if not fam_rows:
                continue
            kept.append(fam_rows.pop(0))
            exhausted = False
            if len(kept) >= limit:
                break
    return kept


# --- Depth: bp budget per genome -------------------------------------------
def per_genome_bp(clade_bp, n_genomes, cfg, genome_size=0):
    """Base pairs to draw from one genome.

    ``clade_bp / n_genomes`` clamped into [min_bp, max_bp], then capped at
    ``max_genome_fraction`` of the assembly. The last cap is what stops a
    3.1 Gb vertebrate from swallowing its clade: without it a single genome
    can out-produce an entire taxonomic group, which is precisely the prior
    skew that flattened v2.0's posteriors (optimal cutoff fell from 1.0 to
    0.45-0.65 -- a symptom of class-prior distortion, not of model capacity).
    """
    if n_genomes <= 0:
        return 0
    share = clade_bp / float(n_genomes)
    share = max(float(cfg.get("min_bp_per_genome", 5_000_000)),
                min(share, float(cfg.get("max_bp_per_genome", 50_000_000))))
    if genome_size > 0:
        share = min(share, genome_size * float(
            cfg.get("max_genome_fraction", 0.8)))
    return int(share)


def allocate_clade_budget(total_fragments, class_share, clade_share,
                          mean_fragment_bp):
    """Fragment and bp budget per clade.

    ``class_share`` is the eukaryote slice of the whole first-stage budget
    (copied from Tiara S3 so the class priors stay comparable across versions);
    ``clade_share`` splits that slice inside the eukaryotes.
    """
    out = {}
    total = float(total_fragments) * float(class_share)
    denom = sum(float(v) for v in clade_share.values()) or 1.0
    for clade, share in clade_share.items():
        frags = total * float(share) / denom
        out[clade] = {"fragments": int(frags),
                      "bp": int(frags * float(mean_fragment_bp))}
    return out


def select(rows, taxonomy, cfg, *, anchors=()):
    """Run the whole plan. Returns ``(selection, report)``.

    ``anchors`` are accessions that must survive regardless of gates or quotas
    -- the cleaned Tiara1 originals. They are the only genomes with a proven
    F1 contribution, so they are never a casualty of our own heuristics.
    """
    gates = cfg.get("quality_gates", {}) or {}
    genus_cfg = cfg.get("genus", {}) or {}
    budget_cfg = cfg.get("budget", {}) or {}
    anchor_set = {str(a).strip() for a in anchors if str(a).strip()}

    eligible, rejected = [], []
    for row in rows:
        acc = _text(row, "assembly_accession", "accession")
        is_anchor = acc in anchor_set or _strip_version(acc) in {
            _strip_version(a) for a in anchor_set}
        ok, reasons = passes_gates(row, gates)
        if ok or is_anchor:
            row["_anchor"] = bool(is_anchor)
            if is_anchor and not ok:
                row["_anchor_override"] = ",".join(reasons)
            eligible.append(row)
        else:
            rejected.append({"accession": acc, "reasons": reasons})

    grouped = group_by_genus(eligible, taxonomy)

    clade_budget = allocate_clade_budget(
        budget_cfg.get("total_fragments", 40_000_000),
        budget_cfg.get("eukarya_share", 0.3167),
        cfg.get("clade_share", {}) or {},
        budget_cfg.get("mean_fragment_bp", 4200),
    )

    selection, clade_report = [], {}
    for clade, genera in sorted(grouped.items()):
        picked = []
        for genus_key, genus_rows in genera.items():
            genus_rows.sort(key=quality_key)
            quota = intra_genus_quota(genus_rows, genus_cfg)
            forced = [r for r in genus_rows if r.get("_anchor")]
            chosen = list(forced)
            for row in genus_rows:
                if len(chosen) >= max(quota, len(forced)):
                    break
                if row not in chosen:
                    chosen.append(row)
            for rank, row in enumerate(chosen):
                row["_genus_rank"] = rank
            picked.extend(chosen)

        budget = clade_budget.get(clade, {"fragments": 0, "bp": 0})
        cap = _capacity(budget["bp"], budget_cfg)
        dropped = 0
        if cap and len(picked) > cap:
            # Depth beats breadth: keep the number of genomes that can each
            # still receive min_bp, chosen family-stratified so the trim does
            # not amputate whole lineages.
            picked.sort(key=lambda r: (r.get("_genus_rank", 0), quality_key(r)))
            keep_anchor = [r for r in picked if r.get("_anchor")]
            rest = [r for r in picked if not r.get("_anchor")]
            room = max(0, cap - len(keep_anchor))
            dropped = len(rest) - min(room, len(rest))
            picked = keep_anchor + _family_round_robin(rest, room)

        for row in picked:
            bp = per_genome_bp(budget["bp"], len(picked), budget_cfg,
                               genome_size=_num(row.get("genome_size")))
            row["_target_bp"] = bp
            row["_target_fragments"] = int(
                bp / float(budget_cfg.get("mean_fragment_bp", 4200)))
            selection.append(row)

        clade_report[clade] = {
            "genera_available": len(genera),
            "genomes_eligible": sum(len(v) for v in genera.values()),
            "genomes_selected": len(picked),
            "genomes_dropped_for_depth": dropped,
            "budget_fragments": budget["fragments"],
            "budget_bp": budget["bp"],
            "bp_per_genome": (per_genome_bp(budget["bp"], len(picked),
                                            budget_cfg) if picked else 0),
            "planned_bp": sum(r["_target_bp"] for r in picked),
            "anchors": sum(1 for r in picked if r.get("_anchor")),
        }

    report = {
        "taxonomy_backend": getattr(taxonomy, "backend", "unknown"),
        "rows_in": len(rows),
        "rejected_by_gates": len(rejected),
        "selected": len(selection),
        "unresolved_taxids": sum(1 for r in selection if not r.get("_tax_ok")),
        "clades": clade_report,
        "rejections": rejected[:200],
    }
    return selection, report


def select_breadth(rows, taxonomy, cfg, *, anchors=()):
    """Breadth-first selection: every genus contributes, no global budget.

    A DIFFERENT objective from ``select``, deliberately.

    ``select`` reproduces Tiara's design: a fixed fragment budget split by
    class prior, divided across clades, with genomes dropped when the budget
    cannot fund them at depth. That maximises accuracy inside Tiara's scope.

    ``select_breadth`` optimises for COVERAGE. This classifier is meant to see
    a much wider slice of the eukaryotic tree than Tiara's 69 genomes, so the
    question is not "how do we fit inside 12.7M fragments" but "which genera
    exist, and do we hold one good representative of each". No clade quota,
    no total-fragment cap, no depth-driven dropping.

    What is kept from the budgeted path, and why:

      * the gates and the quality ordering -- coverage of junk is not coverage;
      * the intra-genus quota -- a genus with 3x size spread or 8 pp GC spread
        still earns more than one representative (Tiara kept 10 Plasmodium
        for exactly this reason);
      * the anchors -- Tiara's originals are never dropped;
      * a per-genome bp CAP -- breadth without a cap is not breadth. An
        uncapped 3.1 Gb vertebrate yields ~740k fragments at 4.2 kb, more than
        a thousand fungal genera put together, and the rare genera we just
        went to the trouble of finding become numerically invisible. The cap
        here is a flat ceiling (``max_bp_per_genome``), NOT a divided budget:
        it bounds the biggest contributor without imposing a class prior.

    ``clades_only`` restricts the run to part of the tree (e.g. eukaryotes)
    and leaves every other class exactly as it already is.
    """
    gates = cfg.get("quality_gates", {}) or {}
    genus_cfg = cfg.get("genus", {}) or {}
    budget_cfg = cfg.get("budget", {}) or {}
    only = {str(c).strip() for c in (cfg.get("clades_only") or ())
            if str(c).strip()}
    anchor_set = {str(a).strip() for a in anchors if str(a).strip()}
    anchor_bare = {_strip_version(a) for a in anchor_set}

    eligible, rejected = [], []
    for row in rows:
        acc = _text(row, "assembly_accession", "accession")
        is_anchor = acc in anchor_set or _strip_version(acc) in anchor_bare
        ok, reasons = passes_gates(row, gates)
        if ok or is_anchor:
            row["_anchor"] = bool(is_anchor)
            if is_anchor and not ok:
                row["_anchor_override"] = ",".join(reasons)
            eligible.append(row)
        else:
            rejected.append({"accession": acc, "reasons": reasons})

    grouped = group_by_genus(eligible, taxonomy)

    mean_bp = float(budget_cfg.get("mean_fragment_bp", 4200)) or 4200.0
    cap_bp = float(budget_cfg.get("max_bp_per_genome", 50_000_000))
    frac = float(budget_cfg.get("max_genome_fraction", 0.8))

    selection, clade_report, skipped = [], {}, {}
    for clade, genera in sorted(grouped.items()):
        if only and clade not in only:
            skipped[clade] = sum(len(v) for v in genera.values())
            continue
        picked, multi = [], 0
        for genus_rows in genera.values():
            genus_rows.sort(key=quality_key)
            quota = intra_genus_quota(genus_rows, genus_cfg)
            forced = [r for r in genus_rows if r.get("_anchor")]
            chosen = list(forced)
            for row in genus_rows:
                if len(chosen) >= max(quota, len(forced)):
                    break
                if row not in chosen:
                    chosen.append(row)
            for rank, row in enumerate(chosen):
                row["_genus_rank"] = rank
            if len(chosen) > 1:
                multi += 1
            picked.extend(chosen)

        for row in picked:
            size = _num(row.get("genome_size"))
            bp = cap_bp if cap_bp > 0 else (size * frac)
            if size > 0:
                bp = min(bp, size * frac)
            row["_target_bp"] = int(max(0.0, bp))
            row["_target_fragments"] = int(row["_target_bp"] / mean_bp)
            selection.append(row)

        clade_report[clade] = {
            "genera_available": len(genera),
            "genomes_eligible": sum(len(v) for v in genera.values()),
            "genomes_selected": len(picked),
            "genomes_dropped_for_depth": 0,
            "multi_representative_genera": multi,
            "planned_bp": sum(r["_target_bp"] for r in picked),
            "planned_fragments": sum(r["_target_fragments"] for r in picked),
            "anchors": sum(1 for r in picked if r.get("_anchor")),
        }

    found = {_strip_version(_text(r, "assembly_accession", "accession"))
             for r in selection if r.get("_anchor")}
    report = {
        "mode": "breadth",
        "taxonomy_backend": getattr(taxonomy, "backend", "unknown"),
        "rows_in": len(rows),
        "rejected_by_gates": len(rejected),
        "selected": len(selection),
        "genera_selected": len({r["_genus_key"] for r in selection}),
        "families_selected": len({r.get("_family") for r in selection}),
        "unresolved_taxids": sum(1 for r in selection if not r.get("_tax_ok")),
        "planned_fragments": sum(r["_target_fragments"] for r in selection),
        "anchors_requested": len(anchor_bare),
        "anchors_found": len(found),
        "anchors_missing": sorted(anchor_bare - found),
        "clades_skipped": skipped,
        "clades": clade_report,
        "rejections": rejected[:200],
    }
    return selection, report


def _capacity(clade_bp, budget_cfg):
    """How many genomes this clade's bp budget can fund at minimum depth."""
    min_bp = float(budget_cfg.get("min_bp_per_genome", 5_000_000))
    if min_bp <= 0 or clade_bp <= 0:
        return 0
    return max(1, int(clade_bp // min_bp))


def _strip_version(acc):
    return str(acc).split(".")[0]


# --- I/O helpers (thin on purpose) -----------------------------------------
SELECTION_COLUMNS = ("assembly_accession", "taxid", "organism_name", "clade",
                     "genus_key", "genus", "family", "assembly_level",
                     "refseq_category", "genome_size", "gc_percent",
                     "contig_count", "target_bp", "target_fragments",
                     "anchor", "tax_ok", "ftp_path")


def read_tables(paths):
    """Read and concatenate several candidate tables.

    NCBI ships assembly_summary split by division -- fungi, protozoa, plant,
    invertebrate, vertebrate_mammalian, vertebrate_other, ... -- so the euk
    candidate set is a dozen files, not one. Concatenating them here (rather
    than making the caller cat them together) keeps provenance: every row is
    tagged with the file it came from, and duplicate accessions across
    refseq/genbank are collapsed, preferring the RefSeq copy because that is
    the one carrying refseq_category.
    """
    import glob as _glob
    expanded = []
    for item in ([paths] if isinstance(paths, (str, bytes)) else list(paths)):
        hits = sorted(_glob.glob(os.path.expanduser(str(item))))
        expanded.extend(hits or [str(item)])
    seen, out = {}, []
    for path in expanded:
        if not os.path.isfile(path):
            continue
        src = os.path.basename(path)
        prefer = "refseq" in src.lower()
        for row in read_table(path):
            row["_source_file"] = src
            acc = _strip_version(_text(row, "assembly_accession", "accession"))
            if not acc:
                continue
            if acc in seen:
                idx, had_refseq = seen[acc]
                if prefer and not had_refseq:
                    out[idx] = row
                    seen[acc] = (idx, True)
                continue
            seen[acc] = (len(out), prefer)
            out.append(row)
    return out


def read_table(path):
    """Read assembly_summary.txt / TSV / CSV into dicts.

    assembly_summary.txt puts its header on a comment line beginning with
    ``#assembly_accession``; csv.DictReader would otherwise treat it as data.
    """
    delim = "," if str(path).endswith(".csv") else "\t"
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    header_idx = 0
    for i, line in enumerate(lines[:5]):
        if "assembly_accession" in line or "accession" in line.split(delim)[0]:
            header_idx = i
            break
    header = [h.lstrip("#").strip() for h in lines[header_idx].split(delim)]
    out = []
    for line in lines[header_idx + 1:]:
        if line.startswith("#"):
            continue
        out.append(dict(zip(header, line.split(delim))))
    return out


def write_selection(path, selection):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(SELECTION_COLUMNS)
        for row in selection:
            writer.writerow([
                _text(row, "assembly_accession", "accession"),
                _text(row, "taxid"), _text(row, "organism_name"),
                row.get("_clade", ""), row.get("_genus_key", ""),
                row.get("_genus", ""), row.get("_family_name", ""),
                _text(row, "assembly_level"), _text(row, "refseq_category"),
                _text(row, "genome_size"), _text(row, "gc_percent"),
                _text(row, "contig_count"),
                row.get("_target_bp", 0), row.get("_target_fragments", 0),
                int(bool(row.get("_anchor"))), int(bool(row.get("_tax_ok"))),
                _text(row, "ftp_path"),
            ])
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", required=True, nargs="+",
                    help="assembly_summary.txt / index TSV / CSV (globs ok)")
    ap.add_argument("--mode", choices=("breadth", "balanced"),
                    default="breadth",
                    help="breadth: 1 genome per genus, no global budget; "
                         "balanced: Tiara-style class/clade fragment budget")
    ap.add_argument("--out", required=True, help="selection TSV to write")
    ap.add_argument("--report", help="JSON report path")
    ap.add_argument("--config", help="JSON file with the curate config block")
    ap.add_argument("--taxdump", help="directory holding nodes.dmp/names.dmp")
    ap.add_argument("--anchors", help="file of accessions that must be kept")
    args = ap.parse_args(argv)

    cfg = {}
    if args.config:
        with open(args.config, encoding="utf-8") as fh:
            cfg = json.load(fh)
    anchors = []
    if args.anchors and os.path.exists(args.anchors):
        with open(args.anchors, encoding="utf-8") as fh:
            anchors = [ln.strip() for ln in fh if ln.strip()
                       and not ln.startswith("#")]
    if taxmod is None:
        raise SystemExit("tiara2.taxonomy is required")
    tax = (taxmod.Taxonomy.from_taxdump(args.taxdump) if args.taxdump
           else taxmod.Taxonomy.from_lineages())

    if args.mode:
        cfg.setdefault("mode", args.mode)
    runner = select_breadth if cfg.get("mode", "breadth") == "breadth" else select
    selection, report = runner(read_tables(args.input), tax, cfg,
                               anchors=anchors)
    write_selection(args.out, selection)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
    print(json.dumps({k: v for k, v in report.items() if k != "rejections"},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

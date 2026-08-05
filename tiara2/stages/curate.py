"""Stage: curate -- decide WHICH genomes and HOW MUCH of each.

Where it sits and why it is its own stage
-----------------------------------------
acquire  ->  CURATE  ->  chop_bin  ->  dedup  ->  regroup  ->  train

acquire already does a ``select`` step inside ncbi_pipeline.py, but that step
is species-level ("one isolate per species, weighted quality score"). Species
is the wrong granularity for the problem we actually have: the corpus is
species-diverse and genus-narrow, and it is genus-level breadth plus a
per-genome depth cap that Tiara1 got right and v2.0 got wrong.

This stage is deliberately NOT a rewrite of that select step. It runs after it,
reads its output table, and emits a PLAN. It moves no sequence data, deletes
nothing, and downloads nothing -- so it is safe to run at any time, including
while a download is in flight, and re-running it is free.

The plan it writes (``curation_selection.tsv``) carries a ``target_bp`` column
per assembly. Downstream chopping can honour that cap; until it does, the plan
is still the authoritative statement of what the corpus SHOULD contain, and
the JSON report is directly comparable between versions.

The freeze guard
----------------
A plan computed while assemblies are still landing describes a corpus that no
longer exists by the time it is used. So before writing, the stage asks
``census.download_activity`` whether anything is being written, and refuses if
so. Override with ``curate.allow_active_download: true`` when the running
download is known to touch a different class than the one being planned.

Everything numeric lives in ``scripts/select_genomes.py`` (pure functions, unit
tested without a corpus). This file is the adapter: find the input table, build
the taxonomy, call select(), write two artifacts, log a readable summary.
"""
from __future__ import annotations

import glob
import os
import importlib.util
import json
from pathlib import Path

from .. import census as _census
from .. import paths, taxonomy as _taxonomy
from ..stage import Stage, register

#: Filenames the acquire/select step is known to produce, best first. The stage
#: autodiscovers rather than demanding a config entry, so it works against an
#: existing tree with no config edits at all.
INPUT_CANDIDATES = (
    "selected_genomes.tsv", "selection.tsv", "download_all.tsv",
    "genome_index.tsv", "index.tsv", "assembly_index.tsv",
    "assembly_summary.txt", "assembly_summary_refseq.txt",
    "assembly_summary_genbank.txt",
)

#: Directories searched for those files, relative to base/data_home.
INPUT_DIRS = ("", "metadata", "index", "acquire", "ncbi", "ncbi/metadata",
              "downloads", "tables")


def _load_selector():
    """Import scripts/select_genomes.py by path.

    ``scripts/`` is not a package (it is a flat directory of standalone tools),
    so a normal import would fail. Loading by path keeps the selection logic
    where the other data-shaping scripts live instead of hiding it in the
    package purely to satisfy the import system.
    """
    path = paths.repo_root() / "scripts" / "select_genomes.py"
    spec = importlib.util.spec_from_file_location("_tiara2_select_genomes", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_input_tables(cfg: dict) -> list:
    """Locate every candidate-genome table. Explicit config always wins.

    Returns a list because NCBI splits assembly_summary by division: the
    eukaryotic candidate set on this box is ten files (fungi, protozoa, plant,
    invertebrate, vertebrate_mammalian, vertebrate_other x refseq/genbank).
    Accepting only one of them would silently curate a fraction of the tree,
    which is the exact failure mode we are trying to fix.

    ``curate.input_tables`` accepts paths and globs; ``curate.input_table``
    stays supported for the single-file case.
    """
    cur = (cfg.get("curate", {}) or {})
    listed = cur.get("input_tables") or []
    if isinstance(listed, str):
        listed = [listed]
    explicit_one = str(cur.get("input_table", "") or "").strip()
    if explicit_one:
        listed = list(listed) + [explicit_one]
    if listed:
        out = []
        for item in listed:
            hits = sorted(glob.glob(os.path.expanduser(str(item))))
            out.extend(Path(h) for h in hits if os.path.isfile(h))
        return out
    found = _autodiscover(cfg)
    return [found] if found else []


def find_input_table(cfg: dict):
    """Back-compat single-table accessor (first match, or None)."""
    tables = find_input_tables(cfg)
    return tables[0] if tables else None


def _autodiscover(cfg: dict):
    cur = (cfg.get("curate", {}) or {})
    explicit = str(cur.get("input_table", "") or "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.exists() else None
    roots = [Path(str(cfg.get("base", "."))),
             Path(str(cfg.get("data_home", ".")))]
    for root in roots:
        for sub in INPUT_DIRS:
            for name in INPUT_CANDIDATES:
                cand = (root / sub / name) if sub else (root / name)
                try:
                    if cand.is_file():
                        return cand
                except OSError:
                    continue
    return None


def load_anchors(cfg: dict) -> list:
    """Accessions that must survive curation regardless of gates or quotas.

    These are the cleaned Tiara1 originals. They are the only genomes in the
    corpus with a published F1 attached to them, so no heuristic of ours gets
    to remove them; if our rules would drop one, that is evidence about the
    rules, and the report records it as an override.
    """
    cur = (cfg.get("curate", {}) or {})
    inline = cur.get("anchors") or []
    out = [str(a).strip() for a in inline if str(a).strip()]
    path = str(cur.get("anchor_file", "") or "").strip()
    if path:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = paths.repo_root() / p
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.split("#")[0].strip()
                if line:
                    out.append(line.split("\t")[0].split(",")[0].strip())
        except OSError:
            pass
    return sorted(set(out))


@register("curate")
class CurateStage(Stage):
    param_keys = ("curate", "classes")

    def _out_dir(self, ctx) -> Path:
        cur = (ctx.cfg.get("curate", {}) or {})
        explicit = str(cur.get("out_dir", "") or "").strip()
        return Path(explicit) if explicit else Path(ctx.work_dir)

    def inputs(self, ctx):
        return [str(p) for p in find_input_tables(ctx.cfg)]

    def outputs(self, ctx):
        out = self._out_dir(ctx)
        return [str(out / "curation_selection.tsv"),
                str(out / "curation_report.json")]

    def run(self, ctx):
        cur = (ctx.cfg.get("curate", {}) or {})
        if not _as_bool(cur.get("enabled"), True):
            ctx.log.info("curate SKIPPED (curate.enabled=false)")
            return {"counts": {"skipped": 1}}

        # 1. Census FIRST. The user-facing requirement is "confirm the current
        #    per-class volumes before deploying", and it is also the only way
        #    to know whether the plan is being computed against a stable tree.
        report_census = _as_bool(cur.get("census"), True)
        activity = None
        if report_census:
            snapshot = _census.run_census(
                ctx.cfg,
                exact=_as_bool(cur.get("census_exact"), False),
                sample_mb=int(cur.get("census_sample_mb",
                                      _census.DEFAULT_SAMPLE_MB)),
                window_s=float(cur.get("active_window_s",
                                       _census.DEFAULT_ACTIVE_WINDOW)))
            activity = snapshot["activity"]
            for line in _census.render_census(snapshot["census"],
                                              snapshot["projection"],
                                              activity).splitlines():
                ctx.log.info("%s", line)

        # 2. Freeze guard.
        if activity and activity["active"] and not _as_bool(
                cur.get("allow_active_download"), False):
            ctx.log.warning(
                "curate: a download appears to be RUNNING (%d partial files, "
                "%d files touched in the last %.0fs). Not freezing a "
                "selection against a moving corpus. Re-run when it settles, "
                "or override with --set curate.allow_active_download=true",
                activity["partial_count"], activity["recent_count"],
                activity["window_s"])
            return {"counts": {"deferred_download_active": 1}}

        # 3. Input table.
        tables = find_input_tables(ctx.cfg)
        if not tables:
            ctx.log.warning(
                "curate: no candidate-genome table found under %s or %s "
                "(looked for %s). Point at one with "
                "--set curate.input_table=/path/to/assembly_summary.txt",
                ctx.cfg.get("base"), ctx.cfg.get("data_home"),
                ", ".join(INPUT_CANDIDATES[:4]))
            return {"counts": {"no_input_table": 1}}
        ctx.log.info("curate: %d input table(s)", len(tables))
        for t in tables[:20]:
            ctx.log.info("curate:   %s", t)

        # 4. Taxonomy. Never fatal: the lineage backend keeps the stage usable
        #    before taxdump has been fetched, and the backend is recorded.
        tax = _taxonomy.build(ctx.cfg)
        if tax.backend == "taxdump":
            ctx.log.info("curate: taxonomy backend=taxdump (%d nodes)",
                         len(tax.nodes))
        else:
            ctx.log.warning(
                "curate: no taxdump found -- falling back to lineage-string "
                "parsing. Genus grouping will be approximate. Fetch "
                "taxdump.tar.gz and set curate.taxdump_dir for exact ranks.")

        selector = _load_selector()
        rows = selector.read_tables([str(t) for t in tables])
        ctx.log.info("curate: %d candidate assemblies (deduplicated)",
                     len(rows))

        anchors = load_anchors(ctx.cfg)
        if anchors:
            ctx.log.info("curate: %d anchor accessions (never dropped)",
                         len(anchors))

        # Mode. breadth = one genome per genus, no global fragment budget
        # (maximise coverage of the tree); balanced = Tiara-style class and
        # clade budget. They answer different questions, so the choice is
        # explicit rather than inferred.
        mode = str(cur.get("mode", "breadth") or "breadth").strip().lower()
        if mode not in ("breadth", "balanced"):
            ctx.log.warning("curate: unknown mode %r, using breadth", mode)
            mode = "breadth"
        runner = (selector.select_breadth if mode == "breadth"
                  else selector.select)
        only = cur.get("clades_only") or []
        ctx.log.info("curate: mode=%s%s", mode,
                     (" clades_only=" + ",".join(map(str, only))) if only else "")
        selection, report = runner(rows, tax, cur, anchors=anchors)

        out_dir = self._out_dir(ctx)
        out_dir.mkdir(parents=True, exist_ok=True)
        sel_path = out_dir / "curation_selection.tsv"
        rep_path = out_dir / "curation_report.json"
        selector.write_selection(str(sel_path), selection)
        report["input_tables"] = [str(t) for t in tables]
        report["mode"] = mode
        report["anchors"] = len(anchors)
        if activity:
            report["download_activity"] = {
                k: v for k, v in activity.items()
                if k in ("active", "partial_count", "recent_count", "window_s")}
        rep_path.write_text(json.dumps(report, indent=2, sort_keys=True),
                            encoding="utf-8")

        for line in render_report(report).splitlines():
            ctx.log.info("%s", line)
        ctx.log.info("curate: selection -> %s", sel_path)
        ctx.log.info("curate: report    -> %s", rep_path)

        counts = {"candidates": len(rows),
                  "selected": len(selection),
                  "rejected": report["rejected_by_gates"],
                  "clades": len(report["clades"])}
        if "genera_selected" in report:
            counts["genera"] = report["genera_selected"]
        if report.get("anchors_missing"):
            # Loud on purpose: a missing anchor means the accession is not in
            # any input table, so the euk set is not the superset of Tiara's
            # that we believe it to be.
            ctx.log.warning("curate: %d anchor accessions NOT FOUND in the "
                            "input tables: %s", len(report["anchors_missing"]),
                            ", ".join(report["anchors_missing"][:10]))
            counts["anchors_missing"] = len(report["anchors_missing"])
        return {"counts": counts,
                "versions": {"taxonomy_backend": tax.backend,
                             "curate_mode": mode}}


def render_report(report: dict) -> str:
    """Per-clade table: what was available, what was kept, at what depth.

    Mode-aware. In ``balanced`` mode the last two columns are the clade's
    allocated depth and fragment budget. In ``breadth`` mode there is no clade
    budget at all -- that is the defining property of the mode -- so they show
    the mean planned depth and the resulting fragment count instead. Printing
    a budget column of zeros would read as "the budget starved this clade",
    which is the opposite of what breadth mode actually did.
    """
    breadth = report.get("mode") == "breadth"
    last = "planned frags" if breadth else "budget frags"
    head = ("  " + "clade".ljust(22) + "genera".rjust(9) + "eligible".rjust(10)
            + "kept".rjust(8) + "anchors".rjust(9) + "bp/genome".rjust(12)
            + last.rjust(14))
    lines = ["curation plan (mode=%s, taxonomy=%s)"
             % (report.get("mode", "balanced"), report.get("taxonomy_backend")),
             head, "  " + "-" * (len(head) - 2)]
    for clade, row in sorted(report.get("clades", {}).items()):
        kept = row.get("genomes_selected", 0)
        if "bp_per_genome" in row:
            per = row["bp_per_genome"]
        else:
            per = (row.get("planned_bp", 0) / kept) if kept else 0
        frags = row.get("budget_fragments", row.get("planned_fragments", 0))
        lines.append(
            "  " + clade.ljust(22)
            + f"{row['genera_available']:,}".rjust(9)
            + f"{row['genomes_eligible']:,}".rjust(10)
            + f"{kept:,}".rjust(8)
            + f"{row['anchors']:,}".rjust(9)
            + f"{per / 1e6:,.1f}M".rjust(12)
            + f"{frags:,}".rjust(14))
        if row.get("genomes_dropped_for_depth"):
            lines.append(
                f"      note: {row['genomes_dropped_for_depth']:,} genomes "
                "dropped so the rest keep min depth (depth beats breadth)")
    lines.append("  selected %s of %s candidates; %s rejected by quality gates"
                 % (f"{report.get('selected', 0):,}",
                    f"{report.get('rows_in', 0):,}",
                    f"{report.get('rejected_by_gates', 0):,}"))
    if report.get("genera_selected"):
        lines.append("  covering %s genera in %s families"
                     % (f"{report['genera_selected']:,}",
                        f"{report.get('families_selected', 0):,}"))
    if report.get("clades_skipped"):
        lines.append("  clades not curated this run (clades_only): %s"
                     % ", ".join(sorted(report["clades_skipped"])))
    if report.get("anchors_missing"):
        lines.append("  !! %s Tiara1 anchor accessions are ABSENT from the "
                     "input tables" % f"{len(report['anchors_missing']):,}")
    if report.get("unresolved_taxids"):
        lines.append("  !! %s selected genomes had unresolvable taxids and "
                     "were grouped by name" % f"{report['unresolved_taxids']:,}")
    return "\n".join(lines)


_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off", ""}


def _as_bool(value, default=True):
    """Config values arrive as real bools or as --set strings."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUTHY:
        return True
    if text in _FALSY:
        return False
    return default

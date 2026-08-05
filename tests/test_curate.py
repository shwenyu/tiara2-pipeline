"""curate: genus-level breadth + per-genome depth.

Invariants protected here:
  * a taxid resolves to the RIGHT clade, and Vertebrata is tested before
    Metazoa (otherwise every vertebrate becomes an invertebrate);
  * one genome per genus by default, more ONLY for a heterogeneous genus;
  * the best assembly in a genus wins (reference > representative, complete >
    contig, fewer contigs, newer);
  * a big genome cannot exceed its per-genome bp cap -- the mechanism that
    stops one vertebrate out-producing an entire clade;
  * anchors survive quality gates and quotas, and the override is recorded;
  * when a clade budget cannot fund min depth for everyone, GENOMES are
    dropped (family-stratified), depth is never lowered;
  * the stage defers instead of freezing a plan while a download is live;
  * the shipped config still sums its clade shares to 1.0.
"""
import importlib.util
import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tiara2 import taxonomy as T  # noqa: E402
from tiara2.stages import curate as C  # noqa: E402


def _load_selector():
    path = ROOT / "scripts" / "select_genomes.py"
    spec = importlib.util.spec_from_file_location("_sel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _load_selector()

# A miniature nodes.dmp/names.dmp: human, mouse, a fly, a fungus, a plant and
# a Plasmodium, each with a genus and family node above it.
MINI_NODES = {
    1: (1, "no rank"), 131567: (1, "no rank"), 2759: (131567, "superkingdom"),
    33208: (2759, "kingdom"), 33511: (33208, "clade"), 7711: (33511, "phylum"),
    7742: (7711, "clade"), 40674: (7742, "class"), 9443: (40674, "order"),
    9604: (9443, "family"), 9605: (9604, "genus"), 9606: (9605, "species"),
    9607: (9605, "species"),
    10066: (7742, "family"), 10088: (10066, "genus"), 10090: (10088, "species"),
    50557: (33208, "class"), 7214: (50557, "family"),
    7215: (7214, "genus"), 7227: (7215, "species"),
    4751: (2759, "kingdom"), 4892: (4751, "family"),
    4930: (4892, "genus"), 4932: (4930, "species"), 4933: (4930, "species"),
    33090: (2759, "kingdom"), 3193: (33090, "clade"), 3700: (3193, "family"),
    3701: (3700, "genus"), 3702: (3701, "species"),
    33630: (2759, "clade"), 5819: (33630, "family"),
    5820: (5819, "genus"), 5833: (5820, "species"), 5834: (5820, "species"),
}
MINI_NAMES = {
    2759: "Eukaryota", 33208: "Metazoa", 7742: "Vertebrata", 9605: "Homo",
    9604: "Hominidae", 9606: "Homo sapiens", 9607: "Homo other",
    10088: "Mus", 10090: "Mus musculus", 10066: "Muridae",
    7215: "Drosophila", 7227: "Drosophila melanogaster", 7214: "Drosophilidae",
    4930: "Saccharomyces", 4932: "Saccharomyces cerevisiae",
    4933: "Saccharomyces paradoxus", 4892: "Saccharomycetaceae",
    3701: "Arabidopsis", 3702: "Arabidopsis thaliana", 3700: "Brassicaceae",
    5820: "Plasmodium", 5833: "Plasmodium falciparum",
    5834: "Plasmodium berghei", 5819: "Plasmodiidae", 3193: "Embryophyta",
    4751: "Fungi", 33630: "Alveolata",
}


def mini_tax():
    return T.Taxonomy(nodes=MINI_NODES, names=MINI_NAMES, backend="taxdump")


def row(acc, taxid, **kw):
    base = {
        "assembly_accession": acc, "taxid": str(taxid),
        "organism_name": MINI_NAMES.get(taxid, "?"),
        "refseq_category": "na", "assembly_level": "Complete Genome",
        "genome_rep": "Full", "excluded_from_refseq": "",
        "version_status": "latest", "genome_size": "30000000",
        "gc_percent": "40", "contig_count": "100",
        "seq_rel_date": "2020/01/01", "ftp_path": "",
    }
    base.update({k: str(v) for k, v in kw.items()})
    return base


BASE_CFG = {
    "genus": {"max_per_genus": 4, "exponent": 0.4, "size_ratio": 3.0,
              "gc_spread_pp": 8.0},
    "budget": {"total_fragments": 40_000_000, "eukarya_share": 0.3167,
               "mean_fragment_bp": 4200, "min_bp_per_genome": 5_000_000,
               "max_bp_per_genome": 50_000_000, "max_genome_fraction": 0.8},
    "clade_share": {"fungi": 0.2, "alveolata": 0.12, "stramenopiles": 0.10,
                    "other_protist": 0.12, "algae": 0.08,
                    "metazoa_invertebrate": 0.15, "land_plant": 0.13,
                    "metazoa_vertebrate": 0.10},
    "quality_gates": {"allowed_levels": ["Complete Genome", "Chromosome",
                                          "Scaffold"],
                      "require_full_genome_rep": True,
                      "exclude_flagged": True,
                      "min_genome_size": 1_000_000},
}


class TestTaxonomy(unittest.TestCase):
    def test_clade_assignment(self):
        tax = mini_tax()
        cases = {9606: "metazoa_vertebrate", 10090: "metazoa_vertebrate",
                 7227: "metazoa_invertebrate", 4932: "fungi",
                 3702: "land_plant", 5833: "alveolata"}
        for taxid, clade in cases.items():
            self.assertEqual(tax.info(taxid).clade, clade, f"taxid {taxid}")

    def test_vertebrata_beats_metazoa(self):
        """Declaration order is load-bearing, not cosmetic."""
        names = [c[0] for c in T.DEFAULT_CLADES]
        self.assertLess(names.index("metazoa_vertebrate"),
                        names.index("metazoa_invertebrate"))

    def test_genus_and_family(self):
        info = mini_tax().info(9606)
        self.assertEqual(info.genus_name, "Homo")
        self.assertEqual(info.family_name, "Hominidae")
        self.assertTrue(info.ok)

    def test_same_genus_shares_key(self):
        tax = mini_tax()
        self.assertEqual(tax.info(4932).genus_key, tax.info(4933).genus_key)
        self.assertNotEqual(tax.info(4932).genus_key, tax.info(9606).genus_key)

    def test_unknown_taxid_is_its_own_group(self):
        """An unresolvable taxid must not merge with every other one."""
        tax = mini_tax()
        a = tax.info(999999, organism="Unknownus alpha")
        b = tax.info(888888, organism="Otherus beta")
        self.assertFalse(a.ok)
        self.assertNotEqual(a.genus_key, b.genus_key)

    def test_lineage_backend(self):
        tax = T.Taxonomy.from_lineages()
        info = tax.info(0, lineage="Eukaryota; Opisthokonta; Fungi; Dikarya; "
                                   "Ascomycota; Saccharomyces")
        self.assertEqual(info.clade, "fungi")
        self.assertEqual(info.genus_name, "Saccharomyces")

    def test_cycle_safe(self):
        tax = T.Taxonomy(nodes={5: (6, "genus"), 6: (5, "family")},
                         names={}, backend="taxdump")
        self.assertEqual(len(list(tax.ancestors(5))), 2)


class TestQuality(unittest.TestCase):
    def test_reference_beats_representative(self):
        best = row("GCA_1", 9606, refseq_category="reference genome")
        rest = row("GCA_2", 9606, refseq_category="representative genome")
        self.assertLess(S.quality_key(best), S.quality_key(rest))

    def test_complete_beats_contig(self):
        a = row("GCA_1", 9606, assembly_level="Complete Genome")
        b = row("GCA_2", 9606, assembly_level="Contig")
        self.assertLess(S.quality_key(a), S.quality_key(b))

    def test_fewer_contigs_wins(self):
        a = row("GCA_1", 9606, contig_count=20)
        b = row("GCA_2", 9606, contig_count=9000)
        self.assertLess(S.quality_key(a), S.quality_key(b))

    def test_newer_breaks_ties(self):
        a = row("GCA_1", 9606, seq_rel_date="2023/05/01")
        b = row("GCA_2", 9606, seq_rel_date="2011/05/01")
        self.assertLess(S.quality_key(a), S.quality_key(b))

    def test_gates_reject_partial(self):
        ok, reasons = S.passes_gates(row("GCA_1", 9606, genome_rep="Partial"),
                                     BASE_CFG["quality_gates"])
        self.assertFalse(ok)
        self.assertIn("partial_genome_rep", reasons)

    def test_gates_reject_flagged(self):
        ok, reasons = S.passes_gates(
            row("GCA_1", 9606, excluded_from_refseq="derived from metagenome"),
            BASE_CFG["quality_gates"])
        self.assertFalse(ok)
        self.assertIn("excluded_from_refseq", reasons)

    def test_gates_accept_clean(self):
        ok, _ = S.passes_gates(row("GCA_1", 9606), BASE_CFG["quality_gates"])
        self.assertTrue(ok)


class TestGenusQuota(unittest.TestCase):
    def test_homogeneous_genus_gets_one(self):
        rows = [row(f"GCA_{i}", 4932, gc_percent=38) for i in range(30)]
        self.assertEqual(S.intra_genus_quota(rows, BASE_CFG["genus"]), 1)

    def test_size_spread_earns_more(self):
        rows = [row("GCA_1", 4932, genome_size=10_000_000),
                row("GCA_2", 4933, genome_size=90_000_000)]
        self.assertGreater(S.intra_genus_quota(rows, BASE_CFG["genus"]), 1)

    def test_gc_spread_earns_more(self):
        rows = [row("GCA_1", 5833, gc_percent=19),
                row("GCA_2", 5834, gc_percent=42)]
        self.assertGreater(S.intra_genus_quota(rows, BASE_CFG["genus"]), 1)

    def test_quota_is_capped(self):
        rows = [row(f"GCA_{i}", 5820 + (i % 2), gc_percent=15 + 30 * (i % 2),
                    species_taxid=90000 + i) for i in range(400)]
        self.assertLessEqual(S.intra_genus_quota(rows, BASE_CFG["genus"]), 4)


class TestDepth(unittest.TestCase):
    def test_cap_applies(self):
        bp = S.per_genome_bp(10_000_000_000, 10, BASE_CFG["budget"],
                             genome_size=3_100_000_000)
        self.assertEqual(bp, 50_000_000)

    def test_floor_applies(self):
        bp = S.per_genome_bp(1_000_000, 100, BASE_CFG["budget"])
        self.assertEqual(bp, 5_000_000)

    def test_small_genome_capped_by_its_own_size(self):
        """Never claim more bp than 80% of the assembly actually has."""
        bp = S.per_genome_bp(10_000_000_000, 10, BASE_CFG["budget"],
                             genome_size=12_000_000)
        self.assertEqual(bp, 9_600_000)

    def test_vertebrate_cannot_swallow_its_clade(self):
        """The regression that flattened v2.0's posteriors.

        Uncapped, one 3.1 Gb genome at Tiara's 0.79x depth yields ~620k
        fragments -- more than twice Tiara's ENTIRE eukaryotic class.
        """
        budget = S.allocate_clade_budget(40_000_000, 0.3167,
                                         BASE_CFG["clade_share"], 4200)
        bp = S.per_genome_bp(budget["metazoa_vertebrate"]["bp"], 500,
                             BASE_CFG["budget"], genome_size=3_100_000_000)
        self.assertLessEqual(bp / 4200, 620_000 / 2)

    def test_clade_budget_sums_to_class_budget(self):
        budget = S.allocate_clade_budget(40_000_000, 0.3167,
                                         BASE_CFG["clade_share"], 4200)
        total = sum(v["fragments"] for v in budget.values())
        self.assertAlmostEqual(total / (40_000_000 * 0.3167), 1.0, places=3)


class TestSelect(unittest.TestCase):
    def test_one_per_genus_and_best_wins(self):
        rows = [row("GCA_bad", 9606, assembly_level="Contig",
                    contig_count=50000),
                row("GCA_good", 9607, refseq_category="reference genome"),
                row("GCA_mouse", 10090)]
        sel, rep = S.select(rows, mini_tax(), BASE_CFG)
        accs = {r["assembly_accession"] for r in sel}
        self.assertIn("GCA_good", accs)
        self.assertNotIn("GCA_bad", accs)     # same genus (Homo), worse
        self.assertIn("GCA_mouse", accs)      # different genus
        self.assertEqual(rep["clades"]["metazoa_vertebrate"]["genera_available"], 2)

    def test_every_selection_has_a_bp_target(self):
        rows = [row("GCA_1", 9606), row("GCA_2", 4932), row("GCA_3", 3702)]
        sel, _ = S.select(rows, mini_tax(), BASE_CFG)
        for r in sel:
            self.assertGreater(r["_target_bp"], 0)
            self.assertGreater(r["_target_fragments"], 0)

    def test_anchor_survives_a_failing_gate(self):
        rows = [row("GCA_anchor", 4932, genome_rep="Partial",
                    assembly_level="Contig")]
        sel, rep = S.select(rows, mini_tax(), BASE_CFG,
                            anchors=["GCA_anchor"])
        self.assertEqual(len(sel), 1)
        self.assertTrue(sel[0]["_anchor"])
        self.assertIn("_anchor_override", sel[0])
        self.assertEqual(rep["rejected_by_gates"], 0)

    def test_anchor_matches_without_version_suffix(self):
        rows = [row("GCA_000818905.1", 4932)]
        sel, _ = S.select(rows, mini_tax(), BASE_CFG,
                          anchors=["GCA_000818905"])
        self.assertTrue(sel[0]["_anchor"])

    def test_anchor_beats_a_better_genome_in_its_genus(self):
        rows = [row("GCA_anchor", 4932, assembly_level="Scaffold",
                    contig_count=9000),
                row("GCA_better", 4933, refseq_category="reference genome")]
        sel, _ = S.select(rows, mini_tax(), BASE_CFG, anchors=["GCA_anchor"])
        self.assertIn("GCA_anchor", {r["assembly_accession"] for r in sel})

    def test_depth_beats_breadth(self):
        """A budget too small for everyone drops genomes, never depth."""
        cfg = json.loads(json.dumps(BASE_CFG))
        cfg["budget"]["total_fragments"] = 100_000     # deliberately tiny
        rows = [row(f"GCA_{i}", 4932, species_taxid=70000 + i,
                    gc_percent=20 + (i % 20)) for i in range(50)]
        # give each row its own genus so breadth is what gets trimmed
        for i, r in enumerate(rows):
            r["taxid"] = str(600000 + i)
        sel, rep = S.select(rows, T.Taxonomy.from_lineages(), cfg)
        for r in sel:
            self.assertGreaterEqual(r["_target_bp"],
                                    cfg["budget"]["min_bp_per_genome"] * 0.99)
        self.assertLess(len(sel), 50)

    def test_trim_is_family_stratified(self):
        """Trimming must not amputate whole families."""
        rows = []
        for fam in range(4):
            for i in range(10):
                r = row(f"GCA_{fam}_{i}", 4932)
                r["lineage"] = f"Eukaryota; Fungi; Fam{fam}; Gen{fam}_{i}"
                rows.append(r)
        cfg = json.loads(json.dumps(BASE_CFG))
        cfg["budget"]["total_fragments"] = 200_000
        sel, _ = S.select(rows, T.Taxonomy.from_lineages(), cfg)
        families = {r["_family"] for r in sel}
        self.assertGreater(len(families), 1, "trim wiped out entire families")

    def test_report_records_the_backend(self):
        _, rep = S.select([row("GCA_1", 9606)], mini_tax(), BASE_CFG)
        self.assertEqual(rep["taxonomy_backend"], "taxdump")

    def test_deterministic(self):
        rows = [row(f"GCA_{i}", 4932 + (i % 2)) for i in range(20)]
        a, _ = S.select(list(rows), mini_tax(), BASE_CFG)
        b, _ = S.select(list(rows), mini_tax(), BASE_CFG)
        self.assertEqual([r["assembly_accession"] for r in a],
                         [r["assembly_accession"] for r in b])


class Ctx:
    def __init__(self, cfg, work_dir):
        self.cfg = cfg
        self.work_dir = Path(work_dir)
        self.log = logging.getLogger("test")
        self.dry_run = False
        self.force = False
        self.debug = False
        self.extra = {}


class TestStage(unittest.TestCase):
    def _cfg(self, base, **curate):
        cfg = {"base": str(base), "data_home": str(base),
               "source_ready": str(base / "src"),
               "work_root": str(base / "work"),
               "splits": ["train"], "classes": ["eukarya"],
               "curate": dict(BASE_CFG)}
        cfg["curate"].update({"enabled": True, "census": False,
                              "anchor_file": ""})
        cfg["curate"].update(curate)
        return cfg

    def _seed_table(self, base):
        path = base / "metadata" / "assembly_summary.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = ["assembly_accession", "taxid", "organism_name",
                "refseq_category", "assembly_level", "genome_rep",
                "excluded_from_refseq", "version_status", "genome_size",
                "gc_percent", "contig_count", "seq_rel_date", "ftp_path"]
        lines = ["# comment line", "#" + "\t".join(cols)]
        for acc, taxid in (("GCA_1", 9606), ("GCA_2", 4932), ("GCA_3", 3702)):
            r = row(acc, taxid)
            lines.append("\t".join(r[c] for c in cols))
        path.write_text("\n".join(lines) + "\n")
        return path

    def test_writes_a_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._seed_table(base)
            ctx = Ctx(self._cfg(base), base / "work" / "curate")
            ctx.work_dir.mkdir(parents=True)
            out = C.CurateStage().run(ctx)
            self.assertEqual(out["counts"]["candidates"], 3)
            self.assertTrue((ctx.work_dir / "curation_selection.tsv").exists())
            report = json.loads(
                (ctx.work_dir / "curation_report.json").read_text())
            self.assertIn("clades", report)

    def test_disabled_is_a_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ctx = Ctx(self._cfg(base, enabled=False), base / "w")
            self.assertEqual(C.CurateStage().run(ctx)["counts"]["skipped"], 1)

    def test_missing_table_warns_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ctx = Ctx(self._cfg(base), base / "w")
            ctx.work_dir.mkdir(parents=True)
            out = C.CurateStage().run(ctx)
            self.assertEqual(out["counts"]["no_input_table"], 1)

    def test_defers_while_a_download_is_live(self):
        """The whole point of the freeze guard."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._seed_table(base)
            src = base / "src" / "train"
            src.mkdir(parents=True)
            (src / "eukarya.fasta.part").write_text(">x\nACGT\n")
            cfg = self._cfg(base, census=True)
            ctx = Ctx(cfg, base / "work" / "curate")
            ctx.work_dir.mkdir(parents=True)
            out = C.CurateStage().run(ctx)
            self.assertEqual(out["counts"]["deferred_download_active"], 1)
            self.assertFalse((ctx.work_dir / "curation_selection.tsv").exists())

    def test_override_forces_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._seed_table(base)
            src = base / "src" / "train"
            src.mkdir(parents=True)
            (src / "eukarya.fasta.part").write_text(">x\nACGT\n")
            cfg = self._cfg(base, census=True, allow_active_download=True)
            ctx = Ctx(cfg, base / "work" / "curate")
            ctx.work_dir.mkdir(parents=True)
            out = C.CurateStage().run(ctx)
            self.assertIn("selected", out["counts"])


class TestShippedConfig(unittest.TestCase):
    def setUp(self):
        from tiara2 import config as _config
        self.cfg = _config.load(str(ROOT / "config" / "config.yaml"))

    def test_curate_is_in_the_corpus_layer_right_after_acquire(self):
        from tiara2 import cli
        self.assertEqual(cli.DEFAULT_ORDER.index("curate"),
                         cli.DEFAULT_ORDER.index("acquire") + 1)
        self.assertLess(cli.DEFAULT_ORDER.index("curate"),
                        cli.DEFAULT_ORDER.index("chop_bin"))
        self.assertIn("curate", cli.CORPUS_STAGES)

    def test_original_order_is_otherwise_untouched(self):
        from tiara2 import cli
        kept = [s for s in cli.DEFAULT_ORDER if s != "curate"]
        self.assertEqual(kept, ["acquire", "chop_bin", "dedup", "regroup",
                                "train", "publish"])

    def test_clade_shares_sum_to_one(self):
        total = sum(self.cfg["curate"]["clade_share"].values())
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_priors_match_tiara_s3(self):
        from tiara2 import census
        for cls, val in census.TIARA_S3_PRIORS.items():
            self.assertAlmostEqual(self.cfg["curate"]["target_priors"][cls],
                                   val, places=6)

    def test_anchor_file_ships_and_parses(self):
        cfg = dict(self.cfg)
        anchors = C.load_anchors(cfg)
        self.assertGreaterEqual(len(anchors), 60)
        self.assertTrue(all(a.startswith(("GCA_", "GCF_")) for a in anchors),
                        anchors[:5])

    def test_multicellular_clades_are_present(self):
        """Tiara1 had no vertebrates or land plants; Tiara2 must."""
        share = self.cfg["curate"]["clade_share"]
        for clade in ("metazoa_vertebrate", "metazoa_invertebrate",
                      "land_plant"):
            self.assertGreater(share.get(clade, 0), 0)



class TestBreadth(unittest.TestCase):
    """Breadth mode: coverage of the tree, not Tiara's class proportions.

    The distinction that matters: ``select`` will DROP genomes when a clade's
    fragment budget cannot fund them at depth. ``select_breadth`` must never
    do that -- dropping a genus is exactly the outcome breadth mode exists to
    prevent. What it must still do is cap the biggest genomes, because an
    uncapped vertebrate erases the rare genera numerically even though they
    are present in the table.
    """

    def _rows(self):
        return [
            row("GCF_1", 9606, genome_size=3_100_000_000),   # Homo
            row("GCF_2", 10090, genome_size=2_700_000_000),  # Mus
            row("GCF_3", 7227, genome_size=140_000_000),     # Drosophila
            row("GCF_4", 4932, genome_size=12_000_000),      # Saccharomyces
            row("GCF_5", 3702, genome_size=135_000_000),     # Arabidopsis
            row("GCF_6", 5833, genome_size=23_000_000),      # Plasmodium
        ]

    def test_one_genome_per_genus(self):
        sel, rep = S.select_breadth(self._rows(), mini_tax(), {})
        self.assertEqual(len(sel), 6)
        self.assertEqual(rep["genera_selected"], 6)
        self.assertEqual(rep["mode"], "breadth")

    def test_duplicate_genus_collapses_to_one(self):
        rows = self._rows() + [row("GCF_7", 9606, genome_size=3_000_000_000)]
        sel, rep = S.select_breadth(rows, mini_tax(), {})
        self.assertEqual(rep["genera_selected"], 6)
        self.assertEqual(len(sel), 6, "same genus must not yield two genomes")

    def test_best_genome_wins_inside_a_genus(self):
        rows = [row("GCF_bad", 9606, assembly_level="Scaffold"),
                row("GCF_good", 9606, refseq_category="reference genome")]
        sel, _ = S.select_breadth(rows, mini_tax(), {})
        self.assertEqual(len(sel), 1)
        self.assertEqual(sel[0]["assembly_accession"], "GCF_good")

    def test_nothing_is_dropped_for_depth(self):
        """The whole point: no global budget, so no budget-driven casualties."""
        rows = self._rows()
        _, rep = S.select_breadth(rows, mini_tax(), {})
        for clade, info in rep["clades"].items():
            self.assertEqual(info["genomes_dropped_for_depth"], 0, clade)

    def test_budget_mode_would_have_dropped_them(self):
        """Contrast test -- proves the two modes really differ.

        Two vertebrate genera and a budget that can fund only one of them at
        minimum depth. ``select`` must drop one; ``select_breadth`` must keep
        both. The clade_share has to name the clade the rows actually live in:
        a clade with no share gets a zero budget, and a zero budget disables
        the cap instead of tightening it.
        """
        rows = [row("GCF_1", 9606, genome_size=3_100_000_000),
                row("GCF_2", 10090, genome_size=2_700_000_000)]
        cfg = {"budget": {"total_fragments": 1000, "eukarya_share": 1.0,
                          "mean_fragment_bp": 4200,
                          "min_bp_per_genome": 5_000_000},
               "clade_share": {"metazoa_vertebrate": 1.0}}
        _, tight = S.select([dict(r) for r in rows], mini_tax(), cfg)
        _, wide = S.select_breadth([dict(r) for r in rows], mini_tax(), cfg)
        self.assertEqual(tight["selected"], 1)
        self.assertEqual(wide["selected"], 2)

    def test_large_genome_is_still_capped(self):
        cfg = {"budget": {"max_bp_per_genome": 50_000_000,
                          "max_genome_fraction": 0.8,
                          "mean_fragment_bp": 4200}}
        sel, _ = S.select_breadth(self._rows(), mini_tax(), cfg)
        human = [r for r in sel if r["assembly_accession"] == "GCF_1"][0]
        self.assertEqual(human["_target_bp"], 50_000_000)
        self.assertLess(human["_target_fragments"], 12_000)

    def test_small_genome_is_capped_by_its_own_size(self):
        cfg = {"budget": {"max_bp_per_genome": 50_000_000,
                          "max_genome_fraction": 0.8,
                          "mean_fragment_bp": 4200}}
        sel, _ = S.select_breadth(self._rows(), mini_tax(), cfg)
        yeast = [r for r in sel if r["assembly_accession"] == "GCF_4"][0]
        self.assertEqual(yeast["_target_bp"], int(12_000_000 * 0.8))

    def test_cap_keeps_the_vertebrate_from_swamping_the_rest(self):
        cfg = {"budget": {"max_bp_per_genome": 50_000_000,
                          "max_genome_fraction": 0.8,
                          "mean_fragment_bp": 4200}}
        sel, _ = S.select_breadth(self._rows(), mini_tax(), cfg)
        by = {r["assembly_accession"]: r["_target_fragments"] for r in sel}
        total = sum(by.values())
        self.assertLess(by["GCF_1"] / total, 0.45,
                        "one vertebrate must not dominate the plan")

    def test_clades_only_restricts_the_run(self):
        cfg = {"clades_only": ["fungi"]}
        sel, rep = S.select_breadth(self._rows(), mini_tax(), cfg)
        self.assertEqual([r["assembly_accession"] for r in sel], ["GCF_4"])
        self.assertIn("metazoa_vertebrate", rep["clades_skipped"])
        self.assertNotIn("metazoa_vertebrate", rep["clades"])

    def test_anchor_is_kept_even_when_it_fails_a_gate(self):
        rows = [row("GCF_anchor", 4932, assembly_level="Contig"),
                row("GCF_other", 4932)]
        cfg = {"quality_gates": {"allowed_levels": ["Complete Genome"]}}
        sel, rep = S.select_breadth(rows, mini_tax(), cfg,
                                    anchors=["GCF_anchor"])
        accs = [r["assembly_accession"] for r in sel]
        self.assertIn("GCF_anchor", accs)
        self.assertEqual(rep["anchors_found"], 1)

    def test_missing_anchor_is_reported_not_silent(self):
        """A missing anchor means our euk table is not a superset of Tiara's.
        Silence here would hide the single most important precondition."""
        _, rep = S.select_breadth(self._rows(), mini_tax(), {},
                                  anchors=["GCF_999999"])
        self.assertEqual(rep["anchors_found"], 0)
        self.assertEqual(rep["anchors_missing"], ["GCF_999999"])

    def test_anchor_version_suffix_is_ignored(self):
        rows = [row("GCA_000146045.2", 4932)]
        _, rep = S.select_breadth(rows, mini_tax(), {},
                                  anchors=["GCA_000146045.1"])
        self.assertEqual(rep["anchors_found"], 1)
        self.assertEqual(rep["anchors_missing"], [])

    def test_heterogeneous_genus_may_exceed_one(self):
        # The quota scales with the number of distinct SPECIES in the
        # genus, so these must be two species, not one species twice.
        rows = [row("A", 5833, species_taxid=5833,
                    genome_size=20_000_000, gc_percent=20),
                row("B", 5833, species_taxid=5855,
                    genome_size=90_000_000, gc_percent=55)]
        cfg = {"genus": {"max_per_genus": 4, "exponent": 1.0}}
        sel, rep = S.select_breadth(rows, mini_tax(), cfg)
        self.assertEqual(len(sel), 2)
        self.assertEqual(
            rep["clades"]["alveolata"]["multi_representative_genera"], 1)

    def test_deterministic(self):
        a, _ = S.select_breadth(self._rows(), mini_tax(), {})
        b, _ = S.select_breadth(self._rows(), mini_tax(), {})
        self.assertEqual([r["assembly_accession"] for r in a],
                         [r["assembly_accession"] for r in b])


class TestReadTables(unittest.TestCase):
    """NCBI splits assembly_summary by division; reading one file silently
    curates a fraction of the tree."""

    HEADER = ("#assembly_accession\ttaxid\torganism_name\t"
              "refseq_category\tassembly_level\tgenome_size")

    def _write(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# comment line", self.HEADER]
        lines += ["\t".join(r) for r in rows]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_concatenates_divisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._write(d / "assembly_summary.refseq.fungi.txt",
                        [["GCF_1", "4932", "S. cerevisiae", "na",
                          "Complete Genome", "12000000"]])
            self._write(d / "assembly_summary.refseq.plant.txt",
                        [["GCF_2", "3702", "A. thaliana", "na",
                          "Complete Genome", "135000000"]])
            rows = S.read_tables([str(d / "assembly_summary.refseq.*.txt")])
            self.assertEqual(len(rows), 2)
            self.assertEqual({r["assembly_accession"] for r in rows},
                             {"GCF_1", "GCF_2"})

    def test_header_comment_is_not_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "assembly_summary.refseq.fungi.txt"
            self._write(p, [["GCF_1", "4932", "S. cerevisiae", "na",
                             "Complete Genome", "12000000"]])
            rows = S.read_tables([str(p)])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["taxid"], "4932")

    def test_refseq_copy_wins_over_genbank_duplicate(self):
        """GCA/GCF pairs share an accession stem; the RefSeq row carries
        refseq_category, which is our strongest quality signal."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._write(d / "assembly_summary.genbank.fungi.txt",
                        [["GCF_1", "4932", "S. cerevisiae", "na",
                          "Scaffold", "12000000"]])
            self._write(d / "assembly_summary.refseq.fungi.txt",
                        [["GCF_1", "4932", "S. cerevisiae",
                          "reference genome", "Complete Genome", "12000000"]])
            rows = S.read_tables([str(d / "assembly_summary.genbank.fungi.txt"),
                                  str(d / "assembly_summary.refseq.fungi.txt")])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["refseq_category"], "reference genome")

    def test_missing_file_is_skipped_not_fatal(self):
        rows = S.read_tables(["/nonexistent/assembly_summary.refseq.fungi.txt"])
        self.assertEqual(rows, [])

    def test_source_file_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "assembly_summary.refseq.plant.txt"
            self._write(p, [["GCF_2", "3702", "A. thaliana", "na",
                             "Complete Genome", "135000000"]])
            rows = S.read_tables([str(p)])
            self.assertEqual(rows[0]["_source_file"],
                             "assembly_summary.refseq.plant.txt")


def _run():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite(
        loader.loadTestsFromTestCase(c) for c in (
            TestTaxonomy, TestQuality, TestGenusQuota, TestDepth, TestSelect,
            TestStage, TestShippedConfig, TestBreadth, TestReadTables))
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    total = res.testsRun
    bad = len(res.failures) + len(res.errors)
    print(f"\n{total - bad}/{total} passed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    _run()

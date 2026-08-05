#!/usr/bin/env python3
"""Regression tests for class assignment and corpus relabelling.

The bug these lock down: ``class_from_meta`` used substring matching on the
supergroup name, so ``Archaeplastida`` (land plants, green algae, red algae --
all eukaryotes) matched ``"plastid" in text`` and every plant genome was
written into plastids.fasta.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tiara2 import labels  # noqa: E402
from tiara2.labels import (CLASSES, GROUP_META_DEFAULT, class_from_group,  # noqa: E402
                           class_from_header, class_from_meta, parse_header)
from scripts import audit_corpus_labels as audit  # noqa: E402
from scripts import repair_corpus_labels as repair  # noqa: E402
from scripts import import_tiara_corpus as imp  # noqa: E402


def legacy_class_from_meta(label: str, supergroup: str):
    """The buggy pre-fix implementation, kept only to prove we changed it."""
    label_l = label.strip().lower()
    super_l = supergroup.strip().lower()
    text = f"{label_l} {super_l}"
    if "mitochond" in text or super_l == "organelle-mito":
        return "mitochondria"
    if "plastid" in text or "chloroplast" in text or super_l == "organelle-plastid":
        return "plastids"
    if label_l in {"euk", "eukaryota", "eukarya"}:
        return "eukarya"
    if label_l in {"prok", "prokaryote", "prokaryota"} and "bacter" in super_l:
        return "bacteria"
    if label_l in {"prok", "prokaryote", "prokaryota"} and "archaea" in super_l:
        return "archaea"
    return None


class TestArchaeplastidaRegression(unittest.TestCase):
    def test_legacy_implementation_really_was_broken(self):
        self.assertEqual(legacy_class_from_meta("euk", "Archaeplastida"), "plastids")

    def test_plants_are_eukarya(self):
        self.assertEqual(class_from_meta("euk", "Archaeplastida"), "eukarya")

    def test_kinetoplastida_is_eukarya(self):
        self.assertEqual(class_from_meta("euk", "Kinetoplastida"), "eukarya")

    def test_no_colliding_taxon_is_misrouted(self):
        for taxon in labels.COLLIDING_TAXA:
            self.assertEqual(class_from_meta("euk", taxon), "eukarya", taxon)

    def test_archae_substring_does_not_make_a_plant_archaeal(self):
        self.assertNotEqual(class_from_meta("euk", "Archaeplastida"), "archaea")

    def test_plant_group_maps_to_eukarya(self):
        self.assertEqual(class_from_group("plant"), "eukarya")

    def test_only_plant_group_changed(self):
        # Restricted to the pre-v2.2 groups on purpose: the virus groups are a
        # deliberate v2.2 addition, not a regression in the Archaeplastida fix.
        changed = [g for g, (l, s) in GROUP_META_DEFAULT.items()
                   if class_from_group(g) != "virus"
                   and legacy_class_from_meta(l, s) != class_from_meta(l, s)]
        self.assertEqual(changed, ["plant"])


class TestClassFromMeta(unittest.TestCase):
    def test_every_default_group_resolves(self):
        for group in GROUP_META_DEFAULT:
            self.assertIn(class_from_group(group), labels.FINE_CLASSES, group)

    def test_real_plastids(self):
        self.assertEqual(class_from_meta("organelle", "Organelle-Plastid"), "plastids")

    def test_real_mitochondria(self):
        self.assertEqual(class_from_meta("organelle", "Organelle-Mito"), "mitochondria")

    def test_bacteria_and_archaea(self):
        self.assertEqual(class_from_meta("prok", "Bacteria"), "bacteria")
        self.assertEqual(class_from_meta("prok", "Archaea"), "archaea")

    def test_legacy_archea_spelling(self):
        self.assertEqual(class_from_meta("prok", "Archea"), "archaea")

    def test_direct_labels_win_over_supergroup(self):
        self.assertEqual(class_from_meta("euk", ""), "eukarya")
        self.assertEqual(class_from_meta("plastid", ""), "plastids")

    def test_case_and_whitespace_insensitive(self):
        self.assertEqual(class_from_meta("  EUK ", " archaeplastida "), "eukarya")

    def test_unknown_returns_none(self):
        self.assertIsNone(class_from_meta("", ""))

    def test_unknown_group_returns_none(self):
        self.assertIsNone(class_from_group("not_a_real_group"))

    def test_self_check_passes(self):
        labels.self_check()


class TestParseHeader(unittest.TestCase):
    HEADER = ">GCA_000411095.1|25 sg=Archaeplastida label=euk epoch=train"

    def test_fields(self):
        meta = parse_header(self.HEADER)
        self.assertEqual(meta["frag_id"], "GCA_000411095.1|25")
        self.assertEqual(meta["sg"], "Archaeplastida")
        self.assertEqual(meta["label"], "euk")
        self.assertEqual(meta["epoch"], "train")

    def test_leading_gt_optional(self):
        self.assertEqual(parse_header(self.HEADER), parse_header(self.HEADER[1:]))

    def test_missing_fields_default_to_empty(self):
        meta = parse_header(">frag_only")
        self.assertEqual(meta["sg"], "")
        self.assertEqual(meta["label"], "")

    def test_class_from_header(self):
        self.assertEqual(class_from_header(self.HEADER), "eukarya")

    def test_accession_from_frag_id(self):
        self.assertEqual(labels.accession_from_frag_id("GCA_000411095.1|25"), "GCA_000411095.1")


def _write_corpus(root: Path) -> None:
    """Build a tiny corpus reproducing the real defect: plants in plastids.fasta."""
    for split in ("train", "validation", "test"):
        (root / split).mkdir(parents=True, exist_ok=True)
        records = {
            "eukarya": [("GCF_E1|0", "Opisthokonta-Fungi", "euk"),
                        ("GCF_E2|0", "Opisthokonta-Metazoa", "euk")],
            "plastids": [("GCA_P1|0", "Archaeplastida", "euk"),
                         ("GCA_P2|0", "Archaeplastida", "euk"),
                         ("GCA_P3|0", "Archaeplastida", "euk"),
                         ("NC_R1|0", "Organelle-Plastid", "organelle")],
            "mitochondria": [("NC_M1|0", "Organelle-Mito", "organelle")],
            "bacteria": [("GCF_B1|0", "Bacteria", "prok")],
            "archaea": [("GCF_A1|0", "Archaea", "prok")],
        }
        for class_name, rows in records.items():
            path = root / split / f"{class_name}.fasta"
            with path.open("w") as handle:
                for fid, sg, label in rows:
                    handle.write(f">{fid} sg={sg} label={label} epoch={split}\nACGTACGTAC\n")


class TestAuditAndRepair(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "train_ready"
        self.out = Path(self._tmp.name) / "train_ready_relabelled"
        _write_corpus(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_audit_flags_the_plants(self):
        report = audit.audit_corpus(self.root)
        self.assertFalse(report["clean"])
        self.assertEqual(report["mismatched"], 9)  # 3 plants x 3 splits

    def test_audit_attributes_them_to_eukarya(self):
        report = audit.audit_corpus(self.root)
        entry = next(f for f in report["files"] if f["file_class"] == "plastids"
                     and "/train/" in f["path"].replace("\\", "/"))
        self.assertEqual(entry["derived"]["eukarya"], 3)
        self.assertEqual(entry["derived"]["plastids"], 1)

    def test_audit_limit_samples(self):
        report = audit.audit_corpus(self.root, limit=1)
        for entry in report["files"]:
            self.assertLessEqual(entry["records"], 1)

    def test_audit_render_does_not_crash(self):
        self.assertIn("MISLABELLED", audit.render(audit.audit_corpus(self.root)))

    def test_repair_moves_plants_to_eukarya(self):
        report = repair.repair_corpus(self.root, self.out)
        train = next(s for s in report["splits"] if s["split"] == "train")
        self.assertEqual(train["before"]["plastids"], 4)
        self.assertEqual(train["after"]["plastids"], 1)
        self.assertEqual(train["before"]["eukarya"], 2)
        self.assertEqual(train["after"]["eukarya"], 5)
        self.assertEqual(train["moves"]["plastids -> eukarya"], 3)

    def test_repair_conserves_records(self):
        report = repair.repair_corpus(self.root, self.out)
        before = sum(sum(s["before"].values()) for s in report["splits"])
        after = sum(sum(s["after"].values()) for s in report["splits"])
        self.assertEqual(before, after)
        self.assertEqual(report["unknown"], 0)

    def test_repair_output_is_clean_on_reaudit(self):
        repair.repair_corpus(self.root, self.out)
        self.assertTrue(audit.audit_corpus(self.out)["clean"])

    def test_repair_is_idempotent(self):
        repair.repair_corpus(self.root, self.out)
        second = Path(self._tmp.name) / "again"
        report = repair.repair_corpus(self.out, second)
        self.assertEqual(report["relocated"], 0)

    def test_repair_preserves_sequences(self):
        repair.repair_corpus(self.root, self.out)
        text = (self.out / "train" / "eukarya.fasta").read_text()
        self.assertEqual(text.count("ACGTACGTAC"), 5)
        self.assertIn("sg=Archaeplastida label=euk", text)

    def test_repair_rebuilds_organelle_file(self):
        repair.repair_corpus(self.root, self.out)
        organelle = (self.out / "train" / "organelle.fasta").read_text()
        self.assertEqual(organelle.count(">"), 2)  # 1 real plastid + 1 mito

    def test_repair_writes_legacy_archea_symlink(self):
        repair.repair_corpus(self.root, self.out)
        self.assertTrue((self.out / "train" / "archea.fasta").exists())

    def test_repair_writes_report(self):
        repair.repair_corpus(self.root, self.out)
        self.assertTrue((self.out / "relabel_report.json").is_file())

    def test_dry_run_writes_no_fasta(self):
        report = repair.repair_corpus(self.root, self.out, dry_run=True)
        self.assertEqual(report["relocated"], 9)
        self.assertFalse((self.out / "train" / "eukarya.fasta").exists())

    def test_unknown_metadata_is_quarantined(self):
        with (self.root / "train" / "eukarya.fasta").open("a") as handle:
            handle.write(">GCA_V1|0 sg=Nonsense-Supergroup label=xyzzy epoch=train\nACGT\n")
        report = repair.repair_corpus(self.root, self.out)
        self.assertEqual(report["unknown"], 1)
        self.assertTrue((self.out / "train" / "unclassified.fasta").is_file())

    def test_repair_render_does_not_crash(self):
        self.assertIn("relabel", repair.render(repair.repair_corpus(self.root, self.out, dry_run=True)))


class TestOrganellePrecedence(unittest.TestCase):
    """`sg` is the compartment, `label` is the host domain.

    Organelle records in this corpus are tagged ``label=euk sg=Organelle-Mito``
    because their host IS a eukaryote.  A label-first rule routed all of them
    into eukarya and emptied both organelle classes.
    """

    def test_mito_hosted_by_eukaryote_stays_mito(self):
        self.assertEqual(class_from_meta("euk", "Organelle-Mito"), "mitochondria")

    def test_plastid_hosted_by_eukaryote_stays_plastid(self):
        self.assertEqual(class_from_meta("euk", "Organelle-Plastid"), "plastids")

    def test_header_form_from_the_real_corpus(self):
        self.assertEqual(
            class_from_header(">NC_001|3 sg=Organelle-Mito label=euk epoch=train"),
            "mitochondria")

    def test_organelle_prefix_tokens(self):
        for sg, expected in (("Organelle-Mitochondrion", "mitochondria"),
                             ("Organelle-mt", "mitochondria"),
                             ("Organelle-Chloroplast", "plastids"),
                             ("Organelle-Apicoplast", "plastids"),
                             ("Organelle-cp", "plastids")):
            self.assertEqual(class_from_meta("euk", sg), expected, sg)

    def test_unrecognised_organelle_is_quarantined_not_guessed(self):
        self.assertIsNone(class_from_meta("euk", "Organelle-Nucleomorph"))

    def test_compartment_check_cannot_catch_a_plant(self):
        self.assertIsNone(labels.compartment_from_supergroup("archaeplastida"))
        self.assertIsNone(labels.compartment_from_supergroup("kinetoplastida"))

    def test_plants_still_eukarya_under_new_precedence(self):
        self.assertEqual(class_from_meta("euk", "Archaeplastida"), "eukarya")

    def test_compartment_label_still_works_without_supergroup(self):
        self.assertEqual(class_from_meta("plastid", ""), "plastids")
        self.assertEqual(class_from_meta("mito", ""), "mitochondria")


def _write_corpus_euk_labelled_organelles(root):
    """Reproduces the real corpus: organelles carry label=euk."""
    for split in ("train",):
        (root / split).mkdir(parents=True, exist_ok=True)
        records = {
            "eukarya": [("GCF_E1|0", "Opisthokonta-Fungi", "euk")],
            "plastids": [("GCA_P1|0", "Archaeplastida", "euk"),
                         ("NC_R1|0", "Organelle-Plastid", "euk")],
            "mitochondria": [("NC_M1|0", "Organelle-Mito", "euk")],
            "bacteria": [("GCF_B1|0", "Bacteria", "prok")],
            "archaea": [("GCF_A1|0", "Archaea", "prok")],
        }
        for class_name, rows in records.items():
            with (root / split / f"{class_name}.fasta").open("w") as handle:
                for fid, sg, label in rows:
                    handle.write(f">{fid} sg={sg} label={label} epoch={split}\nACGTACGTAC\n")


class TestClassDropGuard(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "train_ready"
        self.out = Path(self._tmp.name) / "relabelled"
        _write_corpus_euk_labelled_organelles(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_organelles_survive_the_relabel(self):
        report = repair.repair_corpus(self.root, self.out, splits=("train",), dry_run=True)
        train = report["splits"][0]
        self.assertEqual(train["after"]["mitochondria"], 1)
        self.assertEqual(train["after"]["plastids"], 1)
        self.assertEqual(train["after"]["eukarya"], 2)

    def test_only_the_plant_moves(self):
        report = repair.repair_corpus(self.root, self.out, splits=("train",), dry_run=True)
        self.assertEqual(report["splits"][0]["moves"], {"plastids -> eukarya": 1})

    def test_no_class_is_emptied(self):
        report = repair.repair_corpus(self.root, self.out, splits=("train",), dry_run=True)
        self.assertEqual(report["emptied_classes"], [])

    def test_guard_reports_emptied_classes(self):
        # a corpus whose organelle metadata we deliberately break
        broken = Path(self._tmp.name) / "broken"
        (broken / "train").mkdir(parents=True)
        with (broken / "train" / "mitochondria.fasta").open("w") as handle:
            handle.write(">NC_M1|0 sg=Opisthokonta-Fungi label=euk epoch=train\nACGT\n")
        with (broken / "train" / "eukarya.fasta").open("w") as handle:
            handle.write(">GCF_E1|0 sg=Opisthokonta-Fungi label=euk epoch=train\nACGT\n")
        report = repair.repair_corpus(broken, self.out, splits=("train",), dry_run=True)
        self.assertEqual(report["emptied_classes"], ["mitochondria"])

    def test_guard_message_is_rendered(self):
        broken = Path(self._tmp.name) / "broken2"
        (broken / "train").mkdir(parents=True)
        with (broken / "train" / "plastids.fasta").open("w") as handle:
            handle.write(">GCA_P1|0 sg=Archaeplastida label=euk epoch=train\nACGT\n")
        text = repair.render(repair.repair_corpus(broken, self.out, splits=("train",), dry_run=True))
        self.assertIn("REFUSING", text)
        self.assertIn("--allow-class-drop", text)

    def test_cli_exits_nonzero_when_a_class_would_be_emptied(self):
        broken = Path(self._tmp.name) / "broken3"
        (broken / "train").mkdir(parents=True)
        with (broken / "train" / "plastids.fasta").open("w") as handle:
            handle.write(">GCA_P1|0 sg=Archaeplastida label=euk epoch=train\nACGT\n")
        code = repair.main(["--root", str(broken), "--out", str(self.out),
                            "--splits", "train", "--dry-run"])
        self.assertEqual(code, 3)

    def test_cli_override_allows_the_drop(self):
        broken = Path(self._tmp.name) / "broken4"
        (broken / "train").mkdir(parents=True)
        with (broken / "train" / "plastids.fasta").open("w") as handle:
            handle.write(">GCA_P1|0 sg=Archaeplastida label=euk epoch=train\nACGT\n")
        code = repair.main(["--root", str(broken), "--out", str(self.out),
                            "--splits", "train", "--dry-run", "--allow-class-drop"])
        self.assertEqual(code, 0)

class TestHierarchy(unittest.TestCase):
    """stage 1: archaea | bacteria | eukarya | other(organelle + virus)."""

    def test_stage1_vocabulary(self):
        self.assertEqual(labels.STAGE1_CLASSES,
                         ("archaea", "bacteria", "eukarya", "other"))

    def test_every_fine_class_has_a_stage1_home(self):
        for fine in labels.FINE_CLASSES:
            self.assertIn(labels.stage1_of(fine), labels.STAGE1_CLASSES, fine)

    def test_organelles_and_viruses_are_other(self):
        for fine in ("mitochondria", "plastids", "virus"):
            self.assertEqual(labels.stage1_of(fine), "other", fine)

    def test_domains_pass_through(self):
        for fine in ("archaea", "bacteria", "eukarya"):
            self.assertEqual(labels.stage1_of(fine), fine)

    def test_other_members_matches_the_map(self):
        derived = {c for c, s in labels.STAGE1_OF.items() if s == "other"}
        self.assertEqual(set(labels.OTHER_MEMBERS), derived)

    def test_stage1_of_none_is_none(self):
        self.assertIsNone(labels.stage1_of(None))

    def test_virus_is_a_fine_class(self):
        self.assertIn("virus", labels.FINE_CLASSES)
        self.assertNotIn("virus", labels.CLASSES)

    def test_virus_recognised_from_label_and_supergroup(self):
        self.assertEqual(class_from_meta("virus", ""), "virus")
        self.assertEqual(class_from_meta("", "Riboviria"), "virus")
        self.assertEqual(class_from_meta("viral", "Duplodnaviria"), "virus")

    def test_phage_is_not_filed_under_its_host(self):
        self.assertEqual(class_from_meta("virus", "Bacteria"), "virus")

    def test_stage1_from_header(self):
        self.assertEqual(
            labels.stage1_from_header(">x|0 sg=Organelle-Mito label=euk epoch=train"),
            "other")
        self.assertEqual(
            labels.stage1_from_header(">x|0 sg=Archaeplastida label=euk epoch=train"),
            "eukarya")

    def test_stage2_names_match_the_curate_clades(self):
        from tiara2 import taxonomy
        curate_clades = {name for name, _, _ in taxonomy.DEFAULT_CLADES}
        for clade in labels.STAGE2_EUK_CLASSES:
            if clade == "eukarya_other":
                continue
            self.assertIn(clade, curate_clades, clade)


class TestEukSubclass(unittest.TestCase):
    def test_fungi_from_supergroup(self):
        self.assertEqual(labels.euk_subclass("euk", "Opisthokonta-Fungi"),
                         ("fungi", "supergroup"))

    def test_metazoa_supergroup_is_ambiguous(self):
        clade, how = labels.euk_subclass("euk", "Opisthokonta-Metazoa")
        self.assertIsNone(clade)
        self.assertEqual(how, "ambiguous")

    def test_archaeplastida_is_ambiguous_not_wrong(self):
        clade, how = labels.euk_subclass("euk", "Archaeplastida")
        self.assertIsNone(clade)
        self.assertEqual(how, "ambiguous")

    def test_group_resolves_the_ambiguity(self):
        self.assertEqual(labels.euk_subclass("euk", "Archaeplastida", group="plant"),
                         ("land_plant", "group"))
        self.assertEqual(
            labels.euk_subclass("euk", "Opisthokonta-Metazoa", group="vertebrate_mammalian"),
            ("metazoa_vertebrate", "group"))

    def test_non_eukaryote_has_no_subclass(self):
        self.assertEqual(labels.euk_subclass("prok", "Bacteria"), (None, "not_eukaryotic"))
        self.assertEqual(labels.euk_subclass("euk", "Organelle-Mito"),
                         (None, "not_eukaryotic"))

    def test_taxid_wins_when_a_taxonomy_is_supplied(self):
        class FakeInfo:
            clade = "metazoa_vertebrate"

        class FakeTaxonomy:
            def info(self, taxid):
                return FakeInfo()

        self.assertEqual(
            labels.euk_subclass("euk", "Opisthokonta-Metazoa", group="invertebrate",
                                taxid=9606, taxonomy=FakeTaxonomy()),
            ("metazoa_vertebrate", "taxid"))


class TestImportTiaraCorpus(unittest.TestCase):
    def test_prokaryote_rows_split_by_domain(self):
        self.assertEqual(
            imp.classify_row("Prokaryotic chromosome", "Archaea", "")["fine_class"], "archaea")
        self.assertEqual(
            imp.classify_row("Prokaryotic chromosome", "Bacteria", "")["fine_class"], "bacteria")

    def test_organelle_rows(self):
        self.assertEqual(imp.classify_row("Mitochondrion", "Eukarya", "")["fine_class"],
                         "mitochondria")
        self.assertEqual(imp.classify_row("Plastid", "Eukarya", "")["fine_class"], "plastids")

    def test_organelles_land_in_stage1_other(self):
        for genome_type in ("Mitochondrion", "Plastid"):
            self.assertEqual(imp.classify_row(genome_type, "Eukarya", "")["stage1_class"],
                             "other")

    def test_nuclear_eukaryote_gets_a_clade(self):
        row = imp.classify_row("Eukarya nuclear", "Eukarya", "Fungi;Ascomycota")
        self.assertEqual(row["fine_class"], "eukarya")
        self.assertEqual(row["stage1_class"], "eukarya")
        self.assertEqual(row["stage2_clade"], "fungi")
        self.assertEqual(row["group"], "fungi")

    def test_most_specific_rank_wins(self):
        row = imp.classify_row("Eukarya nuclear", "Eukarya", "Eukaryota;Metazoa;Chordata")
        self.assertEqual(row["stage2_clade"], "metazoa_vertebrate")

    def test_unknown_eukaryote_falls_back_not_crashes(self):
        row = imp.classify_row("Eukarya nuclear", "Eukarya", "Something;Unheard Of")
        self.assertEqual(row["stage2_clade"], "eukarya_other")
        self.assertEqual(row["clade_source"], "fallback")

    def test_blank_row_is_unmapped(self):
        self.assertEqual(imp.classify_row("", "", "")["clade_source"], "unmapped")

    def test_accession_normalisation(self):
        self.assertEqual(imp.extract_accessions("GCA_000411095.1"), ["GCA_000411095.1"])
        self.assertEqual(imp.base_accession("GCA_000411095.1"), "GCA_000411095")

    def test_diff_marks_new_and_held_and_duplicates(self):
        rows = [
            {"accession": "GCA_1.1", "base_accession": "GCA_1"},
            {"accession": "GCA_1.1", "base_accession": "GCA_1"},
            {"accession": "GCA_2.1", "base_accession": "GCA_2"},
            {"accession": "GCA_3.2", "base_accession": "GCA_3"},
        ]
        exact = {"GCA_2.1": [{}]}
        base = {"GCA_2": [{}], "GCA_3": [{}]}
        out = imp.diff_against_index(rows, exact, base)
        self.assertEqual([r["status"] for r in out],
                         ["new", "duplicate_in_source", "already_held",
                          "already_held_other_version"])

    def test_anchors_are_the_nuclear_eukaryotes(self):
        self.assertEqual(imp.classify_row("Eukarya nuclear", "Eukarya", "Fungi")["fine_class"],
                         "eukarya")
        self.assertEqual(imp.GROUP_BY_CLASS["mitochondria"], "mitochondrion")

if __name__ == "__main__":
    unittest.main(verbosity=2)

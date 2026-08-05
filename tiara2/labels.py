"""Canonical class assignment for Tiara2 corpora.

Single source of truth for mapping ``(label, supergroup)`` metadata onto the
five training classes.

This logic previously lived inline in ``03_04_prepare_dataset_v2_evaluate.py``
and matched the supergroup name with substrings.  That silently routed every
Archaeplastida genome (land plants, green algae, red algae) into the
``plastids`` class, because::

    "plastid" in "archaeplastida"   ->  True
    "archae"  in "archaeplastida"   ->  True

Two rules are enforced here and covered by tests/test_labels.py:

1. ``label`` is authoritative and is examined first.
2. ``supergroup`` is only ever compared with ``==`` against a closed set.
   Never use ``in`` / substring matching on taxon names.
"""

from __future__ import annotations

# The five legacy on-disk classes.  Kept as-is so existing corpora, manifests
# and the published v1.x models keep resolving.
CLASSES = ("bacteria", "archaea", "eukarya", "mitochondria", "plastids")

# v2.2 adds virus as a sixth fine class.  It is not a new stage-1 output: it
# joins the organelles inside "other".
FINE_CLASSES = CLASSES + ("virus",)

FLAT_NAMES = {
    "virus": "virus_fr.fasta",
    "bacteria": "bacteria_fr.fasta",
    "archaea": "archaea_fr.fasta",
    "eukarya": "eukarya_fr.fasta",
    "mitochondria": "mitochondria_fr.fasta",
    "plastids": "plast_fr.fasta",
}

MITO_LABELS = frozenset({"mito", "mitochondrion", "mitochondria"})
PLASTID_LABELS = frozenset({"plast", "plastid", "plastids", "chloroplast"})
EUK_LABELS = frozenset({"euk", "euk_nuclear", "eukarya", "eukaryota", "eukaryote"})
PROK_LABELS = frozenset({"prok", "prokaryote", "prokaryota"})
VIRUS_LABELS = frozenset({"vir", "virus", "viral", "viruses", "virion"})
BACTERIA_LABELS = frozenset({"bac", "bacteria", "bacterium"})
ARCHAEA_LABELS = frozenset({"arc", "arch", "archaea", "archaeon", "archea"})

MITO_SUPER = frozenset({"organelle-mito", "organelle-mitochondrion", "mitochondrion", "mitochondria"})
PLASTID_SUPER = frozenset({"organelle-plastid", "organelle-chloroplast", "plastid", "plastids", "chloroplast"})
BACTERIA_SUPER = frozenset({"bacteria", "bacterium", "eubacteria"})
ARCHAEA_SUPER = frozenset({"archaea", "archaeon", "archea"})
VIRUS_SUPER = frozenset({"virus", "viruses", "viral", "riboviria", "duplodnaviria",
                         "monodnaviria", "varidnaviria", "adnaviria", "ribozyviria"})

# Taxon names that CONTAIN another class name as a substring.  Kept so the
# regression tests can assert none of them is ever misrouted again.
COLLIDING_TAXA = (
    "Archaeplastida",   # contains 'plastid' AND 'archae'; all eukaryotes
    "Kinetoplastida",   # contains 'plastid'; Leishmania / Trypanosoma
    "Apicomplexa",
    "Glaucocystophyta",
)

GROUP_META_DEFAULT = {
    "fungi": ("euk", "Opisthokonta-Fungi"),
    "protozoa": ("euk", "Protist(SAR/Excavata/Amoebozoa)"),
    "plant": ("euk", "Archaeplastida"),
    "invertebrate": ("euk", "Opisthokonta-Metazoa"),
    "vertebrate_other": ("euk", "Opisthokonta-Metazoa"),
    "vertebrate_mammalian": ("euk", "Host-Mammalia"),
    "bacteria": ("prok", "Bacteria"),
    "archaea": ("prok", "Archaea"),
    "mitochondrion": ("organelle", "Organelle-Mito"),
    "mitochondria": ("organelle", "Organelle-Mito"),
    "mito": ("organelle", "Organelle-Mito"),
    "plastid": ("organelle", "Organelle-Plastid"),
    "plastids": ("organelle", "Organelle-Plastid"),
    "chloroplast": ("organelle", "Organelle-Plastid"),
    "viral": ("virus", "Virus"),
    "virus": ("virus", "Virus"),
    "viruses": ("virus", "Virus"),
}


class LabelError(ValueError):
    """Raised when metadata cannot be mapped onto one of the five classes."""


ORGANELLE_PREFIX = "organelle-"
MITO_COMPARTMENTS = frozenset({"mito", "mt", "mitochondrion", "mitochondria", "mitochondrial"})
PLASTID_COMPARTMENTS = frozenset({"plast", "cp", "plastid", "plastids", "chloroplast", "apicoplast"})


def compartment_from_supergroup(super_l: str) -> str | None:
    """Return the organelle compartment encoded by a supergroup string.

    ``None`` means "not an organelle supergroup".  ``LabelError`` is raised for
    an ``organelle-*`` string we do not recognise, so a new compartment is
    surfaced instead of silently falling through to the host domain.

    Matching is exact against closed sets, or exact against the token after the
    ``organelle-`` prefix.  Taxon names are never substring-searched, so
    ``Archaeplastida`` and ``Kinetoplastida`` cannot reach this function's
    positive branches.
    """
    if super_l in MITO_SUPER:
        return "mitochondria"
    if super_l in PLASTID_SUPER:
        return "plastids"
    if super_l.startswith(ORGANELLE_PREFIX):
        rest = super_l[len(ORGANELLE_PREFIX):]
        if rest in MITO_COMPARTMENTS:
            return "mitochondria"
        if rest in PLASTID_COMPARTMENTS:
            return "plastids"
        raise LabelError(f"unrecognised organelle supergroup: {super_l!r}")
    return None


def class_from_meta(label: str, supergroup: str) -> str | None:
    """Map ``(label, supergroup)`` onto one of :data:`CLASSES`, or ``None``.

    Precedence, in order:

    1. **Compartment** (``sg``).  ``sg`` records which physical compartment the
       sequence came from; ``label`` records the host's domain.  An organelle
       record is routinely tagged ``label=euk sg=Organelle-Mito`` because its
       host *is* a eukaryote -- the two fields are not competing answers to the
       same question, and the compartment is the more specific one.  Checking
       ``label`` first would collapse every mitochondrion and plastid into
       ``eukarya`` and delete both organelle classes.
    2. Explicit organelle ``label`` values, for corpora that put the
       compartment in ``label`` and leave ``sg`` empty.
    3. Domain ``label`` values (``euk`` / ``bac`` / ``arc``).
    4. Generic ``prok`` label disambiguated by an exact supergroup.

    Supergroups are only ever compared with ``==`` (or an exact token after the
    ``organelle-`` prefix).  Never substring-match a taxon name here:
    ``"plastid" in "archaeplastida"`` is True and cost us a training run.
    """
    label_l = (label or "").strip().lower()
    super_l = (supergroup or "").strip().lower()

    # 1. compartment wins over host domain
    try:
        compartment = compartment_from_supergroup(super_l)
    except LabelError:
        return None
    if compartment is not None:
        return compartment

    # 2. viruses -- checked before the host domain for the same reason as
    #    compartments: a phage is not its host.
    if label_l in VIRUS_LABELS or super_l in VIRUS_SUPER:
        return "virus"

    # 3. compartment carried in the label instead
    if label_l in MITO_LABELS:
        return "mitochondria"
    if label_l in PLASTID_LABELS:
        return "plastids"

    # 4. host domain
    if label_l in EUK_LABELS:
        return "eukarya"
    if label_l in BACTERIA_LABELS:
        return "bacteria"
    if label_l in ARCHAEA_LABELS:
        return "archaea"

    # 5. generic prokaryote label + exact supergroup
    if label_l in PROK_LABELS and super_l in BACTERIA_SUPER:
        return "bacteria"
    if label_l in PROK_LABELS and super_l in ARCHAEA_SUPER:
        return "archaea"
    return None


def class_from_group(group: str, group_meta: dict | None = None) -> str | None:
    """Map a raw-directory group name (``plant``, ``fungi``, ...) to a class."""
    meta = group_meta if group_meta is not None else GROUP_META_DEFAULT
    entry = meta.get(group)
    if entry is None:
        return None
    return class_from_meta(entry[0], entry[1])


def parse_header(header: str) -> dict:
    """Parse ``fid sg=... label=... epoch=...`` into a dict."""
    text = header[1:] if header.startswith(">") else header
    parts = text.split()
    out: dict = {"frag_id": parts[0] if parts else ""}
    for token in parts[1:]:
        if "=" in token:
            key, _, value = token.partition("=")
            out[key] = value
    out.setdefault("sg", "")
    out.setdefault("label", "")
    out.setdefault("epoch", "")
    return out


def class_from_header(header: str) -> str | None:
    """Derive the class of a single FASTA record from its header metadata."""
    meta = parse_header(header)
    return class_from_meta(meta["label"], meta["sg"])


def accession_from_frag_id(frag_id: str) -> str:
    """``GCA_000411095.1|25`` -> ``GCA_000411095.1``."""
    return frag_id.split("|", 1)[0]


# ---------------------------------------------------------------------------
# Two-stage class hierarchy (v2.2)
# ---------------------------------------------------------------------------
# The goal of this classifier is to pull eukaryotes -- chiefly fungi -- out of
# a mixed assembly.  Splitting five flat classes in one shot spends most of the
# model's capacity on a boundary we do not care about (mitochondrion vs
# plastid) while leaving the one we do care about (fungus vs plant vs metazoan)
# entirely unmodelled.
#
#   stage 1:  archaea | bacteria | eukarya | other
#             "other" = mitochondria + plastids + virus.  All three are small,
#             high-copy, compositionally odd, and are all things we want OUT of
#             the eukaryotic bin.  Grouping them turns three rare classes into
#             one class with enough support to train on, and takes the
#             plastid/mitochondrion decision off the critical path.
#
#   stage 2:  the eukaryotic clades, applied only to reads stage 1 called
#             eukarya.
#
# Stage-2 names are exactly tiara2.taxonomy.DEFAULT_CLADES, so the clades that
# `curate` samples by genus are the clades the model predicts.  Do not let the
# two drift.

STAGE1_CLASSES = ("archaea", "bacteria", "eukarya", "other")
OTHER_MEMBERS = ("mitochondria", "plastids", "virus")

STAGE1_OF = {
    "archaea": "archaea",
    "bacteria": "bacteria",
    "eukarya": "eukarya",
    "mitochondria": "other",
    "plastids": "other",
    "virus": "other",
}

STAGE2_EUK_CLASSES = (
    "fungi",
    "land_plant",
    "algae",
    "metazoa_vertebrate",
    "metazoa_invertebrate",
    "alveolata",
    "stramenopiles",
    "other_protist",
    "eukarya_other",
)

STAGE_FILES = {1: STAGE1_CLASSES, 2: STAGE2_EUK_CLASSES}

# NCBI division / raw-directory name -> stage-2 clade.
SUBCLASS_BY_GROUP = {
    "fungi": "fungi",
    "plant": "land_plant",
    "protozoa": "other_protist",
    "invertebrate": "metazoa_invertebrate",
    "vertebrate_other": "metazoa_vertebrate",
    "vertebrate_mammalian": "metazoa_vertebrate",
}

# supergroup string -> stage-2 clade, for corpora that only carry `sg=`.
SUBCLASS_BY_SUPERGROUP = {
    "opisthokonta-fungi": "fungi",
    "host-mammalia": "metazoa_vertebrate",
    "opisthokonta-metazoa-vertebrate": "metazoa_vertebrate",
    "opisthokonta-metazoa-invertebrate": "metazoa_invertebrate",
    "archaeplastida-landplant": "land_plant",
    "archaeplastida-alga": "algae",
    "sar-alveolata": "alveolata",
    "sar-stramenopiles": "stramenopiles",
}

# Supergroups that genuinely cannot be resolved from the string alone.  The old
# chopper wrote "Opisthokonta-Metazoa" for BOTH the invertebrate and the
# vertebrate_other divisions, and "Archaeplastida" for land plants and algae
# alike, so these have to be resolved from the taxid instead.
AMBIGUOUS_SUPERGROUPS = {
    "opisthokonta-metazoa": ("metazoa_vertebrate", "metazoa_invertebrate"),
    "archaeplastida": ("land_plant", "algae"),
    "protist(sar/excavata/amoebozoa)": ("alveolata", "stramenopiles", "other_protist"),
}


def stage1_of(fine_class):
    """Map a fine class onto its stage-1 output."""
    if fine_class is None:
        return None
    return STAGE1_OF.get(fine_class)


def stage1_from_meta(label: str, supergroup: str):
    """Stage-1 class straight from record metadata."""
    return stage1_of(class_from_meta(label, supergroup))


def stage1_from_header(header: str):
    return stage1_of(class_from_header(header))


def euk_subclass(label: str, supergroup: str, *, group: str = "",
                 taxid=None, taxonomy=None):
    """Resolve the stage-2 clade of a eukaryotic record.

    Returns ``(clade, how)`` where ``how`` is one of ``taxid``, ``group``,
    ``supergroup``, ``fallback`` or ``ambiguous``.  Callers should record it:
    a corpus resolved mostly by ``supergroup`` is a corpus whose vertebrate and
    invertebrate reads are not actually separable.

    Resolution order:

    1. ``taxid`` via :mod:`tiara2.taxonomy` -- authoritative.
    2. the NCBI division / raw directory the genome came from.
    3. the ``sg=`` string, when it is unambiguous.
    """
    if class_from_meta(label, supergroup) != "eukarya":
        return None, "not_eukaryotic"

    if taxid and taxonomy is not None:
        try:
            info = taxonomy.info(int(taxid))
        except (TypeError, ValueError):
            info = None
        if info is not None and getattr(info, "clade", None):
            return info.clade, "taxid"

    group_l = (group or "").strip().lower()
    if group_l in SUBCLASS_BY_GROUP:
        return SUBCLASS_BY_GROUP[group_l], "group"

    super_l = (supergroup or "").strip().lower()
    if super_l in SUBCLASS_BY_SUPERGROUP:
        return SUBCLASS_BY_SUPERGROUP[super_l], "supergroup"
    if super_l in AMBIGUOUS_SUPERGROUPS:
        return None, "ambiguous"
    return "eukarya_other", "fallback"


def self_check() -> None:
    """Assert the historical failures cannot come back.  Cheap; safe to call."""
    # substring collision: plants are eukaryotes, not plastids
    assert class_from_meta("euk", "Archaeplastida") == "eukarya"
    assert class_from_meta("euk", "Kinetoplastida") == "eukarya"
    # precedence: an organelle hosted by a eukaryote stays an organelle
    assert class_from_meta("euk", "Organelle-Mito") == "mitochondria"
    assert class_from_meta("euk", "Organelle-Plastid") == "plastids"
    assert class_from_meta("organelle", "Organelle-Plastid") == "plastids"
    assert class_from_meta("organelle", "Organelle-Mito") == "mitochondria"
    assert class_from_meta("prok", "Bacteria") == "bacteria"
    assert class_from_meta("prok", "Archaea") == "archaea"
    # viruses are their own fine class and live under "other"
    assert class_from_meta("virus", "Riboviria") == "virus"
    assert stage1_of("virus") == "other"
    for group in GROUP_META_DEFAULT:
        assert class_from_group(group) in FINE_CLASSES, group
    # the hierarchy must be total and consistent
    assert set(STAGE1_OF) == set(FINE_CLASSES)
    assert set(STAGE1_OF.values()) == set(STAGE1_CLASSES)
    assert set(OTHER_MEMBERS) == {c for c, s in STAGE1_OF.items() if s == "other"}

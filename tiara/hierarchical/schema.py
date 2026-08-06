"""Versioned hierarchical label schema with one-primary-variable releases."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json

EUK = ("fungi", "land_plant", "algae", "metazoa_vertebrate", "metazoa_invertebrate", "alveolata", "stramenopiles", "other_protist")
PROK = ("bacteria", "archaea")
ORGANELLE = ("mitochondria", "plastid")
BASE_PRIORS = {"bacteria": .42438644, "euk_nuclear": .31667691, "archaea": .17336867, "plastid": .05765327, "mitochondria": .02791470}
EUK_WEIGHTS = {"fungi": .20, "metazoa_invertebrate": .15, "land_plant": .13, "alveolata": .12, "other_protist": .12, "stramenopiles": .10, "algae": .08, "metazoa_vertebrate": .10}

@dataclass(frozen=True)
class VersionProfile:
    version: str
    root: tuple[str, ...]
    virus_enabled: bool = False
    calibration_enabled: bool = False
    virus_ratio: float | None = None
    euk_completeness_enabled: bool = False
    branch_balancing_enabled: bool = False
    def to_dict(self): return asdict(self)

PROFILES = {
    "2.3.0": VersionProfile("2.3.0", ("euk_nuclear", "prok", "organelle")),
    "2.3.1": VersionProfile("2.3.1", ("euk_nuclear", "prok", "organelle"), euk_completeness_enabled=True),
    "2.3.2": VersionProfile("2.3.2", ("euk_nuclear", "prok", "organelle"), euk_completeness_enabled=True, branch_balancing_enabled=True),
    "2.4.0": VersionProfile("2.4.0", ("euk_nuclear", "prok", "organelle", "virus"), virus_enabled=True, virus_ratio=.10),
}

def profile(version="2.3.0", **overrides):
    if version not in PROFILES: raise ValueError(f"unsupported profile {version}; have {sorted(PROFILES)}")
    data = PROFILES[version].to_dict(); data.update({k: v for k, v in overrides.items() if v is not None})
    ratio = data["virus_ratio"]
    if ratio is not None and not 0 < float(ratio) < 1: raise ValueError("virus_ratio must be in (0,1)")
    if data["virus_enabled"] != ("virus" in tuple(data["root"])): raise ValueError("virus_enabled/root mismatch")
    return VersionProfile(**data)

def scaled_priors(virus_ratio=None):
    if virus_ratio is None: return dict(BASE_PRIORS)
    ratio = float(virus_ratio); out = {k: v * (1-ratio) for k, v in BASE_PRIORS.items()}; out["virus"] = ratio; return out

@dataclass(frozen=True)
class HierarchySchema:
    profile: VersionProfile
    euk: tuple[str, ...] = EUK
    prok: tuple[str, ...] = PROK
    organelle: tuple[str, ...] = ORGANELLE
    def classes(self, head): return self.profile.root if head == "root" else getattr(self, head)
    def index(self, head): return {name: i for i, name in enumerate(self.classes(head))}
    def to_dict(self): return {"profile": self.profile.to_dict(), "root": list(self.profile.root), "euk": list(self.euk), "prok": list(self.prok), "organelle": list(self.organelle)}
    @classmethod
    def from_dict(cls, data): return cls(VersionProfile(**data["profile"]), tuple(data["euk"]), tuple(data["prok"]), tuple(data["organelle"]))
    def dumps(self): return json.dumps(self.to_dict(), sort_keys=True)

def schema(version="2.3.0", **overrides): return HierarchySchema(profile(version, **overrides))

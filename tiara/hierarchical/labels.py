"""Strict metadata/header to hierarchical-label mapping."""
from __future__ import annotations
import re
from .schema import EUK
_TOKEN=re.compile(r"(?:^|[|;\s])(?:sg|supergroup|euk_group)=([^|;\s]+)",re.I)
ALIASES={
 "fungi":"fungi","opisthokonta-fungi":"fungi","land_plant":"land_plant","plant":"land_plant",
 "algae":"algae","metazoa_vertebrate":"metazoa_vertebrate","vertebrate":"metazoa_vertebrate","host-mammalia":"metazoa_vertebrate",
 "metazoa_invertebrate":"metazoa_invertebrate","invertebrate":"metazoa_invertebrate","opisthokonta-metazoa":"metazoa_invertebrate",
 "alveolata":"alveolata","stramenopiles":"stramenopiles","other_protist":"other_protist","protist":"other_protist","protist(sar/excavata/amoebozoa)":"other_protist",
}
def norm(x): return str(x or "").strip().lower().replace(" ","_")
def euk_group(header, explicit=None, policy="error"):
    raw=explicit
    if not raw:
        m=_TOKEN.search(header or ""); raw=m.group(1) if m else None
    key=norm(raw); value=ALIASES.get(key,key if key in EUK else None)
    if value: return value
    if policy=="other_protist": return "other_protist"
    if policy=="skip": return None
    raise ValueError(f"cannot resolve euk_group from header: {header[:160]}")
def labels_for(legacy_class, header="", explicit_euk=None, policy="error"):
    c=norm(legacy_class)
    if c=="eukarya": return {"root":"euk_nuclear","euk":euk_group(header,explicit_euk,policy),"prok":None,"organelle":None}
    if c in ("bacteria","archaea"): return {"root":"prok","euk":None,"prok":c,"organelle":None}
    if c in ("mitochondria","plastids","plastid"): return {"root":"organelle","euk":None,"prok":None,"organelle":"plastid" if c.startswith("plast") else "mitochondria"}
    if c=="virus": return {"root":"virus","euk":None,"prok":None,"organelle":None}
    raise ValueError(f"unknown legacy class {legacy_class!r}")

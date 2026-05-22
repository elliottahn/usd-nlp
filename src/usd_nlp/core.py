"""USD-NLP: Core data model — Scene/Stage/Layer/Prim hierarchy."""
from __future__ import annotations
import json, uuid, time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

class LayerPurpose(Enum):
    RAW_SOURCE = "L0"
    ENTITY_EXTRACTION = "L1"
    TERM_BASE = "L2"
    STRUCTURAL = "L3"
    REGISTER = "L4"
    TRANSLATION_STATE = "L5"
    REVIEW_STATE = "L6"

class PrimType(Enum):
    TERM = "T"
    NUMERAL = "N"
    VERB_PRED = "V"
    HONORIFIC = "H"
    CONTEXT = "C"

class LanguageCode(Enum):
    EN = "en"; KO = "ko"; JA = "ja"; FR = "fr"; ZH = "zh"

@dataclass
class Variant:
    name: str; content: str; language: LanguageCode = LanguageCode.EN
    metadata: Dict[str, Any] = field(default_factory=dict)
    def to_dict(self): return {"name":self.name,"content":self.content,"language":self.language.value,"metadata":self.metadata}
    @classmethod
    def from_dict(cls, d): return cls(name=d["name"],content=d["content"],language=LanguageCode(d.get("language","en")),metadata=d.get("metadata",{}))

@dataclass
class Prim:
    content: str; prim_type: PrimType = PrimType.TERM; language: LanguageCode = LanguageCode.EN
    prim_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    metadata: Dict[str,Any] = field(default_factory=dict)
    variants: Dict[str,Variant] = field(default_factory=dict)
    references: Set[str] = field(default_factory=set)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_offset: Optional[tuple] = None
    def add_variant(self, name, variant): self.variants[name] = variant
    def add_reference(self, target_id): self.references.add(target_id)
    def to_dict(self):
        return {"prim_id":self.prim_id,"content":self.content,"prim_type":self.prim_type.value,
                "language":self.language.value,"metadata":self.metadata,
                "variants":{k:v.to_dict() for k,v in self.variants.items()},
                "references":list(self.references),"created_at":self.created_at,"source_offset":self.source_offset}
    @classmethod
    def from_dict(cls, d):
        return cls(prim_id=d["prim_id"],content=d["content"],prim_type=PrimType(d["prim_type"]),
                   language=LanguageCode(d["language"]),metadata=d.get("metadata",{}),
                   variants={k:Variant.from_dict(v) for k,v in d.get("variants",{}).items()},
                   references=set(d.get("references",[])),created_at=d.get("created_at",""),
                   source_offset=tuple(d["source_offset"]) if d.get("source_offset") else None)

@dataclass
class Layer:
    purpose: LayerPurpose; language: LanguageCode = LanguageCode.EN
    layer_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    prims: List[Prim] = field(default_factory=list)
    overrides: Dict[str,Dict[str,Any]] = field(default_factory=dict)
    metadata: Dict[str,Any] = field(default_factory=dict)
    def add_prim(self, prim): self.prims.append(prim)
    def add_override(self, prim_id, new_content, reason=""):
        self.overrides[prim_id] = {"new_content":new_content,"reason":reason,"timestamp":datetime.now(timezone.utc).isoformat()}
    def resolve_prim(self, prim_id):
        if prim_id in self.overrides: return self.overrides[prim_id]["new_content"]
        for p in self.prims:
            if p.prim_id == prim_id: return p.content
        return None
    def get_prims_by_type(self, prim_type): return [p for p in self.prims if p.prim_type == prim_type]
    def to_dict(self):
        return {"layer_id":self.layer_id,"purpose":self.purpose.value,"language":self.language.value,
                "prims":[p.to_dict() for p in self.prims],"overrides":self.overrides,"metadata":self.metadata}
    @classmethod
    def from_dict(cls, d):
        layer = cls(layer_id=d["layer_id"],purpose=LayerPurpose(d["purpose"]),language=LanguageCode(d["language"]),metadata=d.get("metadata",{}))
        layer.prims = [Prim.from_dict(p) for p in d.get("prims",[])]
        layer.overrides = d.get("overrides",{})
        return layer

@dataclass
class Stage:
    name: str; stage_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    layers: List[Layer] = field(default_factory=list); metadata: Dict[str,Any] = field(default_factory=dict)
    def add_layer(self, layer): self.layers.append(layer)
    def get_layer(self, purpose, language=None):
        for l in self.layers:
            if l.purpose == purpose and (language is None or l.language == language): return l
        return None
    def compose(self):
        resolved = {}
        for layer in sorted(self.layers, key=lambda l: int(l.purpose.value[1:])):
            for prim in layer.prims:
                c = layer.resolve_prim(prim.prim_id)
                if c is not None: resolved[prim.prim_id] = c
        return resolved
    def to_dict(self): return {"stage_id":self.stage_id,"name":self.name,"layers":[l.to_dict() for l in self.layers],"metadata":self.metadata}
    @classmethod
    def from_dict(cls, d):
        stage = cls(stage_id=d["stage_id"],name=d["name"],metadata=d.get("metadata",{}))
        stage.layers = [Layer.from_dict(l) for l in d.get("layers",[])]
        return stage

@dataclass
class Scene:
    name: str; scene_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    stages: List[Stage] = field(default_factory=list); languages: Set[LanguageCode] = field(default_factory=set)
    metadata: Dict[str,Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    def add_stage(self, stage):
        self.stages.append(stage)
        for layer in stage.layers: self.languages.add(layer.language)
    def get_stage(self, name):
        for s in self.stages:
            if s.name == name: return s
        return None
    def find_prims(self, prim_type=None, language=None, content_contains=None):
        results = []
        for stage in self.stages:
            for layer in stage.layers:
                if language and layer.language != language: continue
                for prim in layer.prims:
                    if prim_type and prim.prim_type != prim_type: continue
                    if content_contains and content_contains not in prim.content: continue
                    results.append(prim)
        return results
    def summary(self):
        total = sum(len(l.prims) for s in self.stages for l in s.layers)
        tc = {}
        for s in self.stages:
            for l in s.layers:
                for p in l.prims: tc[p.prim_type.value] = tc.get(p.prim_type.value, 0) + 1
        return {"scene_id":self.scene_id,"name":self.name,"num_stages":len(self.stages),
                "num_layers":sum(len(s.layers) for s in self.stages),"num_prims":total,
                "languages":[l.value for l in self.languages],"prim_type_distribution":tc}
    def to_dict(self):
        return {"scene_id":self.scene_id,"name":self.name,"stages":[s.to_dict() for s in self.stages],
                "languages":[l.value for l in self.languages],"metadata":self.metadata,"created_at":self.created_at}
    def to_json(self, path=None, indent=2):
        data = self.to_dict()
        s = json.dumps(data, ensure_ascii=False, indent=indent)
        if path: Path(path).write_text(s, encoding="utf-8")
        return s
    @classmethod
    def from_dict(cls, d):
        scene = cls(scene_id=d["scene_id"],name=d["name"],metadata=d.get("metadata",{}),created_at=d.get("created_at",""))
        scene.stages = [Stage.from_dict(s) for s in d.get("stages",[])]
        scene.languages = {LanguageCode(l) for l in d.get("languages",[])}
        return scene
    @classmethod
    def from_json(cls, path): return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

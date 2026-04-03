"""USD-NLP Core — Scene/Stage/Layer/Prim hierarchy with USD composition semantics.

Maps Pixar USD concepts to NLP:
  Scene  → Project / Session (M&A deal, RSI session)
  Stage  → Document / Segment (SPA, speech block)
  Layer  → Language version / Processing state (L0-L6)
  Prim   → Linguistic unit (clause, term, utterance)
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class PrimType(Enum):
    """Five-type information taxonomy for CASI (T/N/V/H/C)."""
    T = "terminology"
    N = "numerical"
    V = "verb_pattern"
    H = "register_marker"
    C = "structural_cue"


class LanguageCode(Enum):
    """ISO 639-1 language codes."""
    JA = "ja"
    KO = "ko"
    EN = "en"
    FR = "fr"
    ZH = "zh"
    DE = "de"


class LayerPurpose(Enum):
    """Seven-layer processing stack (L0-L6) from Language Twin."""
    RAW_SOURCE = "L0"
    SEGMENTED = "L1"
    TERM_EXTRACTED = "L2"
    MT_DRAFT = "L3"
    POST_EDITED = "L4"
    REVIEWED = "L5"
    FINALISED = "L6"


@dataclass
class Variant:
    """Domain-specific rendering (e.g., JP-GAAP vs K-IFRS)."""
    name: str
    content: str
    language: LanguageCode = LanguageCode.EN
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "content": self.content,
            "language": self.language.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Variant:
        return cls(
            name=d["name"],
            content=d["content"],
            language=LanguageCode(d["language"]),
            metadata=d.get("metadata", {}),
        )


@dataclass
class Override:
    """Non-destructive amendment to a Prim."""
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    reason: str = ""
    author: str = ""

    def to_dict(self) -> dict:
        return {"content": self.content, "timestamp": self.timestamp,
                "reason": self.reason, "author": self.author}

    @classmethod
    def from_dict(cls, d: dict) -> Override:
        return cls(**d)


@dataclass
class Prim:
    """Linguistic unit — the atomic element of USD-NLP.

    Carries typed content with optional variants and override stack.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str = ""
    prim_type: PrimType = PrimType.T
    language: LanguageCode = LanguageCode.EN
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    variants: Dict[str, Variant] = field(default_factory=dict)
    _overrides: List[Override] = field(default_factory=list)

    @property
    def resolved_content(self) -> str:
        """Return override content if present, else original."""
        return self._overrides[-1].content if self._overrides else self.content

    def add_variant(self, key: str, variant: Variant) -> None:
        self.variants[key] = variant

    def get_variant(self, key: str) -> Optional[Variant]:
        return self.variants.get(key)

    def add_override(self, content: str, reason: str = "", author: str = "") -> Override:
        ov = Override(content=content, reason=reason, author=author)
        self._overrides.append(ov)
        return ov

    def remove_override(self) -> Optional[Override]:
        return self._overrides.pop() if self._overrides else None

    @property
    def override_count(self) -> int:
        return len(self._overrides)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "content": self.content,
            "prim_type": self.prim_type.value,
            "language": self.language.value,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "variants": {k: v.to_dict() for k, v in self.variants.items()},
            "overrides": [o.to_dict() for o in self._overrides],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Prim:
        p = cls(
            id=d["id"], content=d["content"],
            prim_type=PrimType(d["prim_type"]),
            language=LanguageCode(d["language"]),
            confidence=d.get("confidence", 1.0),
            metadata=d.get("metadata", {}),
        )
        for k, v in d.get("variants", {}).items():
            p.variants[k] = Variant.from_dict(v)
        for o in d.get("overrides", []):
            p._overrides.append(Override.from_dict(o))
        return p


@dataclass
class Layer:
    """Language version or processing state — composable, non-destructive."""
    name: str = ""
    purpose: LayerPurpose = LayerPurpose.RAW_SOURCE
    language: LanguageCode = LanguageCode.EN
    prims: Dict[str, Prim] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_prim(self, prim: Prim) -> Prim:
        self.prims[prim.id] = prim
        return prim

    def get_prim(self, prim_id: str) -> Optional[Prim]:
        return self.prims.get(prim_id)

    def remove_prim(self, prim_id: str) -> Optional[Prim]:
        return self.prims.pop(prim_id, None)

    def add_override(self, prim_id: str, content: str, reason: str = "") -> Optional[Override]:
        prim = self.prims.get(prim_id)
        if prim:
            return prim.add_override(content, reason=reason)
        return None

    def get_prims_by_type(self, prim_type: PrimType) -> List[Prim]:
        return [p for p in self.prims.values() if p.prim_type == prim_type]

    @property
    def prim_count(self) -> int:
        return len(self.prims)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "purpose": self.purpose.value,
            "language": self.language.value,
            "prims": {k: v.to_dict() for k, v in self.prims.items()},
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Layer:
        layer = cls(
            name=d["name"],
            purpose=LayerPurpose(d["purpose"]),
            language=LanguageCode(d["language"]),
            metadata=d.get("metadata", {}),
        )
        for k, v in d.get("prims", {}).items():
            layer.prims[k] = Prim.from_dict(v)
        return layer


@dataclass
class Stage:
    """Document or segment — contains composable layers."""
    name: str = ""
    layers: Dict[str, Layer] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_layer(self, layer: Layer) -> Layer:
        self.layers[layer.purpose.value] = layer
        return layer

    def get_layer(self, purpose: LayerPurpose) -> Optional[Layer]:
        return self.layers.get(purpose.value)

    def remove_layer(self, purpose: LayerPurpose) -> Optional[Layer]:
        return self.layers.pop(purpose.value, None)

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "layers": {k: v.to_dict() for k, v in self.layers.items()},
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Stage:
        stage = cls(name=d["name"], metadata=d.get("metadata", {}))
        for k, v in d.get("layers", {}).items():
            stage.layers[k] = Layer.from_dict(v)
        return stage


@dataclass
class Scene:
    """Project or session — the top-level container."""
    name: str = ""
    stages: Dict[str, Stage] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_stage(self, stage: Stage) -> Stage:
        self.stages[stage.name] = stage
        return stage

    def get_stage(self, name: str) -> Optional[Stage]:
        return self.stages.get(name)

    def remove_stage(self, name: str) -> Optional[Stage]:
        return self.stages.pop(name, None)

    @property
    def stage_count(self) -> int:
        return len(self.stages)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, path: str) -> Scene:
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Scene:
        scene = cls(name=d["name"], metadata=d.get("metadata", {}))
        for k, v in d.get("stages", {}).items():
            scene.stages[k] = Stage.from_dict(v)
        return scene

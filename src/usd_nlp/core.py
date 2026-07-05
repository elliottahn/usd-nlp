"""USD-NLP core data model.

The module maps Universal Scene Description concepts to multilingual NLP
workflows while remaining dependency-free:

    Scene -> project or session
    Stage -> document, segment, or utterance block
    Layer -> language or processing state
    Prim  -> atomic linguistic unit, token, span, term, or annotation node

The implementation is deliberately small so it can be embedded in research
prototypes and audited easily by reviewers.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional


class PrimType(Enum):
    """Primary routing categories used by USD-NLP."""
    TERM = "terminology"
    NUMERAL = "numeral"
    VERB_PRED = "verb_pattern"
    HONORIFIC = "register_marker"
    CONTEXT = "contextual_cue"

    # Backwards-compatible short aliases from the original taxonomy.
    T = "terminology"
    N = "numeral"
    V = "verb_pattern"
    H = "register_marker"
    C = "contextual_cue"


class LanguageCode(Enum):
    """Common ISO 639-1 language codes used in examples."""
    JA = "ja"
    KO = "ko"
    EN = "en"
    FR = "fr"
    ZH = "zh"
    DE = "de"
    ES = "es"
    UNK = "und"


class LayerPurpose(Enum):
    """Seven-layer processing stack used by LanguageTwin."""
    RAW_SOURCE = "L0"
    ENTITY_EXTRACTION = "L1"
    TERM_BASE = "L2"
    STRUCTURAL = "L3"
    REGISTER = "L4"
    TRANSLATION_STATE = "L5"
    REVIEW_STATE = "L6"

    # Backwards-compatible aliases from earlier releases.
    SEGMENTED = "L1"
    TERM_EXTRACTED = "L2"
    MT_DRAFT = "L3"
    POST_EDITED = "L4"
    REVIEWED = "L5"
    FINALISED = "L6"


@dataclass
class Variant:
    """Alternative rendering of a linguistic unit, e.g. a target-language term."""
    name: str
    content: str
    language: LanguageCode = LanguageCode.EN
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "content": self.content,
            "language": self.language.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Variant":
        return cls(
            name=data["name"],
            content=data["content"],
            language=LanguageCode(data.get("language", LanguageCode.EN.value)),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Override:
    """Non-destructive amendment to a Prim."""
    content: str
    reason: str = ""
    author: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "reason": self.reason,
            "author": self.author,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Override":
        return cls(
            content=data["content"],
            reason=data.get("reason", ""),
            author=data.get("author", ""),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )


@dataclass
class Reference:
    """Typed cross-prim or cross-document link.

    Dependency arcs imported from CoNLL-U are represented as references from
    a dependent token Prim to its head token Prim, with the relation label stored
    in metadata.
    """
    source_stage: str
    source_prim_id: str
    target_stage: str
    target_prim_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_stage": self.source_stage,
            "source_prim_id": self.source_prim_id,
            "target_stage": self.target_stage,
            "target_prim_id": self.target_prim_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Reference":
        return cls(
            source_stage=data["source_stage"],
            source_prim_id=data["source_prim_id"],
            target_stage=data["target_stage"],
            target_prim_id=data["target_prim_id"],
            metadata=data.get("metadata", {}),
        )


@dataclass(init=False)
class Prim:
    """Atomic linguistic unit.

    A Prim can represent a clause, term, token, span annotation, utterance, or
    dependency-linked annotation node depending on the layer that contains it.
    """
    prim_id: str
    content: str
    prim_type: PrimType
    language: LanguageCode
    confidence: float
    metadata: Dict[str, Any]
    variants: Dict[str, Variant]
    references: List[str]
    _overrides: List[Override]

    def __init__(
        self,
        content: str = "",
        prim_type: PrimType = PrimType.TERM,
        language: LanguageCode = LanguageCode.EN,
        prim_id: Optional[str] = None,
        id: Optional[str] = None,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        variants: Optional[Dict[str, Variant]] = None,
        references: Optional[Iterable[str]] = None,
    ) -> None:
        self.prim_id = prim_id or id or str(uuid.uuid4())[:8]
        self.content = content
        self.prim_type = prim_type
        self.language = language
        self.confidence = confidence
        self.metadata = dict(metadata or {})
        self.variants = dict(variants or {})
        self.references = list(references or [])
        self._overrides = []

    @property
    def id(self) -> str:
        """Alias retained for early prototype compatibility."""
        return self.prim_id

    @property
    def resolved_content(self) -> str:
        return self._overrides[-1].content if self._overrides else self.content

    @property
    def override_count(self) -> int:
        return len(self._overrides)

    def add_variant(self, key: str, variant: Variant) -> None:
        self.variants[key] = variant

    def get_variant(self, key: str) -> Optional[Variant]:
        return self.variants.get(key)

    def add_reference(self, prim_id: str) -> None:
        if prim_id not in self.references:
            self.references.append(prim_id)

    def add_override(self, content: str, reason: str = "", author: str = "") -> Override:
        override = Override(content=content, reason=reason, author=author)
        self._overrides.append(override)
        return override

    def remove_override(self) -> Optional[Override]:
        return self._overrides.pop() if self._overrides else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prim_id": self.prim_id,
            "content": self.content,
            "prim_type": self.prim_type.value,
            "language": self.language.value,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "variants": {k: v.to_dict() for k, v in self.variants.items()},
            "references": list(self.references),
            "overrides": [o.to_dict() for o in self._overrides],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Prim":
        prim = cls(
            content=data.get("content", ""),
            prim_type=PrimType(data.get("prim_type", PrimType.TERM.value)),
            language=LanguageCode(data.get("language", LanguageCode.EN.value)),
            prim_id=data.get("prim_id") or data.get("id"),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
            variants={k: Variant.from_dict(v) for k, v in data.get("variants", {}).items()},
            references=data.get("references", []),
        )
        for override in data.get("overrides", []):
            prim._overrides.append(Override.from_dict(override))
        return prim


@dataclass
class Layer:
    """Language version or processing state containing ordered Prims."""
    purpose: LayerPurpose = LayerPurpose.RAW_SOURCE
    language: LanguageCode = LanguageCode.EN
    name: str = ""
    prims: List[Prim] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_prim(self, prim: Prim) -> Prim:
        self.prims.append(prim)
        return prim

    def get_prim(self, prim_id: str) -> Optional[Prim]:
        return next((p for p in self.prims if p.prim_id == prim_id), None)

    def remove_prim(self, prim_id: str) -> Optional[Prim]:
        for idx, prim in enumerate(self.prims):
            if prim.prim_id == prim_id:
                return self.prims.pop(idx)
        return None

    def add_override(self, prim_id: str, content: str, reason: str = "", author: str = "") -> Optional[Override]:
        prim = self.get_prim(prim_id)
        if prim is None:
            return None
        return prim.add_override(content, reason=reason, author=author)

    def resolve_prim(self, prim_id: str) -> Optional[str]:
        prim = self.get_prim(prim_id)
        return prim.resolved_content if prim is not None else None

    def get_prims_by_type(self, prim_type: PrimType) -> List[Prim]:
        return [p for p in self.prims if p.prim_type == prim_type]

    @property
    def prim_count(self) -> int:
        return len(self.prims)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "purpose": self.purpose.value,
            "language": self.language.value,
            "name": self.name,
            "prims": [p.to_dict() for p in self.prims],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Layer":
        return cls(
            purpose=LayerPurpose(data.get("purpose", LayerPurpose.RAW_SOURCE.value)),
            language=LanguageCode(data.get("language", LanguageCode.EN.value)),
            name=data.get("name", ""),
            prims=[Prim.from_dict(p) for p in data.get("prims", [])],
            metadata=data.get("metadata", {}),
        )


@dataclass
class Stage:
    """Document, segment, or utterance block containing composable layers."""
    name: str = ""
    layers: List[Layer] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_layer(self, layer: Layer) -> Layer:
        self.layers.append(layer)
        return layer

    def get_layer(self, purpose: LayerPurpose) -> Optional[Layer]:
        for layer in reversed(self.layers):
            if layer.purpose == purpose:
                return layer
        return None

    def remove_layer(self, purpose: LayerPurpose) -> Optional[Layer]:
        for idx in range(len(self.layers) - 1, -1, -1):
            if self.layers[idx].purpose == purpose:
                return self.layers.pop(idx)
        return None

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    def compose(self) -> Dict[str, Prim]:
        resolved: Dict[str, Prim] = {}
        order = {purpose.value: idx for idx, purpose in enumerate(LayerPurpose)}
        for layer in sorted(self.layers, key=lambda x: order.get(x.purpose.value, 99)):
            for prim in layer.prims:
                resolved[prim.prim_id] = prim
        return resolved

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "layers": [l.to_dict() for l in self.layers],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Stage":
        return cls(
            name=data.get("name", ""),
            layers=[Layer.from_dict(l) for l in data.get("layers", [])],
            metadata=data.get("metadata", {}),
        )


@dataclass
class Scene:
    """Top-level project or interpreting session."""
    name: str = ""
    stages: List[Stage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    references: List[Reference] = field(default_factory=list)

    def add_stage(self, stage: Stage) -> Stage:
        self.stages.append(stage)
        return stage

    def get_stage(self, name: str) -> Optional[Stage]:
        return next((s for s in self.stages if s.name == name), None)

    def remove_stage(self, name: str) -> Optional[Stage]:
        for idx, stage in enumerate(self.stages):
            if stage.name == name:
                return self.stages.pop(idx)
        return None

    def add_reference(self, reference: Reference) -> Reference:
        self.references.append(reference)
        return reference

    def get_references_for_stage(self, stage_name: str) -> List[Reference]:
        return [r for r in self.references if r.source_stage == stage_name or r.target_stage == stage_name]

    def find_prims(
        self,
        prim_type: Optional[PrimType] = None,
        language: Optional[LanguageCode] = None,
        text: Optional[str] = None,
    ) -> List[Prim]:
        results: List[Prim] = []
        for stage in self.stages:
            for layer in stage.layers:
                for prim in layer.prims:
                    if prim_type is not None and prim.prim_type != prim_type:
                        continue
                    if language is not None and prim.language != language:
                        continue
                    if text is not None and text not in prim.content:
                        continue
                    results.append(prim)
        return results

    @property
    def stage_count(self) -> int:
        return len(self.stages)

    def summary(self) -> Dict[str, int]:
        n_layers = sum(len(stage.layers) for stage in self.stages)
        n_prims = sum(len(layer.prims) for stage in self.stages for layer in stage.layers)
        return {"num_stages": len(self.stages), "num_layers": n_layers, "num_prims": n_prims}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "stages": [s.to_dict() for s in self.stages],
            "metadata": self.metadata,
            "references": [r.to_dict() for r in self.references],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scene":
        return cls(
            name=data.get("name", ""),
            stages=[Stage.from_dict(s) for s in data.get("stages", [])],
            metadata=data.get("metadata", {}),
            references=[Reference.from_dict(r) for r in data.get("references", [])],
        )

    def to_json(self, path: Optional[str] = None) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(payload)
        return payload

    @classmethod
    def from_json(cls, path: str) -> "Scene":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

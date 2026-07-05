"""Ready-to-use LanguageTwin and InterpretationTwin architectures."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .core import LanguageCode, Layer, LayerPurpose, Prim, PrimType, Scene, Stage, Variant
from .state_layers import StateLayerStack


class DegradationLevel(Enum):
    NOMINAL = "nominal"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILURE = "failure"


@dataclass
class QoSConfig:
    min_confidence: float = 0.5
    degraded_latency_ms: int = 800
    critical_latency_ms: int = 2000
    failure_latency_ms: int = 4000

    def classify_latency(self, latency_ms: int) -> DegradationLevel:
        if latency_ms >= self.failure_latency_ms:
            return DegradationLevel.FAILURE
        if latency_ms >= self.critical_latency_ms:
            return DegradationLevel.CRITICAL
        if latency_ms >= self.degraded_latency_ms:
            return DegradationLevel.DEGRADED
        return DegradationLevel.NOMINAL


@dataclass
class Hint:
    content: str
    prim_type: PrimType = PrimType.TERM
    confidence: float = 1.0
    latency_ms: int = 0
    ttl_ms: int = 5000
    delivered: bool = True
    degradation: DegradationLevel = DegradationLevel.NOMINAL

    def to_dict(self) -> Dict[str, object]:
        return {
            "content": self.content,
            "prim_type": self.prim_type.value,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "ttl_ms": self.ttl_ms,
            "delivered": self.delivered,
            "degradation": self.degradation.value,
        }


@dataclass
class LanguageTwin:
    """Shared-state multilingual document translation helper."""
    source_lang: LanguageCode = LanguageCode.JA
    scene: Scene = field(default_factory=lambda: Scene(name="LanguageTwin"))
    term_base: Dict[str, Prim] = field(default_factory=dict)
    layer_stacks: Dict[str, StateLayerStack] = field(default_factory=dict)

    @property
    def document_count(self) -> int:
        return len(self.scene.stages)

    @property
    def term_count(self) -> int:
        return len(self.term_base)

    def ingest_document(self, name: str, text: str, segments: Optional[List[str]] = None) -> Stage:
        stage = Stage(name=name, metadata={"source_lang": self.source_lang.value})
        raw = Layer(name="raw", purpose=LayerPurpose.RAW_SOURCE, language=self.source_lang)
        raw.add_prim(Prim(content=text, prim_type=PrimType.CONTEXT, language=self.source_lang, prim_id=f"{name}:raw"))
        stage.add_layer(raw)

        if segments:
            segmented = Layer(name="segments", purpose=LayerPurpose.ENTITY_EXTRACTION, language=self.source_lang)
            for idx, segment in enumerate(segments, start=1):
                segmented.add_prim(Prim(content=segment, prim_type=PrimType.CONTEXT,
                                        language=self.source_lang, prim_id=f"{name}:seg{idx}"))
            stage.add_layer(segmented)

        self.scene.add_stage(stage)
        stack = StateLayerStack()
        for layer in stage.layers:
            stack.add_layer(layer)
        self.layer_stacks[name] = stack
        return stage

    def add_term(self, content: str, translations: Optional[Dict[str, str]] = None,
                 prim_type: PrimType = PrimType.TERM) -> Prim:
        prim = Prim(content=content, prim_type=prim_type, language=self.source_lang, prim_id=f"term:{content}")
        for lang, value in (translations or {}).items():
            prim.add_variant(lang, Variant(name=lang, content=value, language=LanguageCode(lang)))
        self.term_base[content] = prim
        return prim

    def update_term(self, content: str, lang: str, value: str) -> Variant:
        prim = self.term_base[content]
        variant = Variant(name=lang, content=value, language=LanguageCode(lang))
        prim.add_variant(lang, variant)
        return variant

    def get_term(self, content: str) -> Optional[Prim]:
        return self.term_base.get(content)

    def get_terms_by_type(self, prim_type: PrimType) -> List[Prim]:
        return [p for p in self.term_base.values() if p.prim_type == prim_type]

    def get_document_state(self, name: str) -> Optional[Dict[str, Prim]]:
        stack = self.layer_stacks.get(name)
        return stack.compose() if stack else None

    def get_layer_stack(self, name: str) -> Optional[StateLayerStack]:
        return self.layer_stacks.get(name)

    def to_dict(self) -> Dict[str, object]:
        return {
            "source_lang": self.source_lang.value,
            "scene": self.scene.to_dict(),
            "term_base": {k: v.to_dict() for k, v in self.term_base.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "LanguageTwin":
        twin = cls(source_lang=LanguageCode(data.get("source_lang", "ja")))
        twin.scene = Scene.from_dict(data["scene"])
        twin.term_base = {k: Prim.from_dict(v) for k, v in data.get("term_base", {}).items()}
        twin.layer_stacks = {}
        for stage in twin.scene.stages:
            stack = StateLayerStack()
            for layer in stage.layers:
                stack.add_layer(layer)
            twin.layer_stacks[stage.name] = stack
        return twin

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "LanguageTwin":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


@dataclass
class InterpretationTwin:
    """Real-time simultaneous interpreting assistance helper."""
    qos: QoSConfig = field(default_factory=QoSConfig)
    scene: Scene = field(default_factory=lambda: Scene(name="InterpretationTwin"))
    current_stage: Optional[Stage] = None
    hints: List[Hint] = field(default_factory=list)

    @property
    def hint_count(self) -> int:
        return len(self.hints)

    def start_session(self, name: str) -> Stage:
        self.current_stage = Stage(name=name)
        self.scene.add_stage(self.current_stage)
        return self.current_stage

    def add_hint(self, content: str, prim_type: PrimType = PrimType.TERM,
                 confidence: float = 1.0, latency_ms: int = 0, ttl_ms: int = 5000) -> Optional[Hint]:
        if confidence < self.qos.min_confidence:
            return None
        degradation = self.qos.classify_latency(latency_ms)
        hint = Hint(content=content, prim_type=prim_type, confidence=confidence,
                    latency_ms=latency_ms, ttl_ms=ttl_ms,
                    delivered=degradation != DegradationLevel.FAILURE,
                    degradation=degradation)
        self.hints.append(hint)
        if self.current_stage is not None:
            layer = self.current_stage.get_layer(LayerPurpose.RAW_SOURCE)
            if layer is None:
                layer = self.current_stage.add_layer(Layer(name="hints", purpose=LayerPurpose.RAW_SOURCE))
            layer.add_prim(Prim(content=content, prim_type=prim_type, confidence=confidence,
                                prim_id=f"hint:{len(self.hints)}"))
        return hint

    def end_session(self) -> Dict[str, object]:
        delivered = sum(1 for h in self.hints if h.delivered)
        failures = sum(1 for h in self.hints if h.degradation == DegradationLevel.FAILURE)
        return {"total_hints": len(self.hints), "delivered_hints": delivered, "failures": failures}

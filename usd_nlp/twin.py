"""USD-NLP Twin Architectures — LanguageTwin and InterpretationTwin.

LanguageTwin: T = (S, U, R, G) for shared-state document translation.
InterpretationTwin: Real-time SI assistance with QoS-aware hint delivery.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .core import (
    LanguageCode, Layer, LayerPurpose, Override, Prim,
    PrimType, Scene, Stage, Variant,
)
from .state_layers import StateLayerStack


# ============================================================
# Language Twin: T = (S, U, R, G)
# ============================================================

@dataclass
class LanguageTwin:
    """Shared-state architecture for cross-lingual document translation.

    Implements T = (S, U, R, G):
      S = Shared state (Scene + StateLayerStack)
      U = Update function (ingestion, term addition)
      R = Retrieval function (term lookup, context retrieval)
      G = Generation function (translation, post-editing)
    """
    scene: Scene = field(default_factory=Scene)
    source_lang: LanguageCode = LanguageCode.JA
    target_langs: List[LanguageCode] = field(
        default_factory=lambda: [LanguageCode.KO, LanguageCode.EN]
    )
    term_base: Dict[str, Prim] = field(default_factory=dict)
    _layer_stacks: Dict[str, StateLayerStack] = field(default_factory=dict)

    # --- S: State management ---

    def ingest_document(self, name: str, content: str,
                        segments: Optional[List[str]] = None) -> Stage:
        """Ingest a document into the scene (U function).

        Creates a Stage with L0 (raw) and optionally L1 (segmented) layers.
        """
        stage = Stage(name=name)
        # L0: raw source
        l0 = Layer(name=f"{name}_L0", purpose=LayerPurpose.RAW_SOURCE,
                    language=self.source_lang)
        raw_prim = Prim(content=content, language=self.source_lang,
                        prim_type=PrimType.C)
        l0.add_prim(raw_prim)
        stage.add_layer(l0)

        # L1: segmented (if segments provided)
        if segments:
            l1 = Layer(name=f"{name}_L1", purpose=LayerPurpose.SEGMENTED,
                       language=self.source_lang)
            for i, seg in enumerate(segments):
                seg_prim = Prim(id=f"seg_{i:03d}", content=seg,
                                language=self.source_lang, prim_type=PrimType.C)
                l1.add_prim(seg_prim)
            stage.add_layer(l1)

        self.scene.add_stage(stage)

        # Create layer stack for this document
        stack = StateLayerStack()
        for layer in stage.layers.values():
            stack.add_layer(layer)
        self._layer_stacks[name] = stack

        return stage

    # --- U: Update functions ---

    def add_term(self, term: str, translations: Optional[Dict[str, str]] = None,
                 prim_type: PrimType = PrimType.T) -> Prim:
        """Add a term to the shared term base."""
        prim = Prim(content=term, prim_type=prim_type,
                    language=self.source_lang)
        if translations:
            for lang, trans in translations.items():
                prim.add_variant(lang, Variant(
                    name=f"{term}_{lang}", content=trans,
                    language=LanguageCode(lang)
                ))
        self.term_base[term] = prim
        return prim

    def update_term(self, term: str, lang: str, new_content: str,
                    reason: str = "") -> Optional[Variant]:
        """Update a term's translation for a specific language."""
        prim = self.term_base.get(term)
        if prim is None:
            return None
        variant = Variant(name=f"{term}_{lang}_updated", content=new_content,
                          language=LanguageCode(lang))
        prim.add_variant(lang, variant)
        return variant

    # --- R: Retrieval functions ---

    def get_term(self, term: str) -> Optional[Prim]:
        """Look up a term in the shared term base."""
        return self.term_base.get(term)

    def get_terms_by_type(self, prim_type: PrimType) -> List[Prim]:
        """Retrieve all terms of a specific type."""
        return [p for p in self.term_base.values() if p.prim_type == prim_type]

    def get_document_state(self, doc_name: str) -> Optional[Dict[str, Prim]]:
        """Get composed state of a document (strongest-opinion-wins)."""
        stack = self._layer_stacks.get(doc_name)
        return stack.compose() if stack else None

    # --- G: Generation support ---

    def get_layer_stack(self, doc_name: str) -> Optional[StateLayerStack]:
        """Access the layer stack for a document."""
        return self._layer_stacks.get(doc_name)

    @property
    def document_count(self) -> int:
        return self.scene.stage_count

    @property
    def term_count(self) -> int:
        return len(self.term_base)

    def to_dict(self) -> dict:
        return {
            "scene": self.scene.to_dict(),
            "source_lang": self.source_lang.value,
            "target_langs": [l.value for l in self.target_langs],
            "term_base": {k: v.to_dict() for k, v in self.term_base.items()},
        }


# ============================================================
# Interpretation Twin: Real-time SI assistance
# ============================================================

class DegradationLevel(Enum):
    """QoS degradation levels for RSI sessions."""
    NOMINAL = "nominal"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILURE = "failure"


@dataclass
class QoSConfig:
    """Quality of Service thresholds for RSI hint delivery."""
    latency_nominal_ms: float = 500.0
    latency_degraded_ms: float = 1500.0
    latency_critical_ms: float = 3000.0
    hint_ttl_seconds: float = 10.0
    min_confidence: float = 0.6

    def classify_latency(self, latency_ms: float) -> DegradationLevel:
        """Classify a latency measurement into a degradation level."""
        if latency_ms <= self.latency_nominal_ms:
            return DegradationLevel.NOMINAL
        elif latency_ms <= self.latency_degraded_ms:
            return DegradationLevel.DEGRADED
        elif latency_ms <= self.latency_critical_ms:
            return DegradationLevel.CRITICAL
        else:
            return DegradationLevel.FAILURE


@dataclass
class Hint:
    """A typed hint delivered to the interpreter."""
    prim: Prim
    timestamp: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    degradation: DegradationLevel = DegradationLevel.NOMINAL
    delivered: bool = False
    expired: bool = False

    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    def to_dict(self) -> dict:
        return {
            "prim": self.prim.to_dict(),
            "timestamp": self.timestamp,
            "latency_ms": self.latency_ms,
            "degradation": self.degradation.value,
            "delivered": self.delivered,
            "expired": self.expired,
        }


@dataclass
class LatencyRecord:
    """Record of a single latency measurement."""
    timestamp: float
    latency_ms: float
    degradation: DegradationLevel
    prim_type: PrimType

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "latency_ms": self.latency_ms,
            "degradation": self.degradation.value,
            "prim_type": self.prim_type.value,
        }


@dataclass
class InterpretationTwin:
    """Real-time simultaneous interpreting assistance with QoS monitoring.

    Provides typed hint delivery, latency recording, confidence filtering,
    TTL-based hint expiry, and session analytics.
    """
    scene: Scene = field(default_factory=Scene)
    source_lang: LanguageCode = LanguageCode.JA
    target_lang: LanguageCode = LanguageCode.KO
    qos: QoSConfig = field(default_factory=QoSConfig)
    hints: List[Hint] = field(default_factory=list)
    latency_log: List[LatencyRecord] = field(default_factory=list)
    _session_start: float = field(default_factory=time.time)
    _active: bool = False

    def start_session(self, name: str = "session") -> Stage:
        """Start an RSI session."""
        self._session_start = time.time()
        self._active = True
        stage = Stage(name=name, metadata={"start": self._session_start})
        self.scene.add_stage(stage)
        return stage

    def end_session(self) -> Dict[str, Any]:
        """End the session and return analytics."""
        self._active = False
        return self.get_analytics()

    def add_hint(self, content: str, prim_type: PrimType,
                 confidence: float = 1.0,
                 latency_ms: float = 0.0) -> Optional[Hint]:
        """Add a typed hint with confidence filtering and QoS classification."""
        # Confidence filter
        if confidence < self.qos.min_confidence:
            return None

        prim = Prim(
            content=content,
            prim_type=prim_type,
            language=self.target_lang,
            confidence=confidence,
        )

        degradation = self.qos.classify_latency(latency_ms)
        hint = Hint(
            prim=prim,
            latency_ms=latency_ms,
            degradation=degradation,
            delivered=(degradation != DegradationLevel.FAILURE),
        )
        self.hints.append(hint)

        # Record latency
        self.latency_log.append(LatencyRecord(
            timestamp=time.time(),
            latency_ms=latency_ms,
            degradation=degradation,
            prim_type=prim_type,
        ))

        return hint

    def expire_hints(self) -> int:
        """Expire hints that have exceeded TTL. Returns count of expired."""
        now = time.time()
        count = 0
        for hint in self.hints:
            if not hint.expired and (now - hint.timestamp) > self.qos.hint_ttl_seconds:
                hint.expired = True
                count += 1
        return count

    def get_active_hints(self) -> List[Hint]:
        """Return non-expired, delivered hints."""
        return [h for h in self.hints if h.delivered and not h.expired]

    def get_hints_by_type(self, prim_type: PrimType) -> List[Hint]:
        """Filter hints by information type."""
        return [h for h in self.hints if h.prim.prim_type == prim_type]

    def get_analytics(self) -> Dict[str, Any]:
        """Compute session analytics."""
        if not self.latency_log:
            return {"total_hints": 0, "duration_s": 0}

        latencies = [r.latency_ms for r in self.latency_log]
        degradation_counts = {}
        type_counts = {}
        for r in self.latency_log:
            degradation_counts[r.degradation.value] = \
                degradation_counts.get(r.degradation.value, 0) + 1
            type_counts[r.prim_type.value] = \
                type_counts.get(r.prim_type.value, 0) + 1

        delivered = sum(1 for h in self.hints if h.delivered)
        expired = sum(1 for h in self.hints if h.expired)

        return {
            "total_hints": len(self.hints),
            "delivered": delivered,
            "expired": expired,
            "delivery_rate": delivered / len(self.hints) if self.hints else 0,
            "mean_latency_ms": sum(latencies) / len(latencies),
            "max_latency_ms": max(latencies),
            "min_latency_ms": min(latencies),
            "degradation_counts": degradation_counts,
            "type_counts": type_counts,
            "duration_s": time.time() - self._session_start,
        }

    @property
    def hint_count(self) -> int:
        return len(self.hints)

    def to_dict(self) -> dict:
        return {
            "scene": self.scene.to_dict(),
            "source_lang": self.source_lang.value,
            "target_lang": self.target_lang.value,
            "qos": vars(self.qos),
            "hints": [h.to_dict() for h in self.hints],
            "latency_log": [r.to_dict() for r in self.latency_log],
        }

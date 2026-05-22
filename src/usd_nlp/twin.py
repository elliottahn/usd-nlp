"""USD-NLP: Twin architectures.

Provides two ready-to-use instantiations of the USD-NLP core model:

- LanguageTwin: shared-state translation of multi-document projects,
  with user-supplied segmentation, multilingual term-base management,
  and update / retrieval / generation helper methods.

- InterpretationTwin: real-time simultaneous interpreting assistance
  with configurable network quality-of-service thresholds,
  confidence-filtered hint delivery, TTL-based hint expiry, and four
  graduated degradation levels.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .core import (
    Scene, Stage, Layer, Prim, Variant,
    LayerPurpose, LanguageCode, PrimType,
)
from .state_layers import StateLayerStack


# ============================================================
# LanguageTwin
# ============================================================

class LanguageTwin:
    """Shared-state translation twin for multi-document projects.

    Document ingestion uses *user-supplied* segmentation: the caller
    passes the list of segments. The twin then builds the L0 raw layer
    and, when segments are supplied, an L1 entity layer. Term-base
    management (L2) supports jurisdiction-specific variants.
    """

    def __init__(self, name: str = "LanguageTwin"):
        self.scene = Scene(name=name)
        self.term_base: Dict[str, Prim] = {}

    def ingest_document(self, doc_name: str, segments: List[str],
                        language: LanguageCode = LanguageCode.EN) -> Stage:
        """Ingest a document from user-supplied segments.

        `segments` is provided by the caller; the twin does not perform
        automatic sentence segmentation.
        """
        stage = Stage(name=doc_name)
        l0 = Layer(purpose=LayerPurpose.RAW_SOURCE, language=language)
        for seg in segments:
            l0.add_prim(Prim(content=seg, prim_type=PrimType.CONTEXT,
                             language=language))
        stage.add_layer(l0)
        self.scene.add_stage(stage)
        return stage

    def add_term(self, term_id: str, content: str,
                 language: LanguageCode = LanguageCode.EN,
                 variants: Optional[Dict[str, str]] = None) -> Prim:
        """Register a term in the shared term base (L2).

        `variants` maps a jurisdiction label to its rendering, e.g.
        {"JP-GAAP": "...", "K-IFRS": "..."}.
        """
        prim = Prim(content=content, prim_type=PrimType.TERM,
                    language=language, prim_id=term_id)
        if variants:
            for label, text in variants.items():
                prim.add_variant(label, Variant(name=label, content=text,
                                                language=language))
        self.term_base[term_id] = prim
        return prim

    # -- helper methods (not a pluggable callback framework) --------------
    def update(self, term_id: str, content: str) -> bool:
        """Update helper: replace a term's content. Returns success."""
        if term_id not in self.term_base:
            return False
        self.term_base[term_id].content = content
        return True

    def retrieve(self, term_id: str) -> Optional[Prim]:
        """Retrieval helper: fetch a term by id."""
        return self.term_base.get(term_id)

    def generate(self, term_id: str, jurisdiction: str) -> Optional[str]:
        """Generation helper: render a term for a given jurisdiction."""
        prim = self.term_base.get(term_id)
        if prim is None:
            return None
        if jurisdiction in prim.variants:
            return prim.variants[jurisdiction].content
        return prim.content

    def summary(self) -> Dict[str, Any]:
        return {"scene": self.scene.summary(),
                "term_base_size": len(self.term_base)}


# ============================================================
# InterpretationTwin
# ============================================================

class DegradationLevel(Enum):
    NOMINAL = "nominal"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILURE = "failure"


@dataclass
class QoSThresholds:
    """Network quality-of-service thresholds in milliseconds."""
    degraded_ms: float = 150.0
    critical_ms: float = 250.0
    failure_ms: float = 500.0

    def classify(self, latency_ms: float) -> DegradationLevel:
        if latency_ms >= self.failure_ms:
            return DegradationLevel.FAILURE
        if latency_ms >= self.critical_ms:
            return DegradationLevel.CRITICAL
        if latency_ms >= self.degraded_ms:
            return DegradationLevel.DEGRADED
        return DegradationLevel.NOMINAL


@dataclass
class Hint:
    """A typed interpreting hint with confidence and TTL."""
    content: str
    prim_type: PrimType
    confidence: float
    ttl_seconds: float
    created_at: float

    def is_expired(self, now: float) -> bool:
        return (now - self.created_at) > self.ttl_seconds


class InterpretationTwin:
    """Real-time simultaneous interpreting assistance twin.

    Monitors network latency, classifies it into four graduated
    degradation levels, and delivers typed, confidence-filtered hints
    with time-to-live expiry.
    """

    def __init__(self, name: str = "InterpretationTwin",
                 thresholds: Optional[QoSThresholds] = None,
                 min_confidence: float = 0.6):
        self.name = name
        self.thresholds = thresholds or QoSThresholds()
        self.min_confidence = min_confidence
        self.latency_log: List[float] = []
        self.hints: List[Hint] = []

    def record_latency(self, latency_ms: float) -> DegradationLevel:
        """Record a latency sample and return its degradation level."""
        self.latency_log.append(latency_ms)
        return self.thresholds.classify(latency_ms)

    def generate_hint(self, content: str, prim_type: PrimType,
                      confidence: float, ttl_seconds: float = 8.0,
                      now: Optional[float] = None) -> Optional[Hint]:
        """Create a hint if confidence clears the minimum threshold.

        Returns the Hint, or None if the confidence is too low.
        """
        if confidence < self.min_confidence:
            return None
        ts = now if now is not None else datetime.now(
            timezone.utc).timestamp()
        hint = Hint(content=content, prim_type=prim_type,
                    confidence=confidence, ttl_seconds=ttl_seconds,
                    created_at=ts)
        self.hints.append(hint)
        return hint

    def active_hints(self, now: Optional[float] = None) -> List[Hint]:
        """Return hints that have not yet expired."""
        ts = now if now is not None else datetime.now(
            timezone.utc).timestamp()
        return [h for h in self.hints if not h.is_expired(ts)]

    def session_analytics(self) -> Dict[str, Any]:
        """Summarise the session's latency profile and hint delivery."""
        if not self.latency_log:
            return {"samples": 0}
        n = len(self.latency_log)
        counts = {lvl.value: 0 for lvl in DegradationLevel}
        for lat in self.latency_log:
            counts[self.thresholds.classify(lat).value] += 1
        return {
            "samples": n,
            "mean_latency_ms": round(sum(self.latency_log) / n, 1),
            "max_latency_ms": round(max(self.latency_log), 1),
            "degradation_distribution": counts,
            "hints_generated": len(self.hints),
        }

"""USD-NLP: Seven-layer processing stack (L0-L6).

Implements the StateLayerStack described in the manuscript: a seven-layer
processing model with non-destructive composition, inter-layer diffing,
purpose-specific validation, and an append-only audit log.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .core import Layer, LayerPurpose, LanguageCode, Prim


# Canonical L0-L6 purpose ordering
LAYER_ORDER = [
    LayerPurpose.RAW_SOURCE,          # L0
    LayerPurpose.ENTITY_EXTRACTION,   # L1
    LayerPurpose.TERM_BASE,           # L2
    LayerPurpose.STRUCTURAL,          # L3
    LayerPurpose.REGISTER,            # L4
    LayerPurpose.TRANSLATION_STATE,   # L5
    LayerPurpose.REVIEW_STATE,        # L6
]

LAYER_DESCRIPTIONS = {
    LayerPurpose.RAW_SOURCE:        "Original documents as deposited",
    LayerPurpose.ENTITY_EXTRACTION: "Company names, amounts, dates",
    LayerPurpose.TERM_BASE:         "Multilingual term mappings + variants",
    LayerPurpose.STRUCTURAL:        "Clause numbering, cross-reference graph",
    LayerPurpose.REGISTER:          "Required register level per language",
    LayerPurpose.TRANSLATION_STATE: "Which clauses translated, by whom",
    LayerPurpose.REVIEW_STATE:      "Reviewed by counsel, board-approved",
}


@dataclass
class AuditEntry:
    """A single append-only audit log record."""
    action: str
    layer: str
    detail: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, str]:
        return {"action": self.action, "layer": self.layer,
                "detail": self.detail, "timestamp": self.timestamp}


class StateLayerStack:
    """A seven-layer (L0-L6) non-destructive processing stack.

    Layers are composed with strongest-opinion-wins semantics: a higher
    layer (closer to L6) overrides a lower layer for the same prim id.
    All mutations are recorded in an append-only audit log.
    """

    def __init__(self, language: LanguageCode = LanguageCode.EN):
        self.language = language
        self._layers: Dict[LayerPurpose, Layer] = {}
        self.audit_log: List[AuditEntry] = []

    # -- layer management -------------------------------------------------
    def set_layer(self, layer: Layer) -> None:
        """Insert or replace a layer at its purpose slot."""
        self._layers[layer.purpose] = layer
        self.audit_log.append(AuditEntry(
            action="set_layer", layer=layer.purpose.value,
            detail=f"{len(layer.prims)} prims"))

    def get_layer(self, purpose: LayerPurpose) -> Optional[Layer]:
        return self._layers.get(purpose)

    def has_layer(self, purpose: LayerPurpose) -> bool:
        return purpose in self._layers

    # -- composition ------------------------------------------------------
    def compose(self) -> Dict[str, str]:
        """Resolve all layers L0->L6 with strongest-opinion-wins semantics.

        Returns a mapping of prim id to resolved content.
        """
        resolved: Dict[str, str] = {}
        for purpose in LAYER_ORDER:
            layer = self._layers.get(purpose)
            if layer is None:
                continue
            for prim in layer.prims:
                content = layer.resolve_prim(prim.prim_id)
                if content is not None:
                    resolved[prim.prim_id] = content
        return resolved

    def diff(self, lower: LayerPurpose,
             upper: LayerPurpose) -> Dict[str, Dict[str, Any]]:
        """Compute per-prim differences between two layers.

        Returns a mapping of prim id to {"lower": ..., "upper": ...}
        for prims whose resolved content differs between the layers.
        """
        low = self._layers.get(lower)
        up = self._layers.get(upper)
        if low is None or up is None:
            return {}
        diffs: Dict[str, Dict[str, Any]] = {}
        low_ids = {p.prim_id for p in low.prims}
        up_ids = {p.prim_id for p in up.prims}
        for pid in low_ids | up_ids:
            lc = low.resolve_prim(pid) if pid in low_ids else None
            uc = up.resolve_prim(pid) if pid in up_ids else None
            if lc != uc:
                diffs[pid] = {"lower": lc, "upper": uc}
        return diffs

    # -- validation -------------------------------------------------------
    def validate_all(self) -> Dict[str, List[str]]:
        """Run purpose-specific validation rules on every layer.

        Returns a mapping of layer purpose value to a list of issues.
        An empty list means the layer passed validation.
        """
        issues: Dict[str, List[str]] = {}
        for purpose, layer in self._layers.items():
            layer_issues: List[str] = []
            # Every prim must carry a non-empty id and content.
            for prim in layer.prims:
                if not prim.prim_id:
                    layer_issues.append("prim with empty id")
                if not prim.content:
                    layer_issues.append(f"prim {prim.prim_id} has empty content")
            # L2 (term base) prims should declare at least one variant
            # or be explicitly marked monolingual.
            if purpose == LayerPurpose.TERM_BASE:
                for prim in layer.prims:
                    if not prim.variants and not prim.metadata.get("monolingual"):
                        layer_issues.append(
                            f"term {prim.prim_id} has no variant")
            issues[purpose.value] = layer_issues
        self.audit_log.append(AuditEntry(
            action="validate_all", layer="*",
            detail=f"{sum(len(v) for v in issues.values())} issues"))
        return issues

    # -- serialisation ----------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        return {
            "language": self.language.value,
            "layers_present": sorted(p.value for p in self._layers),
            "total_prims": sum(len(l.prims) for l in self._layers.values()),
            "audit_entries": len(self.audit_log),
        }

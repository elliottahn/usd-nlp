"""USD-NLP State Layer Stack — Seven-layer processing model (L0-L6).

Implements the Language Twin's shared-state architecture where each
document passes through progressive processing layers with
strongest-opinion-wins composition semantics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .core import Layer, LayerPurpose, Prim, PrimType


# Layer validation rules
LAYER_RULES = {
    LayerPurpose.RAW_SOURCE: lambda p: len(p.content) > 0,
    LayerPurpose.SEGMENTED: lambda p: len(p.content) > 0,
    LayerPurpose.TERM_EXTRACTED: lambda p: p.prim_type == PrimType.T or len(p.variants) > 0 or len(p.content) > 0,
    LayerPurpose.MT_DRAFT: lambda p: len(p.content) > 0,
    LayerPurpose.POST_EDITED: lambda p: len(p.content) > 0,
    LayerPurpose.REVIEWED: lambda p: "reviewer" in p.metadata or len(p.content) > 0,
    LayerPurpose.FINALISED: lambda p: p.confidence >= 0.9,
}


@dataclass
class AuditEntry:
    """Append-only audit log entry."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    action: str = ""
    layer: str = ""
    prim_id: str = ""
    detail: str = ""

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class StateLayerStack:
    """Seven-layer processing stack with composition and validation.

    Layers are ordered L0 (raw) through L6 (finalised).
    Composition resolves each prim by taking the highest-layer version
    (strongest-opinion-wins).
    """
    layers: Dict[str, Layer] = field(default_factory=dict)
    audit_log: List[AuditEntry] = field(default_factory=list)

    def add_layer(self, layer: Layer) -> Layer:
        """Add or replace a layer in the stack."""
        self.layers[layer.purpose.value] = layer
        self._log("add_layer", layer.purpose.value)
        return layer

    def get_layer(self, purpose: LayerPurpose) -> Optional[Layer]:
        return self.layers.get(purpose.value)

    def remove_layer(self, purpose: LayerPurpose) -> Optional[Layer]:
        removed = self.layers.pop(purpose.value, None)
        if removed:
            self._log("remove_layer", purpose.value)
        return removed

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    def compose(self) -> Dict[str, Prim]:
        """Resolve all layers: strongest-opinion-wins (highest layer number).

        Returns a dict of prim_id → resolved Prim.
        """
        resolved: Dict[str, Prim] = {}
        # Process in layer order (L0 first, L6 last → last wins)
        for purpose in LayerPurpose:
            layer = self.layers.get(purpose.value)
            if layer is None:
                continue
            for prim_id, prim in layer.prims.items():
                resolved[prim_id] = prim
        return resolved

    def diff(self, purpose_a: LayerPurpose, purpose_b: LayerPurpose) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
        """Compute differences between two layers.

        Returns dict of prim_id → (content_a, content_b) for differing prims.
        """
        layer_a = self.layers.get(purpose_a.value)
        layer_b = self.layers.get(purpose_b.value)
        if layer_a is None or layer_b is None:
            return {}

        all_ids = set(layer_a.prims.keys()) | set(layer_b.prims.keys())
        diffs: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        for pid in all_ids:
            pa = layer_a.prims.get(pid)
            pb = layer_b.prims.get(pid)
            ca = pa.resolved_content if pa else None
            cb = pb.resolved_content if pb else None
            if ca != cb:
                diffs[pid] = (ca, cb)
        return diffs

    def validate(self, purpose: LayerPurpose) -> List[str]:
        """Validate a specific layer against its purpose rules.

        Returns list of error messages (empty = valid).
        """
        layer = self.layers.get(purpose.value)
        if layer is None:
            return [f"Layer {purpose.value} not found"]
        errors = []
        rule = LAYER_RULES.get(purpose)
        if rule:
            for pid, prim in layer.prims.items():
                if not rule(prim):
                    errors.append(f"{purpose.value}/{pid}: validation failed")
        return errors

    def validate_all(self) -> Dict[str, List[str]]:
        """Validate all layers. Returns dict of layer → errors."""
        return {
            purpose.value: self.validate(purpose)
            for purpose in LayerPurpose
            if purpose.value in self.layers
        }

    def get_processing_status(self) -> Dict[str, int]:
        """Return prim counts per layer."""
        return {
            purpose.value: layer.prim_count
            for purpose, layer in (
                (LayerPurpose(k), v) for k, v in self.layers.items()
            )
        }

    def _log(self, action: str, layer: str, prim_id: str = "", detail: str = "") -> None:
        self.audit_log.append(AuditEntry(
            action=action, layer=layer, prim_id=prim_id, detail=detail
        ))

    def to_dict(self) -> dict:
        return {
            "layers": {k: v.to_dict() for k, v in self.layers.items()},
            "audit_log": [e.to_dict() for e in self.audit_log],
        }

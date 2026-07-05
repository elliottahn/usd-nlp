"""Seven-layer USD-NLP state stack."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from .core import Layer, LayerPurpose, Prim


@dataclass
class AuditEntry:
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    action: str = ""
    layer: str = ""
    prim_id: str = ""
    detail: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {
            "timestamp": self.timestamp,
            "action": self.action,
            "layer": self.layer,
            "prim_id": self.prim_id,
            "detail": self.detail,
        }


@dataclass
class StateLayerStack:
    """L0-L6 composition stack with an append-only audit log."""
    layers: Dict[str, Layer] = field(default_factory=dict)
    audit_log: List[AuditEntry] = field(default_factory=list)

    def add_layer(self, layer: Layer) -> Layer:
        self.layers[layer.purpose.value] = layer
        self._log("add_layer", layer.purpose.value)
        return layer

    def get_layer(self, purpose: LayerPurpose) -> Optional[Layer]:
        return self.layers.get(purpose.value)

    def remove_layer(self, purpose: LayerPurpose) -> Optional[Layer]:
        removed = self.layers.pop(purpose.value, None)
        if removed is not None:
            self._log("remove_layer", purpose.value)
        return removed

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    def compose(self) -> Dict[str, Prim]:
        resolved: Dict[str, Prim] = {}
        for purpose in LayerPurpose:
            layer = self.layers.get(purpose.value)
            if layer is None:
                continue
            for prim in layer.prims:
                resolved[prim.prim_id] = prim
        return resolved

    def diff(self, purpose_a: LayerPurpose, purpose_b: LayerPurpose) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
        layer_a = self.layers.get(purpose_a.value)
        layer_b = self.layers.get(purpose_b.value)
        if layer_a is None or layer_b is None:
            return {}
        ids = {p.prim_id for p in layer_a.prims} | {p.prim_id for p in layer_b.prims}
        diffs: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        for prim_id in ids:
            pa = layer_a.get_prim(prim_id)
            pb = layer_b.get_prim(prim_id)
            ca = pa.resolved_content if pa else None
            cb = pb.resolved_content if pb else None
            if ca != cb:
                diffs[prim_id] = (ca, cb)
        return diffs

    def validate(self, purpose: LayerPurpose) -> List[str]:
        layer = self.layers.get(purpose.value)
        if layer is None:
            return [f"Layer {purpose.value} not found"]
        errors: List[str] = []
        for prim in layer.prims:
            if not prim.content:
                errors.append(f"{purpose.value}/{prim.prim_id}: empty content")
            if purpose == LayerPurpose.REVIEW_STATE and prim.confidence < 0.9:
                errors.append(f"{purpose.value}/{prim.prim_id}: review confidence below 0.9")
            if purpose == LayerPurpose.TRANSLATION_STATE and "translator" not in prim.metadata:
                # Warning-level rule: recorded as validation message, not fatal.
                errors.append(f"{purpose.value}/{prim.prim_id}: translator metadata missing")
        return errors

    def validate_all(self) -> Dict[str, List[str]]:
        return {purpose.value: self.validate(purpose) for purpose in LayerPurpose if purpose.value in self.layers}

    def get_processing_status(self) -> Dict[str, int]:
        return {key: layer.prim_count for key, layer in self.layers.items()}

    def _log(self, action: str, layer: str, prim_id: str = "", detail: str = "") -> None:
        self.audit_log.append(AuditEntry(action=action, layer=layer, prim_id=prim_id, detail=detail))

    def to_dict(self) -> Dict[str, object]:
        return {
            "layers": {k: v.to_dict() for k, v in self.layers.items()},
            "audit_log": [entry.to_dict() for entry in self.audit_log],
        }

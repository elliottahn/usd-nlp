"""USD-NLP: Universal Scene Description for Multilingual NLP Pipelines."""
__version__ = "0.1.2"

from .core import (
    LanguageCode,
    Layer,
    LayerPurpose,
    Override,
    Prim,
    PrimType,
    Reference,
    Scene,
    Stage,
    Variant,
)
from .state_layers import AuditEntry, StateLayerStack
from .twin import DegradationLevel, Hint, InterpretationTwin, LanguageTwin, QoSConfig
from .adapters import add_named_entity_span, conllu_to_scene, from_conllu, stanza_doc_to_scene

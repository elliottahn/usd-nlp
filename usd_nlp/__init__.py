"""USD-NLP: Universal Scene Description for Multilingual NLP Pipelines."""
__version__ = "0.1.0"
from .core import (Scene, Stage, Layer, Prim, Variant, Override,
                   PrimType, LanguageCode, LayerPurpose)
from .state_layers import StateLayerStack, AuditEntry
from .twin import (LanguageTwin, InterpretationTwin,
                   QoSConfig, DegradationLevel, Hint)

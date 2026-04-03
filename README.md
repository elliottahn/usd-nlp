# USD-NLP: Universal Scene Description for Multilingual NLP Pipelines

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org)
[![Tests](https://img.shields.io/badge/tests-27%2F27-green.svg)](tests/)

A Python toolkit that adapts Pixar's [Universal Scene Description (USD)](https://openusd.org) pipeline concepts to multilingual natural language processing workflows.

## Key Features

- **USD-to-NLP mapping**: Scene→Project, Stage→Document, Layer→Language version, Prim→Linguistic unit
- **Non-destructive composition**: Overrides never mutate originals; full audit trail
- **Seven-layer processing stack** (L0-L6): Raw → Segmented → Terms → MT Draft → Post-edited → Reviewed → Finalised
- **Type-first design**: Every unit carries a T/N/V/H/C type annotation
- **Two Twin architectures**: LanguageTwin (document translation) + InterpretationTwin (real-time SI)
- **Zero dependencies**: Pure Python standard library only
- **27 unit tests**: Full coverage of core, state layers, and twin modules

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from usd_nlp import *

# Create an M&A deal room
scene = Scene(name="Project Sakura")
twin = LanguageTwin(scene=scene, source_lang=LanguageCode.JA)

# Ingest document with segmentation
twin.ingest_document("SPA_v3", "Stock Purchase Agreement...",
    segments=["Clause 1: Definitions", "Clause 2: Purchase Price"])

# Add trilingual term with jurisdiction variants
prim = twin.add_term("企業価値",
    translations={"en": "Enterprise Value", "ko": "기업가치"})
prim.add_variant("JP-GAAP", Variant(name="JP-GAAP",
    content="企業価値 (JP基準)", language=LanguageCode.JA))

# Non-destructive amendment
stage = scene.get_stage("SPA_v3")
l0 = stage.get_layer(LayerPurpose.RAW_SOURCE)
l0.add_override(list(l0.prims.keys())[0], "Amended clause text",
    reason="SPA v3.1 amendment")

# Serialize
scene.to_json("project_sakura.json")
```

## Examples

- `examples/ma_deal_room.py` — Cross-border M&A with multilingual terms
- `examples/rsi_qos_session.py` — RSI session with QoS monitoring

## Running Tests

```bash
python -m pytest tests/ -v
```

## Architecture

```
Scene (Project)
├── Stage (Document)
│   ├── Layer L0 (Raw Source)
│   │   ├── Prim [T] "表明保証" → "Representations and Warranties"
│   │   ├── Prim [N] "3,500億円"
│   │   └── Prim [H] "ご確認いただけますでしょうか"
│   ├── Layer L1 (Segmented)
│   ├── ...
│   └── Layer L6 (Finalised)
└── Stage (Document 2)
    └── ...
```

## Citation

If you use USD-NLP in your research, please cite:

```bibtex
@article{ahn2025usd,
  title={A Study on Integrating Digital Twin Pipelines Using USD and AI Simulation},
  author={Ahn, Seok-Hyun},
  journal={Machine Science},
  year={2025},
  doi={10.61413/QMBH8242}
}
```

## License

MIT License. See [LICENSE](LICENSE).

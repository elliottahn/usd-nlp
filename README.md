# USD-NLP

**Universal Scene Description for Multilingual NLP Pipelines**

USD-NLP adapts Pixar's Universal Scene Description (USD) composition
model to multilingual natural language processing. It maps USD's
Scene / Stage / Layer / Prim abstractions to a Project / Document /
Language-version / Linguistic-unit hierarchy, preserving USD's
non-destructive layer composition, cross-reference arcs, and override
semantics.

The library is pure Python with **zero external dependencies** and is
released under the MIT licence.

## Why USD-NLP?

Existing NLP tools (spaCy, Stanza, NLTK) process documents as flat
token sequences. They have no built-in concept of:

- **Hierarchical structure** — project → document → section → clause → term
- **Non-destructive versioning** — amendments overlay originals without destroying them
- **Parallel multi-language state** — the same unit in JA / KO / EN as composable layers
- **Typed information routing** — terminology, numerals, verb patterns,
  register markers, and contextual cues each routed differently

Translation memories (Trados, MemoQ) handle segment-level versioning
but lack composable document hierarchies. JSON-LD and XML provide
nesting but not composition semantics. USD-NLP fills this gap.

## Installation

```bash
git clone https://github.com/elliottahn/usd-nlp.git
cd usd-nlp
# No dependencies to install — pure Python standard library
```

Requires Python >= 3.9.

## Repository Layout

```
usd-nlp/
├── src/
│   └── usd_nlp/
│       ├── core.py          # Scene / Stage / Layer / Prim / Variant
│       ├── state_layers.py  # Seven-layer L0-L6 processing stack
│       └── twin.py          # LanguageTwin + InterpretationTwin
├── tests/                   # Unit test suite
├── examples/
│   ├── ma_deal_room.py      # Cross-border M&A example
│   └── rsi_qos_session.py   # Remote interpreting QoS example
├── benchmarks/
│   └── run_benchmarks.py    # Performance benchmark harness
├── README.md
└── LICENSE.txt
```

## Architecture

USD-NLP maps USD concepts to NLP as follows:

| USD       | NLP                  | Example                       |
|-----------|----------------------|-------------------------------|
| Scene     | Project / Session    | M&A deal, RSI session         |
| Stage     | Document / Segment   | SPA, speech block             |
| Layer     | Language / State     | L0 (raw) → L6 (reviewed)      |
| Prim      | Linguistic unit      | Clause, term, utterance       |
| Variant   | Domain rendering     | JP-GAAP vs. K-IFRS            |
| Reference | Cross-doc linkage    | Same term across documents    |
| Override  | Version amendment    | SPA v3 amends clause 4.2      |

### Seven-Layer Processing Stack (L0–L6)

| Layer | Purpose              | Contents                              |
|-------|----------------------|---------------------------------------|
| L0    | Raw source           | Original documents as deposited       |
| L1    | Entity extraction    | Company names, amounts, dates         |
| L2    | Term base            | Multilingual term mappings + variants |
| L3    | Structural           | Clause numbering, cross-reference graph |
| L4    | Register metadata    | Required register level per language |
| L5    | Translation state    | Which clauses translated, by whom     |
| L6    | Review state         | Reviewed by counsel, board-approved   |

## Quick Start

```python
from usd_nlp.core import (
    Scene, Stage, Layer, Prim,
    LayerPurpose, LanguageCode, PrimType,
)

# Create a deal (Scene) with a document (Stage)
scene = Scene(name="Project Sakura")
stage = Stage(name="SPA_v3")

# L0: source document in Japanese
l0 = Layer(purpose=LayerPurpose.RAW_SOURCE, language=LanguageCode.JA)
for clause in ["Clause 1: Definitions",
                "Clause 2: Representations",
                "Clause 3: Indemnification"]:
    l0.add_prim(Prim(content=clause, prim_type=PrimType.TERM,
                     language=LanguageCode.JA))

# Non-destructive amendment (SPA v3)
target_id = l0.prims[2].prim_id
l0.add_override(
    target_id,
    "Clause 3: Indemnification (amended: cap at 20% of deal value)",
    reason="SPA v3 amendment",
)

# Original preserved; override applied on resolution
print(l0.prims[2].content)          # -> "Clause 3: Indemnification"
print(l0.resolve_prim(target_id))   # -> "Clause 3: Indemnification (amended: ...)"

stage.add_layer(l0)
scene.add_stage(stage)

# Serialise (round-trip verified)
scene.to_json("project_sakura.json")
scene2 = Scene.from_json("project_sakura.json")
assert scene2.summary()["num_prims"] == 3
```

## Information Taxonomy

Each `Prim` carries a type annotation from a five-category taxonomy.
Each category is routed differently by downstream systems:

- **Terminology** — domain-specific terms requiring consistent rendition
- **Numerals** — numbers, dates, quantities requiring format conversion
- **Verb patterns** — predicates requiring structural transformation
- **Register markers** — politeness and formality signals
- **Contextual cues** — discourse structure and topic transitions

For example, numeral-type prims require near-immediate delivery in
interpreting because numbers are hard to retain in working memory,
while terminology-type prims tolerate higher latency because semantic
context aids recall.

## Twin Architectures

- **LanguageTwin** — shared-state translation of multi-document
  projects, with multilingual term base management and
  jurisdiction-specific variants.
- **InterpretationTwin** — real-time simultaneous interpreting
  assistance with configurable network quality-of-service thresholds,
  confidence-filtered hint delivery, TTL-based hint expiry, and
  four graduated degradation levels (nominal, degraded, critical,
  failure).

## Testing

```bash
python -m unittest discover tests/
```

## Benchmarks

```bash
python benchmarks/run_benchmarks.py
```

The harness reports the median of repeated runs for scene
construction, layer composition, JSON serialisation, override
application, prim search, and memory footprint. Absolute numbers
are hardware-dependent; see the manuscript for the reference
measurement environment.

## Citing

If you use USD-NLP in academic work, please cite the accompanying
SoftwareX paper (under review).

## License

MIT — see [LICENSE.txt](LICENSE.txt).

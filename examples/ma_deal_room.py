"""Cross-border M&A example with non-destructive amendment."""
from usd_nlp import LanguageCode, Layer, LayerPurpose, Prim, PrimType, Scene, Stage

scene = Scene(name="Project Sakura")
stage = Stage(name="SPA_v3")
l0 = Layer(purpose=LayerPurpose.RAW_SOURCE, language=LanguageCode.JA)

for clause in ["Clause 1: Definitions", "Clause 2: Representations", "Clause 3: Indemnification"]:
    l0.add_prim(Prim(content=clause, prim_type=PrimType.TERM, language=LanguageCode.JA))

target_id = l0.prims[2].prim_id
l0.add_override(target_id, "Clause 3: Indemnification (amended: cap at 20% of deal value)", reason="SPA v3 amendment")

stage.add_layer(l0)
scene.add_stage(stage)
print(l0.prims[2].content)
print(l0.resolve_prim(target_id))
print(scene.summary())

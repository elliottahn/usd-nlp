#!/usr/bin/env python3
"""Example: cross-border M&A deal room with non-destructive versioning.

Demonstrates the USD-NLP core model for a Japanese-Korean-English
share purchase agreement workflow.

Run:  python examples/ma_deal_room.py
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from usd_nlp.core import (
    Scene, Stage, Layer, Prim, Variant,
    LayerPurpose, LanguageCode, PrimType,
)


def main():
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

    print("Original clause :", l0.prims[2].content)
    print("Resolved clause :", l0.resolve_prim(target_id))

    stage.add_layer(l0)

    # L2: term base with jurisdiction-specific variants
    l2 = Layer(purpose=LayerPurpose.TERM_BASE, language=LanguageCode.EN)
    term = Prim(content="indemnification", prim_type=PrimType.TERM,
                language=LanguageCode.EN, prim_id="term_indem")
    term.add_variant("JP-GAAP", Variant(name="JP-GAAP",
                                        content="補償 (JP-GAAP basis)",
                                        language=LanguageCode.JA))
    term.add_variant("K-IFRS", Variant(name="K-IFRS",
                                       content="보상 (K-IFRS basis)",
                                       language=LanguageCode.KO))
    l2.add_prim(term)
    stage.add_layer(l2)
    scene.add_stage(stage)

    # Serialise (round-trip verified)
    out_path = "project_sakura.json"
    scene.to_json(out_path)
    scene2 = Scene.from_json(out_path)
    assert scene2.summary()["num_prims"] == 4

    print("\nScene summary:")
    for key, value in scene.summary().items():
        print(f"  {key}: {value}")

    os.remove(out_path)
    print("\n[OK] M&A deal room example completed.")


if __name__ == "__main__":
    main()

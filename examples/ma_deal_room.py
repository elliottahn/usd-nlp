"""Example: Cross-border M&A Deal Room with USD-NLP.

Demonstrates LanguageTwin with multilingual term management,
non-destructive versioning, and jurisdiction-specific variants.
"""
from usd_nlp import *

def main():
    # Create deal (Scene) and Language Twin
    scene = Scene(name="Project Sakura")
    twin = LanguageTwin(
        scene=scene,
        source_lang=LanguageCode.JA,
        target_langs=[LanguageCode.KO, LanguageCode.EN]
    )

    # Ingest SPA document
    twin.ingest_document("SPA_v3",
        "株式譲渡契約書 第3版",
        segments=[
            "第1条 定義",
            "第2条 譲渡価格: 3,500億円",
            "第3条 表明保証",
        ]
    )

    # Add trilingual terms with jurisdiction variants
    ev = twin.add_term("企業価値",
        translations={"en": "Enterprise Value", "ko": "기업가치"})
    ev.add_variant("JP-GAAP", Variant(
        name="JP-GAAP", content="企業価値 (日本基準)",
        language=LanguageCode.JA))
    ev.add_variant("K-IFRS", Variant(
        name="K-IFRS", content="기업가치 (K-IFRS)",
        language=LanguageCode.KO))

    rw = twin.add_term("表明保証",
        translations={"en": "Representations and Warranties",
                       "ko": "진술 및 보증"})

    # Non-destructive amendment
    stage = scene.get_stage("SPA_v3")
    l1 = stage.get_layer(LayerPurpose.SEGMENTED)
    l1.add_override("seg_001", "第2条 譲渡価格: 3,800億円 (改定)",
                    reason="Price revision per board resolution 2025-03-15")

    # Check state
    stack = twin.get_layer_stack("SPA_v3")
    resolved = stack.compose()
    print(f"Documents: {twin.document_count}")
    print(f"Terms: {twin.term_count}")
    print(f"Resolved prims: {len(resolved)}")
    for pid, prim in resolved.items():
        print(f"  {pid}: {prim.resolved_content[:50]}")

    # Serialize
    print(f"\nSerialized: {len(json.dumps(scene.to_dict()))} bytes")
    print("Done.")

if __name__ == "__main__":
    import json
    main()

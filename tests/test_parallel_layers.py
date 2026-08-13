"""Tests for parallel multi-language layers (manuscript capability #3)
and sub-Prim containment via typed part-of References.

These tests demonstrate that:
1. A Stage can hold JA/KO/EN layers of the same purpose simultaneously,
   each individually retrievable and preserved through a JSON round trip.
2. Containment below the Prim level (section > clause > term) is
   expressible with the existing ``references`` mechanism.
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from usd_nlp.core import (
    LanguageCode,
    Layer,
    LayerPurpose,
    Prim,
    PrimType,
    Scene,
    Stage,
)


def _make_layer(language: LanguageCode, content: str) -> Layer:
    layer = Layer(purpose=LayerPurpose.RAW_SOURCE, language=language)
    layer.add_prim(
        Prim(content=content, prim_type=PrimType.TERM, language=language)
    )
    return layer


class TestParallelLanguageLayers(unittest.TestCase):
    def setUp(self):
        self.stage = Stage(name="SPA_v3")
        self.stage.add_layer(_make_layer(LanguageCode.JA, "第3条 補償"))
        self.stage.add_layer(_make_layer(LanguageCode.KO, "제3조 손해배상"))
        self.stage.add_layer(_make_layer(LanguageCode.EN, "Clause 3: Indemnification"))

    def test_three_languages_coexist(self):
        self.assertEqual(self.stage.layer_count, 3)

    def test_language_specific_retrieval(self):
        ja = self.stage.get_layer(LayerPurpose.RAW_SOURCE, LanguageCode.JA)
        ko = self.stage.get_layer(LayerPurpose.RAW_SOURCE, LanguageCode.KO)
        en = self.stage.get_layer(LayerPurpose.RAW_SOURCE, LanguageCode.EN)
        self.assertEqual(ja.prims[0].content, "第3条 補償")
        self.assertEqual(ko.prims[0].content, "제3조 손해배상")
        self.assertEqual(en.prims[0].content, "Clause 3: Indemnification")

    def test_backward_compatible_default(self):
        # Without language, previous behaviour: last matching layer wins.
        last = self.stage.get_layer(LayerPurpose.RAW_SOURCE)
        self.assertEqual(last.language, LanguageCode.EN)

    def test_json_round_trip_preserves_all_languages(self):
        scene = Scene(name="Project Sakura")
        scene.add_stage(self.stage)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scene.json")
            scene.to_json(path)
            restored = Scene.from_json(path)
        stage = restored.stages[0]
        self.assertEqual(stage.layer_count, 3)
        for code, expected in [
            (LanguageCode.JA, "第3条 補償"),
            (LanguageCode.KO, "제3조 손해배상"),
            (LanguageCode.EN, "Clause 3: Indemnification"),
        ]:
            layer = stage.get_layer(LayerPurpose.RAW_SOURCE, code)
            self.assertIsNotNone(layer, f"layer for {code} lost in round trip")
            self.assertEqual(layer.prims[0].content, expected)


class TestPartOfReferences(unittest.TestCase):
    def test_section_clause_term_containment(self):
        """section > clause > term expressed via typed part-of References."""
        layer = Layer(purpose=LayerPurpose.RAW_SOURCE, language=LanguageCode.EN)
        section = layer.add_prim(
            Prim(content="Section 4: Indemnities", prim_type=PrimType.CONTEXT,
                 language=LanguageCode.EN)
        )
        clause = layer.add_prim(
            Prim(content="Clause 4.2: Cap on liability", prim_type=PrimType.CONTEXT,
                 language=LanguageCode.EN,
                 metadata={"part_of": section.prim_id})
        )
        clause.references.append(section.prim_id)
        term = layer.add_prim(
            Prim(content="liability cap", prim_type=PrimType.TERM,
                 language=LanguageCode.EN,
                 metadata={"part_of": clause.prim_id})
        )
        term.references.append(clause.prim_id)

        # Containment chain is recoverable from references alone.
        self.assertIn(clause.prim_id, term.references)
        self.assertIn(section.prim_id, clause.references)
        # And survives serialisation.
        restored = Prim.from_dict(term.to_dict())
        self.assertIn(clause.prim_id, restored.references)


if __name__ == "__main__":
    unittest.main()

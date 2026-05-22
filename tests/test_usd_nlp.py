"""Unit tests for the USD-NLP toolkit."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from usd_nlp.core import (
    Scene, Stage, Layer, Prim, Variant,
    LayerPurpose, LanguageCode, PrimType,
)
from usd_nlp.state_layers import StateLayerStack, LAYER_ORDER
from usd_nlp.twin import (
    LanguageTwin, InterpretationTwin, QoSThresholds, DegradationLevel,
)


class TestPrim(unittest.TestCase):

    def test_prim_has_prim_id(self):
        p = Prim(content="hello")
        self.assertTrue(p.prim_id)

    def test_prim_default_type_is_term(self):
        p = Prim(content="hello")
        self.assertEqual(p.prim_type, PrimType.TERM)

    def test_prim_type_values(self):
        self.assertEqual(PrimType.TERM.value, "T")
        self.assertEqual(PrimType.NUMERAL.value, "N")
        self.assertEqual(PrimType.VERB_PRED.value, "V")
        self.assertEqual(PrimType.HONORIFIC.value, "H")
        self.assertEqual(PrimType.CONTEXT.value, "C")

    def test_prim_add_variant(self):
        p = Prim(content="term")
        p.add_variant("alt", Variant(name="alt", content="rendering"))
        self.assertIn("alt", p.variants)

    def test_prim_add_reference(self):
        p = Prim(content="term")
        p.add_reference("other_id")
        self.assertIn("other_id", p.references)

    def test_prim_roundtrip(self):
        p = Prim(content="term", prim_type=PrimType.NUMERAL,
                 language=LanguageCode.JA)
        p.add_variant("v", Variant(name="v", content="x"))
        p2 = Prim.from_dict(p.to_dict())
        self.assertEqual(p2.content, p.content)
        self.assertEqual(p2.prim_type, PrimType.NUMERAL)
        self.assertIn("v", p2.variants)


class TestLayer(unittest.TestCase):

    def test_prims_is_list(self):
        layer = Layer(purpose=LayerPurpose.RAW_SOURCE)
        self.assertIsInstance(layer.prims, list)

    def test_add_prim_appends(self):
        layer = Layer(purpose=LayerPurpose.RAW_SOURCE)
        layer.add_prim(Prim(content="a"))
        layer.add_prim(Prim(content="b"))
        self.assertEqual(len(layer.prims), 2)

    def test_resolve_prim_returns_content(self):
        layer = Layer(purpose=LayerPurpose.RAW_SOURCE)
        p = Prim(content="original", prim_id="x1")
        layer.add_prim(p)
        self.assertEqual(layer.resolve_prim("x1"), "original")

    def test_override_is_non_destructive(self):
        layer = Layer(purpose=LayerPurpose.RAW_SOURCE)
        p = Prim(content="original", prim_id="x1")
        layer.add_prim(p)
        layer.add_override("x1", "amended", reason="test")
        # Original prim object is untouched
        self.assertEqual(layer.prims[0].content, "original")
        # Resolution returns the override
        self.assertEqual(layer.resolve_prim("x1"), "amended")

    def test_get_prims_by_type(self):
        layer = Layer(purpose=LayerPurpose.RAW_SOURCE)
        layer.add_prim(Prim(content="t", prim_type=PrimType.TERM))
        layer.add_prim(Prim(content="n", prim_type=PrimType.NUMERAL))
        self.assertEqual(len(layer.get_prims_by_type(PrimType.TERM)), 1)

    def test_layer_roundtrip(self):
        layer = Layer(purpose=LayerPurpose.TERM_BASE,
                      language=LanguageCode.KO)
        layer.add_prim(Prim(content="a", prim_id="p1"))
        layer.add_override("p1", "b")
        layer2 = Layer.from_dict(layer.to_dict())
        self.assertEqual(layer2.resolve_prim("p1"), "b")


class TestStage(unittest.TestCase):

    def test_add_and_get_layer(self):
        stage = Stage(name="doc")
        layer = Layer(purpose=LayerPurpose.RAW_SOURCE)
        stage.add_layer(layer)
        self.assertIsNotNone(stage.get_layer(LayerPurpose.RAW_SOURCE))

    def test_compose_resolves_prims(self):
        stage = Stage(name="doc")
        layer = Layer(purpose=LayerPurpose.RAW_SOURCE)
        layer.add_prim(Prim(content="c", prim_id="p1"))
        stage.add_layer(layer)
        self.assertEqual(stage.compose()["p1"], "c")

    def test_compose_strongest_layer_wins(self):
        stage = Stage(name="doc")
        l0 = Layer(purpose=LayerPurpose.RAW_SOURCE)
        l0.add_prim(Prim(content="raw", prim_id="p1"))
        l6 = Layer(purpose=LayerPurpose.REVIEW_STATE)
        l6.add_prim(Prim(content="reviewed", prim_id="p1"))
        stage.add_layer(l0)
        stage.add_layer(l6)
        self.assertEqual(stage.compose()["p1"], "reviewed")


class TestScene(unittest.TestCase):

    def test_add_stage_tracks_languages(self):
        scene = Scene(name="proj")
        stage = Stage(name="doc")
        stage.add_layer(Layer(purpose=LayerPurpose.RAW_SOURCE,
                              language=LanguageCode.JA))
        scene.add_stage(stage)
        self.assertIn(LanguageCode.JA, scene.languages)

    def test_find_prims_by_type(self):
        scene = Scene(name="proj")
        stage = Stage(name="doc")
        layer = Layer(purpose=LayerPurpose.RAW_SOURCE)
        layer.add_prim(Prim(content="t", prim_type=PrimType.TERM))
        layer.add_prim(Prim(content="n", prim_type=PrimType.NUMERAL))
        stage.add_layer(layer)
        scene.add_stage(stage)
        self.assertEqual(len(scene.find_prims(prim_type=PrimType.TERM)), 1)

    def test_summary_counts_prims(self):
        scene = Scene(name="proj")
        stage = Stage(name="doc")
        layer = Layer(purpose=LayerPurpose.RAW_SOURCE)
        layer.add_prim(Prim(content="a"))
        stage.add_layer(layer)
        scene.add_stage(stage)
        self.assertEqual(scene.summary()["num_prims"], 1)

    def test_json_roundtrip_in_memory(self):
        scene = Scene(name="proj")
        stage = Stage(name="doc")
        layer = Layer(purpose=LayerPurpose.RAW_SOURCE)
        layer.add_prim(Prim(content="a", prim_id="p1"))
        stage.add_layer(layer)
        scene.add_stage(stage)
        s = scene.to_json()
        scene2 = Scene.from_dict(json.loads(s))
        self.assertEqual(scene2.summary()["num_prims"], 1)

    def test_json_roundtrip_file(self):
        scene = Scene(name="proj")
        stage = Stage(name="doc")
        layer = Layer(purpose=LayerPurpose.RAW_SOURCE)
        layer.add_prim(Prim(content="a", prim_id="p1"))
        stage.add_layer(layer)
        scene.add_stage(stage)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "scene.json")
            scene.to_json(path)
            scene2 = Scene.from_json(path)
            self.assertEqual(scene2.summary()["num_prims"], 1)


class TestStateLayerStack(unittest.TestCase):

    def test_layer_order_has_seven(self):
        self.assertEqual(len(LAYER_ORDER), 7)

    def test_set_and_get_layer(self):
        stack = StateLayerStack()
        layer = Layer(purpose=LayerPurpose.RAW_SOURCE)
        stack.set_layer(layer)
        self.assertTrue(stack.has_layer(LayerPurpose.RAW_SOURCE))

    def test_compose_strongest_wins(self):
        stack = StateLayerStack()
        l0 = Layer(purpose=LayerPurpose.RAW_SOURCE)
        l0.add_prim(Prim(content="raw", prim_id="p1"))
        l5 = Layer(purpose=LayerPurpose.TRANSLATION_STATE)
        l5.add_prim(Prim(content="translated", prim_id="p1"))
        stack.set_layer(l0)
        stack.set_layer(l5)
        self.assertEqual(stack.compose()["p1"], "translated")

    def test_audit_log_records_mutations(self):
        stack = StateLayerStack()
        stack.set_layer(Layer(purpose=LayerPurpose.RAW_SOURCE))
        self.assertGreater(len(stack.audit_log), 0)

    def test_diff_detects_changes(self):
        stack = StateLayerStack()
        l0 = Layer(purpose=LayerPurpose.RAW_SOURCE)
        l0.add_prim(Prim(content="old", prim_id="p1"))
        l5 = Layer(purpose=LayerPurpose.TRANSLATION_STATE)
        l5.add_prim(Prim(content="new", prim_id="p1"))
        stack.set_layer(l0)
        stack.set_layer(l5)
        diffs = stack.diff(LayerPurpose.RAW_SOURCE,
                           LayerPurpose.TRANSLATION_STATE)
        self.assertIn("p1", diffs)

    def test_validate_flags_empty_content(self):
        stack = StateLayerStack()
        layer = Layer(purpose=LayerPurpose.RAW_SOURCE)
        layer.add_prim(Prim(content="", prim_id="p1"))
        stack.set_layer(layer)
        issues = stack.validate_all()
        self.assertTrue(issues["L0"])


class TestLanguageTwin(unittest.TestCase):

    def test_ingest_document_creates_stage(self):
        lt = LanguageTwin()
        lt.ingest_document("SPA", ["c1", "c2"], LanguageCode.JA)
        self.assertEqual(len(lt.scene.stages), 1)

    def test_add_term_with_variants(self):
        lt = LanguageTwin()
        lt.add_term("t1", "indemnification", LanguageCode.EN,
                    {"JP-GAAP": "x", "K-IFRS": "y"})
        self.assertEqual(lt.generate("t1", "K-IFRS"), "y")

    def test_update_helper(self):
        lt = LanguageTwin()
        lt.add_term("t1", "old", LanguageCode.EN)
        self.assertTrue(lt.update("t1", "new"))
        self.assertEqual(lt.retrieve("t1").content, "new")

    def test_update_missing_term_returns_false(self):
        lt = LanguageTwin()
        self.assertFalse(lt.update("missing", "x"))

    def test_generate_falls_back_to_content(self):
        lt = LanguageTwin()
        lt.add_term("t1", "base", LanguageCode.EN)
        self.assertEqual(lt.generate("t1", "unknown-jurisdiction"), "base")


class TestInterpretationTwin(unittest.TestCase):

    def test_qos_classify_nominal(self):
        q = QoSThresholds()
        self.assertEqual(q.classify(50.0), DegradationLevel.NOMINAL)

    def test_qos_classify_critical(self):
        q = QoSThresholds()
        self.assertEqual(q.classify(300.0), DegradationLevel.CRITICAL)

    def test_qos_classify_failure(self):
        q = QoSThresholds()
        self.assertEqual(q.classify(600.0), DegradationLevel.FAILURE)

    def test_record_latency_returns_level(self):
        it = InterpretationTwin()
        self.assertEqual(it.record_latency(50.0),
                         DegradationLevel.NOMINAL)

    def test_low_confidence_hint_rejected(self):
        it = InterpretationTwin(min_confidence=0.6)
        h = it.generate_hint("x", PrimType.NUMERAL, 0.3, now=0.0)
        self.assertIsNone(h)

    def test_high_confidence_hint_accepted(self):
        it = InterpretationTwin(min_confidence=0.6)
        h = it.generate_hint("x", PrimType.NUMERAL, 0.9, now=0.0)
        self.assertIsNotNone(h)

    def test_hint_ttl_expiry(self):
        it = InterpretationTwin()
        it.generate_hint("x", PrimType.NUMERAL, 0.9,
                         ttl_seconds=5.0, now=0.0)
        self.assertEqual(len(it.active_hints(now=2.0)), 1)
        self.assertEqual(len(it.active_hints(now=10.0)), 0)

    def test_session_analytics(self):
        it = InterpretationTwin()
        it.record_latency(50.0)
        it.record_latency(300.0)
        a = it.session_analytics()
        self.assertEqual(a["samples"], 2)


if __name__ == "__main__":
    unittest.main()

import json
import os
import tempfile
import unittest

from usd_nlp import *


class CoreTests(unittest.TestCase):
    def test_scene_layer_override_roundtrip(self):
        scene = Scene(name="Project Sakura")
        stage = Stage(name="SPA_v3")
        layer = Layer(purpose=LayerPurpose.RAW_SOURCE, language=LanguageCode.JA)
        p = layer.add_prim(Prim(content="Clause 3: Indemnification",
                                prim_type=PrimType.TERM,
                                language=LanguageCode.JA,
                                prim_id="c3"))
        layer.add_override("c3", "Clause 3: Indemnification (amended)", reason="SPA v3")
        self.assertEqual(p.content, "Clause 3: Indemnification")
        self.assertEqual(layer.resolve_prim("c3"), "Clause 3: Indemnification (amended)")
        stage.add_layer(layer)
        scene.add_stage(stage)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            scene.to_json(path)
            scene2 = Scene.from_json(path)
            self.assertEqual(scene2.summary()["num_prims"], 1)
            self.assertEqual(scene2.get_stage("SPA_v3").get_layer(LayerPurpose.RAW_SOURCE).resolve_prim("c3"),
                             "Clause 3: Indemnification (amended)")
        finally:
            os.unlink(path)

    def test_find_prims_and_references(self):
        scene = Scene(name="Test")
        stage = Stage(name="doc")
        layer = Layer(purpose=LayerPurpose.TERM_BASE)
        p1 = layer.add_prim(Prim(content="EV", prim_type=PrimType.TERM, prim_id="p1"))
        p2 = layer.add_prim(Prim(content="100", prim_type=PrimType.NUMERAL, prim_id="p2"))
        p2.add_reference("p1")
        stage.add_layer(layer)
        scene.add_stage(stage)
        self.assertEqual(len(scene.find_prims(prim_type=PrimType.TERM)), 1)
        self.assertEqual(p2.references, ["p1"])


class AdapterTests(unittest.TestCase):
    SAMPLE = """# sent_id = 1
1	The	the	DET	DT	Definite=Def|PronType=Art	2	det	_	_
2	buyer	buyer	NOUN	NN	Number=Sing	3	nsubj	_	_
3	pays	pay	VERB	VBZ	Mood=Ind|Tense=Pres	0	root	_	_
4	cash	cash	NOUN	NN	Number=Sing	3	obj	_	_
"""

    def test_conllu_importer_maps_tokens_and_dependencies(self):
        scene = from_conllu(self.SAMPLE, stage_name="spa", language=LanguageCode.EN)
        self.assertEqual(scene.summary()["num_prims"], 4)
        layer = scene.get_stage("spa").get_layer(LayerPurpose.STRUCTURAL)
        buyer = layer.get_prim("spa:s1:t2")
        self.assertEqual(buyer.metadata["lemma"], "buyer")
        self.assertEqual(buyer.metadata["upos"], "NOUN")
        self.assertEqual(buyer.metadata["head_prim_id"], "spa:s1:t3")
        self.assertEqual(len(scene.references), 3)
        self.assertEqual(scene.references[0].metadata["relation"], "dependency")

    def test_named_entity_span(self):
        scene = from_conllu(self.SAMPLE, stage_name="spa", language=LanguageCode.EN)
        layer = scene.get_stage("spa").get_layer(LayerPurpose.STRUCTURAL)
        span = add_named_entity_span(layer, "ne1", ["spa:s1:t2"], "ORG")
        self.assertEqual(span.metadata["annotation_kind"], "named_entity")
        self.assertEqual(span.references, ["spa:s1:t2"])


class StateLayerTests(unittest.TestCase):
    def test_state_stack_compose_diff_validate(self):
        stack = StateLayerStack()
        l0 = Layer(purpose=LayerPurpose.RAW_SOURCE)
        l0.add_prim(Prim(content="source", prim_id="p1"))
        l5 = Layer(purpose=LayerPurpose.TRANSLATION_STATE)
        l5.add_prim(Prim(content="translation", prim_id="p1", metadata={"translator": "EA"}))
        stack.add_layer(l0)
        stack.add_layer(l5)
        self.assertEqual(stack.compose()["p1"].content, "translation")
        self.assertEqual(stack.diff(LayerPurpose.RAW_SOURCE, LayerPurpose.TRANSLATION_STATE)["p1"],
                         ("source", "translation"))
        self.assertEqual(stack.validate(LayerPurpose.TRANSLATION_STATE), [])


class TwinTests(unittest.TestCase):
    def test_language_twin_json_restores_stacks(self):
        twin = LanguageTwin(source_lang=LanguageCode.JA)
        twin.ingest_document("doc", "full text", segments=["seg1", "seg2"])
        twin.add_term("企業価値", translations={"en": "Enterprise Value"})
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            twin.to_json(path)
            twin2 = LanguageTwin.from_json(path)
            self.assertEqual(twin2.term_count, 1)
            self.assertIsNotNone(twin2.get_document_state("doc"))
            self.assertIsNotNone(twin2.get_layer_stack("doc"))
        finally:
            os.unlink(path)

    def test_interpretation_qos(self):
        twin = InterpretationTwin(qos=QoSConfig(min_confidence=0.7))
        twin.start_session("session")
        self.assertIsNotNone(twin.add_hint("enterprise value", PrimType.TERM, confidence=0.9, latency_ms=100))
        self.assertIsNone(twin.add_hint("low", PrimType.TERM, confidence=0.1, latency_ms=100))
        failure = twin.add_hint("late", PrimType.TERM, confidence=0.9, latency_ms=5000)
        self.assertFalse(failure.delivered)
        self.assertEqual(twin.end_session()["total_hints"], 2)


if __name__ == "__main__":
    unittest.main()

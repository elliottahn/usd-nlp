"""USD-NLP Test Suite — 27 tests covering core, state_layers, and twin modules."""
import json
import os
import tempfile
import pytest
from usd_nlp import *


# ============================================================
# Core Module Tests (12 tests)
# ============================================================

class TestPrim:
    def test_create_prim(self):
        p = Prim(content="Enterprise Value", prim_type=PrimType.T)
        assert p.content == "Enterprise Value"
        assert p.prim_type == PrimType.T

    def test_prim_override(self):
        p = Prim(content="original")
        p.add_override("amended", reason="v2")
        assert p.resolved_content == "amended"
        assert p.override_count == 1

    def test_prim_remove_override(self):
        p = Prim(content="original")
        p.add_override("v2")
        p.add_override("v3")
        removed = p.remove_override()
        assert removed.content == "v3"
        assert p.resolved_content == "v2"

    def test_prim_variant(self):
        p = Prim(content="企業価値", prim_type=PrimType.T)
        p.add_variant("en", Variant(name="EV", content="Enterprise Value"))
        assert p.get_variant("en").content == "Enterprise Value"
        assert p.get_variant("fr") is None

    def test_prim_serialization(self):
        p = Prim(content="test", prim_type=PrimType.N, language=LanguageCode.JA)
        p.add_variant("ko", Variant(name="ko", content="테스트", language=LanguageCode.KO))
        p.add_override("amended")
        d = p.to_dict()
        p2 = Prim.from_dict(d)
        assert p2.content == "test"
        assert p2.prim_type == PrimType.N
        assert p2.get_variant("ko").content == "테스트"
        assert p2.resolved_content == "amended"


class TestLayer:
    def test_add_get_prim(self):
        layer = Layer(name="L0", purpose=LayerPurpose.RAW_SOURCE)
        p = Prim(id="t01", content="term")
        layer.add_prim(p)
        assert layer.get_prim("t01").content == "term"
        assert layer.prim_count == 1

    def test_remove_prim(self):
        layer = Layer(name="L0", purpose=LayerPurpose.RAW_SOURCE)
        p = Prim(id="t01", content="term")
        layer.add_prim(p)
        removed = layer.remove_prim("t01")
        assert removed.content == "term"
        assert layer.prim_count == 0

    def test_get_prims_by_type(self):
        layer = Layer(name="L0", purpose=LayerPurpose.RAW_SOURCE)
        layer.add_prim(Prim(id="t1", content="term1", prim_type=PrimType.T))
        layer.add_prim(Prim(id="n1", content="100", prim_type=PrimType.N))
        layer.add_prim(Prim(id="t2", content="term2", prim_type=PrimType.T))
        assert len(layer.get_prims_by_type(PrimType.T)) == 2
        assert len(layer.get_prims_by_type(PrimType.N)) == 1


class TestStageScene:
    def test_stage_layers(self):
        stage = Stage(name="SPA")
        l0 = Layer(name="raw", purpose=LayerPurpose.RAW_SOURCE)
        l1 = Layer(name="seg", purpose=LayerPurpose.SEGMENTED)
        stage.add_layer(l0)
        stage.add_layer(l1)
        assert stage.layer_count == 2
        assert stage.get_layer(LayerPurpose.RAW_SOURCE) is not None

    def test_scene_json_roundtrip(self):
        scene = Scene(name="Test")
        stage = Stage(name="doc1")
        layer = Layer(name="L0", purpose=LayerPurpose.RAW_SOURCE)
        layer.add_prim(Prim(id="p1", content="hello", prim_type=PrimType.C))
        stage.add_layer(layer)
        scene.add_stage(stage)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            scene.to_json(path)
            scene2 = Scene.from_json(path)
            assert scene2.name == "Test"
            assert scene2.get_stage("doc1").get_layer(
                LayerPurpose.RAW_SOURCE).get_prim("p1").content == "hello"
        finally:
            os.unlink(path)

    def test_scene_remove_stage(self):
        scene = Scene(name="Test")
        scene.add_stage(Stage(name="doc1"))
        removed = scene.remove_stage("doc1")
        assert removed.name == "doc1"
        assert scene.stage_count == 0


# ============================================================
# State Layer Stack Tests (6 tests)
# ============================================================

class TestStateLayerStack:
    def _make_stack(self):
        stack = StateLayerStack()
        l0 = Layer(name="L0", purpose=LayerPurpose.RAW_SOURCE)
        l0.add_prim(Prim(id="p1", content="original"))
        stack.add_layer(l0)
        l3 = Layer(name="L3", purpose=LayerPurpose.MT_DRAFT)
        l3.add_prim(Prim(id="p1", content="translated"))
        stack.add_layer(l3)
        return stack

    def test_compose_strongest_wins(self):
        stack = self._make_stack()
        resolved = stack.compose()
        assert resolved["p1"].content == "translated"  # L3 > L0

    def test_diff(self):
        stack = self._make_stack()
        diffs = stack.diff(LayerPurpose.RAW_SOURCE, LayerPurpose.MT_DRAFT)
        assert "p1" in diffs
        assert diffs["p1"] == ("original", "translated")

    def test_validate(self):
        stack = StateLayerStack()
        l6 = Layer(name="L6", purpose=LayerPurpose.FINALISED)
        l6.add_prim(Prim(id="p1", content="final", confidence=0.5))  # too low
        stack.add_layer(l6)
        errors = stack.validate(LayerPurpose.FINALISED)
        assert len(errors) > 0

    def test_validate_pass(self):
        stack = StateLayerStack()
        l6 = Layer(name="L6", purpose=LayerPurpose.FINALISED)
        l6.add_prim(Prim(id="p1", content="final", confidence=0.95))
        stack.add_layer(l6)
        errors = stack.validate(LayerPurpose.FINALISED)
        assert len(errors) == 0

    def test_audit_log(self):
        stack = StateLayerStack()
        l0 = Layer(name="L0", purpose=LayerPurpose.RAW_SOURCE)
        stack.add_layer(l0)
        stack.remove_layer(LayerPurpose.RAW_SOURCE)
        assert len(stack.audit_log) == 2
        assert stack.audit_log[0].action == "add_layer"
        assert stack.audit_log[1].action == "remove_layer"

    def test_processing_status(self):
        stack = self._make_stack()
        status = stack.get_processing_status()
        assert status["L0"] == 1
        assert status["L3"] == 1


# ============================================================
# Twin Module Tests (9 tests)
# ============================================================

class TestLanguageTwin:
    def test_ingest_document(self):
        twin = LanguageTwin(source_lang=LanguageCode.JA)
        stage = twin.ingest_document("SPA", "content",
                                      segments=["seg1", "seg2"])
        assert twin.document_count == 1
        assert stage.layer_count == 2

    def test_add_term(self):
        twin = LanguageTwin(source_lang=LanguageCode.JA)
        prim = twin.add_term("企業価値",
                             translations={"en": "Enterprise Value"})
        assert prim.get_variant("en").content == "Enterprise Value"
        assert twin.term_count == 1

    def test_get_terms_by_type(self):
        twin = LanguageTwin(source_lang=LanguageCode.JA)
        twin.add_term("t1", prim_type=PrimType.T)
        twin.add_term("100", prim_type=PrimType.N)
        assert len(twin.get_terms_by_type(PrimType.T)) == 1

    def test_document_state(self):
        twin = LanguageTwin(source_lang=LanguageCode.JA)
        twin.ingest_document("doc", "text", segments=["s1"])
        state = twin.get_document_state("doc")
        assert state is not None
        assert len(state) > 0

    def test_update_term(self):
        twin = LanguageTwin(source_lang=LanguageCode.JA)
        twin.add_term("EV", translations={"en": "Enterprise Value"})
        v = twin.update_term("EV", "ko", "기업가치")
        assert v.content == "기업가치"


class TestInterpretationTwin:
    def test_session_lifecycle(self):
        itwin = InterpretationTwin()
        stage = itwin.start_session("test")
        assert stage.name == "test"
        analytics = itwin.end_session()
        assert "total_hints" in analytics

    def test_add_hint_confidence_filter(self):
        itwin = InterpretationTwin(qos=QoSConfig(min_confidence=0.7))
        h1 = itwin.add_hint("term", PrimType.T, confidence=0.8)
        h2 = itwin.add_hint("low", PrimType.T, confidence=0.3)
        assert h1 is not None
        assert h2 is None
        assert itwin.hint_count == 1

    def test_qos_classification(self):
        qos = QoSConfig()
        assert qos.classify_latency(200) == DegradationLevel.NOMINAL
        assert qos.classify_latency(1000) == DegradationLevel.DEGRADED
        assert qos.classify_latency(2500) == DegradationLevel.CRITICAL
        assert qos.classify_latency(5000) == DegradationLevel.FAILURE

    def test_hint_delivery_failure(self):
        itwin = InterpretationTwin()
        h = itwin.add_hint("late", PrimType.T, latency_ms=5000)
        assert h.delivered is False
        assert h.degradation == DegradationLevel.FAILURE


class TestSerialization:
    def test_twin_to_dict(self):
        twin = LanguageTwin(source_lang=LanguageCode.JA)
        twin.ingest_document("doc", "text")
        twin.add_term("term", translations={"en": "term_en"})
        d = twin.to_dict()
        assert d["source_lang"] == "ja"
        assert "term" in d["term_base"]
        assert len(d["scene"]["stages"]) == 1

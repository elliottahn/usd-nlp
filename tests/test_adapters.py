"""Tests for USD-NLP import adapters."""
import unittest

from usd_nlp import LanguageCode, LayerPurpose
from usd_nlp.adapters import add_named_entity_span, from_conllu, parse_feats


class AdapterTests(unittest.TestCase):
    def test_parse_feats(self):
        self.assertEqual(parse_feats("Number=Sing|Tense=Past"), {"Number": "Sing", "Tense": "Past"})
        self.assertEqual(parse_feats("_"), {})

    def test_from_conllu_dependency_tree(self):
        conllu = """# sent_id = 1
# text = Counsel approved amendment.
1\tCounsel\tcounsel\tNOUN\tNN\tNumber=Sing\t2\tnsubj\t_\t_
2\tapproved\tapprove\tVERB\tVBD\tTense=Past|VerbForm=Fin\t0\troot\t_\t_
3\tamendment\tamendment\tNOUN\tNN\tNumber=Sing\t2\tobj\t_\t_
4\t.\t.\tPUNCT\t.\t_\t2\tpunct\t_\t_
"""
        scene = from_conllu(conllu, stage_name="doc", language=LanguageCode.EN)
        stage = scene.get_stage("doc")
        layer = stage.get_layer(LayerPurpose.STRUCTURAL)
        self.assertEqual(scene.summary()["num_prims"], 4)
        tokens = {p.metadata["token_id"]: p for p in layer.prims}
        self.assertTrue(tokens["2"].metadata["is_dependency_root"])
        self.assertEqual(tokens["3"].metadata["deprel"], "obj")
        self.assertEqual(tokens["3"].metadata["head_prim_id"], tokens["2"].prim_id)
        self.assertEqual(len(scene.references), 3)

    def test_named_entity_span(self):
        conllu = """# sent_id = 1
1\tBuyer\tbuyer\tNOUN\tNN\tNumber=Sing\t2\tnsubj\t_\t_
2\tpays\tpay\tVERB\tVBZ\tTense=Pres\t0\troot\t_\t_
"""
        scene = from_conllu(conllu, stage_name="spa", language="en")
        layer = scene.get_stage("spa").get_layer(LayerPurpose.STRUCTURAL)
        span = add_named_entity_span(layer, "ne1", ["spa:s1:t1"], "ORG")
        self.assertEqual(span.metadata["annotation_kind"], "named_entity")
        self.assertEqual(span.references, ["spa:s1:t1"])


if __name__ == "__main__":
    unittest.main()

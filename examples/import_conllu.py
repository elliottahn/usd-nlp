"""Example: import a small CoNLL-U dependency parse into USD-NLP."""

from usd_nlp.adapters import from_conllu

CONLLU = """# sent_id = demo-1
# text = Counsel approved the amendment.
1\tCounsel\tcounsel\tNOUN\tNN\tNumber=Sing\t2\tnsubj\t_\t_
2\tapproved\tapprove\tVERB\tVBD\tTense=Past|VerbForm=Fin\t0\troot\t_\t_
3\tthe\tthe\tDET\tDT\tDefinite=Def|PronType=Art\t4\tdet\t_\t_
4\tamendment\tamendment\tNOUN\tNN\tNumber=Sing\t2\tobj\t_\t_
5\t.\t.\tPUNCT\t.\t_\t2\tpunct\t_\t_
"""

scene = from_conllu(CONLLU, language="en", scene_name="CoNLL-U demo")
stage = scene.stages[0]
layer = stage.layers[0]

print(scene.summary())
for prim in layer.prims:
    print(prim.prim_id, prim.content, prim.metadata.get("upos"), prim.metadata.get("deprel"))

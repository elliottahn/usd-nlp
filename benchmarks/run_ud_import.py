"""UD treebank import pilot: representation-layer scalability check.

Imports the public test splits of three Universal Dependencies treebanks
(English-EWT, Japanese-GSD, Korean-GSD) through the CoNLL-U adapter and
reports import throughput, dependency-edge preservation, and JSON
round-trip cost. This measures representation-layer overhead only; it is
not an annotation-accuracy evaluation.

Usage:
    PYTHONPATH=src python3 benchmarks/run_ud_import.py <file1.conllu> [...]

Treebank files are available from
https://github.com/UniversalDependencies/ (test splits, CC BY-SA).
"""
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from usd_nlp.adapters import from_conllu  # noqa: E402

REPEATS = 5


def count_conllu(path):
    """Count sentences, syntactic-word tokens, and non-root HEAD edges."""
    sentences = 0
    tokens = 0
    edges = 0
    in_sentence = False
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                if in_sentence:
                    sentences += 1
                    in_sentence = False
                continue
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            token_id = fields[0]
            if "-" in token_id or "." in token_id:
                continue  # multi-word tokens / empty nodes are skipped
            in_sentence = True
            tokens += 1
            head = fields[6] if len(fields) > 6 else "_"
            if head not in ("_", "0"):
                edges += 1
    if in_sentence:
        sentences += 1
    return sentences, tokens, edges


def count_scene_edges(scene):
    """Count dependency References created by the adapter."""
    refs = 0
    prims = 0
    for stage in scene.stages:
        for layer in stage.layers:
            for prim in layer.prims:
                prims += 1
                refs += len(prim.references)
    return prims, refs


def bench(path):
    text = open(path, encoding="utf-8").read()
    sentences, tokens, edges_in = count_conllu(path)

    import_times = []
    scene = None
    for _ in range(REPEATS):
        start = time.perf_counter()
        scene = from_conllu(text, scene_name=os.path.basename(path))
        import_times.append(time.perf_counter() - start)
    import_s = statistics.median(import_times)

    prims, edges_out = count_scene_edges(scene)

    ser_times, deser_times = [], []
    payload = None
    for _ in range(REPEATS):
        start = time.perf_counter()
        payload = json.dumps(scene.to_dict())
        ser_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        json.loads(payload)
        deser_times.append(time.perf_counter() - start)

    return {
        "treebank": os.path.basename(path),
        "sentences": sentences,
        "tokens": tokens,
        "import_ms": round(import_s * 1000, 1),
        "tokens_per_s": int(tokens / import_s),
        "edges_in": edges_in,
        "edges_out": edges_out,
        "edge_preservation": round(edges_out / edges_in, 4) if edges_in else None,
        "json_mb": round(len(payload) / 1e6, 2),
        "ser_ms": round(statistics.median(ser_times) * 1000, 1),
        "deser_ms": round(statistics.median(deser_times) * 1000, 1),
    }


def main(paths):
    results = [bench(p) for p in paths]
    header = (
        f"{'treebank':<28}{'sents':>7}{'tokens':>9}{'tok/s':>10}"
        f"{'edges in':>10}{'edges out':>10}{'preserved':>10}"
        f"{'JSON MB':>9}{'ser ms':>8}{'deser ms':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['treebank']:<28}{r['sentences']:>7}{r['tokens']:>9}"
            f"{r['tokens_per_s']:>10,}{r['edges_in']:>10}{r['edges_out']:>10}"
            f"{r['edge_preservation']:>10.2%}{r['json_mb']:>9}{r['ser_ms']:>8}"
            f"{r['deser_ms']:>9}"
        )
    out = os.path.join(os.path.dirname(__file__), "ud_import_results.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])

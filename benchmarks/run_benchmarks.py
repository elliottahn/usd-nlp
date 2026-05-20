#!/usr/bin/env python3
"""
USD-NLP Performance Benchmarks

Measures:
1. Scene construction throughput (prims/second)
2. Layer composition latency (ms per stage)
3. JSON serialization round-trip time and size
4. Non-destructive override overhead (ops/second)
5. Cross-scene prim search latency
6. Memory footprint vs flat dict baseline

Run from repository root:  python benchmarks/run_benchmarks.py
"""

import sys, os, time, json, tracemalloc

# Allow running from repo root or from benchmarks/
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

from usd_nlp.core import (
    Scene, Stage, Layer, Prim, Variant,
    LanguageCode, LayerPurpose, PrimType
)


def bench_construction(n_stages=50, prims_per_stage=200, n_layers=3):
    """Benchmark: build a large scene."""
    start = time.perf_counter()
    scene = Scene(name="BenchScene")
    total_prims = 0
    for si in range(n_stages):
        stage = Stage(name=f"doc_{si}")
        for li in range(n_layers):
            lang = [LanguageCode.JA, LanguageCode.KO, LanguageCode.EN][li % 3]
            layer = Layer(purpose=LayerPurpose(f"L{li}"), language=lang)
            for pi in range(prims_per_stage):
                ptype = [PrimType.TERM, PrimType.NUMERAL, PrimType.VERB_PRED,
                         PrimType.HONORIFIC, PrimType.CONTEXT][pi % 5]
                prim = Prim(content=f"content_{si}_{li}_{pi}", prim_type=ptype,
                            language=lang, prim_id=f"p{si}_{li}_{pi}")
                if pi % 10 == 0:
                    prim.add_variant("alt", Variant(name="alt",
                                                    content=f"var_{pi}",
                                                    language=LanguageCode.EN))
                if pi > 0 and pi % 20 == 0:
                    prim.add_reference(f"p{si}_{li}_{pi-1}")
                layer.add_prim(prim)
                total_prims += 1
            stage.add_layer(layer)
        scene.add_stage(stage)
    elapsed = time.perf_counter() - start
    return scene, total_prims, elapsed


def bench_composition(scene):
    """Benchmark: compose all layers in all stages."""
    start = time.perf_counter()
    total_resolved = 0
    for stage in scene.stages:
        resolved = stage.compose()
        total_resolved += len(resolved)
    elapsed = time.perf_counter() - start
    return total_resolved, elapsed


def bench_serialization(scene):
    """Benchmark: JSON serialization round-trip."""
    start = time.perf_counter()
    json_str = scene.to_json()
    ser_time = time.perf_counter() - start
    size_kb = len(json_str.encode("utf-8")) / 1024

    start = time.perf_counter()
    scene2 = Scene.from_dict(json.loads(json_str))
    deser_time = time.perf_counter() - start

    assert scene.summary()["num_prims"] == scene2.summary()["num_prims"], \
        "Round-trip mismatch!"
    return ser_time, deser_time, size_kb


def bench_override(scene):
    """Benchmark: apply overrides (non-destructive versioning)."""
    start = time.perf_counter()
    n_overrides = 0
    for stage in scene.stages[:10]:
        for layer in stage.layers:
            for prim in layer.prims[:50]:
                layer.add_override(prim.prim_id, f"amended_{prim.prim_id}",
                                   reason="version amendment")
                n_overrides += 1
    elapsed = time.perf_counter() - start

    # Verify originals preserved
    sample_layer = scene.stages[0].layers[0]
    sample_prim = sample_layer.prims[0]
    assert sample_prim.content.startswith("content_"), "Original mutated!"
    assert sample_layer.resolve_prim(sample_prim.prim_id).startswith("amended_"), \
        "Override not applied!"
    return n_overrides, elapsed


def bench_search(scene):
    """Benchmark: cross-scene prim search."""
    start = time.perf_counter()
    terms = scene.find_prims(prim_type=PrimType.TERM)
    elapsed = time.perf_counter() - start
    return len(terms), elapsed


def bench_memory(n_prims=10000):
    """Benchmark: memory footprint of USD-NLP vs flat dict."""
    tracemalloc.start()
    scene = Scene(name="mem_test")
    stage = Stage(name="doc")
    layer = Layer(purpose=LayerPurpose.RAW_SOURCE, language=LanguageCode.JA)
    for i in range(n_prims):
        layer.add_prim(Prim(content=f"term_{i}", prim_id=f"p{i}"))
    stage.add_layer(layer)
    scene.add_stage(stage)
    usd_size = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()

    tracemalloc.start()
    flat = []
    for i in range(n_prims):
        flat.append({"id": f"p{i}", "content": f"term_{i}", "type": "T",
                     "language": "ja", "variants": {}, "refs": []})
    flat_size = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()

    ratio = usd_size / flat_size if flat_size > 0 else 0
    return usd_size / 1024, flat_size / 1024, ratio


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def main(n_runs=5):
    print("=" * 65)
    print(f"USD-NLP Performance Benchmarks (median of {n_runs} runs)")
    print("=" * 65)

    constr_rates, comp_times, ser_times, deser_times = [], [], [], []
    ov_rates, search_times, sizes = [], [], []

    # Repeat measurements to report a stable median rather than a
    # single noisy run.
    for _ in range(n_runs):
        scene, total_prims, t_constr = bench_construction(50, 200, 3)
        constr_rates.append(total_prims / t_constr)

        resolved, t_comp = bench_composition(scene)
        comp_times.append(t_comp / len(scene.stages) * 1000)

        t_ser, t_deser, size_kb = bench_serialization(scene)
        ser_times.append(t_ser * 1000)
        deser_times.append(t_deser * 1000)
        sizes.append(size_kb)

        n_ov, t_ov = bench_override(scene)
        ov_rates.append(n_ov / t_ov)

        n_found, t_search = bench_search(scene)
        search_times.append(t_search * 1000)

    constr_rate = median(constr_rates)
    comp_ms_per_stage = median(comp_times)
    t_ser_ms = median(ser_times)
    t_deser_ms = median(deser_times)
    size_kb = median(sizes)
    ov_ops_per_sec = median(ov_rates)
    search_ms = median(search_times)

    print(f"\n1. Scene Construction:  {constr_rate:,.0f} prims/sec "
          f"({total_prims:,} prims)")
    print(f"2. Layer Composition:   {comp_ms_per_stage:.2f} ms/stage "
          f"({resolved:,} prims resolved)")
    print(f"3. Serialization:       {t_ser_ms:.0f}ms ser / "
          f"{t_deser_ms:.0f}ms deser ({size_kb:.0f} KB)")
    print(f"4. Override:            {ov_ops_per_sec:,.0f} ops/sec "
          f"(non-destructive, originals preserved)")
    print(f"5. Search:              {n_found:,} prims in {search_ms:.1f}ms")

    # 6. Memory
    usd_kb, flat_kb, mem_ratio = bench_memory(10000)
    print(f"6. Memory Footprint:    {mem_ratio:.2f}x vs flat dict "
          f"(USD {usd_kb:.0f} KB / flat {flat_kb:.0f} KB)")

    # Summary — each metric from its own median variable
    print("\n" + "=" * 65)
    print("MANUSCRIPT-READY SUMMARY")
    print("=" * 65)
    print(f"Construction:  {constr_rate:,.0f} prims/sec ({total_prims:,} prims)")
    print(f"Composition:   {comp_ms_per_stage:.2f} ms/stage")
    print(f"Serialization: {t_ser_ms:.0f}ms ser + {t_deser_ms:.0f}ms "
          f"deser ({size_kb:.0f} KB)")
    print(f"Override:      {ov_ops_per_sec:,.0f} ops/sec (non-destructive)")
    print(f"Search:        {n_found:,} prims in {search_ms:.1f}ms")
    print(f"Memory:        {mem_ratio:.2f}x vs flat dict")

    # Export — each value from its own median variable
    results = {
        "n_runs": n_runs,
        "construction_prims_per_sec": round(constr_rate),
        "construction_total_prims": total_prims,
        "composition_ms_per_stage": round(comp_ms_per_stage, 2),
        "serialization_ms": round(t_ser_ms),
        "deserialization_ms": round(t_deser_ms),
        "serialized_size_kb": round(size_kb),
        "override_ops_per_sec": round(ov_ops_per_sec),
        "search_prims_found": n_found,
        "search_ms": round(search_ms, 1),
        "memory_overhead_ratio": round(mem_ratio, 2),
    }

    # Ensure output directory exists before writing
    os.makedirs(_HERE, exist_ok=True)
    out_path = os.path.join(_HERE, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {os.path.relpath(out_path, _ROOT)}")


if __name__ == "__main__":
    main()

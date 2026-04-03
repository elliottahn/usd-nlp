"""Example: RSI Session with QoS Monitoring using USD-NLP.

Simulates a BOJ press conference interpretation (JP→KR) with
typed hint delivery, latency recording, and degradation response.
"""
import random
from usd_nlp import *

def main():
    random.seed(42)

    # Configure QoS thresholds
    qos = QoSConfig(
        latency_nominal_ms=500,
        latency_degraded_ms=1500,
        latency_critical_ms=3000,
        hint_ttl_seconds=10.0,
        min_confidence=0.6,
    )

    # Create Interpretation Twin
    itwin = InterpretationTwin(
        source_lang=LanguageCode.JA,
        target_lang=LanguageCode.KO,
        qos=qos,
    )

    # Start session
    itwin.start_session("BOJ_Press_Conference_2025Q1")

    # Simulate utterances with varying types, latencies, and confidence
    utterances = [
        ("金融政策の変更について", PrimType.T, 0.95, 300),
        ("基準金利を0.25%に据え置き", PrimType.N, 0.92, 450),
        ("総裁は慎重な姿勢を示し", PrimType.H, 0.88, 800),
        ("次に物価見通しについて", PrimType.C, 0.90, 200),
        ("コアCPIは前年比2.3%", PrimType.N, 0.95, 350),
        ("ご質問をお受けいたします", PrimType.H, 0.85, 1200),
        ("為替市場への影響は限定的", PrimType.T, 0.78, 2800),
        ("以上で記者会見を終了", PrimType.C, 0.92, 150),
        ("不確実性が高い状況", PrimType.V, 0.45, 400),  # low confidence → filtered
        ("量的緩和の縮小ペース", PrimType.T, 0.91, 4500),  # high latency → failure
    ]

    print("=== BOJ Press Conference RSI Session ===\n")
    for content, ptype, conf, lat in utterances:
        hint = itwin.add_hint(content, ptype,
                              confidence=conf, latency_ms=lat)
        if hint is None:
            print(f"  FILTERED (conf={conf:.2f}): {content[:30]}")
        elif not hint.delivered:
            print(f"  FAILURE  (lat={lat}ms): {content[:30]}")
        else:
            print(f"  {hint.degradation.value:>8s} ({lat:>4d}ms) [{ptype.value[0].upper()}]: {content[:30]}")

    # Session analytics
    analytics = itwin.end_session()
    print(f"\n=== Session Analytics ===")
    print(f"Total hints: {analytics['total_hints']}")
    print(f"Delivered: {analytics['delivered']}")
    print(f"Delivery rate: {analytics['delivery_rate']:.0%}")
    print(f"Mean latency: {analytics['mean_latency_ms']:.0f}ms")
    print(f"Degradation: {analytics['degradation_counts']}")
    print(f"By type: {analytics['type_counts']}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Example: remote simultaneous interpreting session with QoS monitoring.

Demonstrates the InterpretationTwin for a simulated central-bank press
conference: typed hint generation, latency recording, and graduated
degradation response.

Run:  python examples/rsi_qos_session.py
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from usd_nlp.core import PrimType
from usd_nlp.twin import InterpretationTwin, QoSThresholds


def main():
    # Configure a twin with explicit QoS thresholds
    twin = InterpretationTwin(
        name="CentralBankPresser",
        thresholds=QoSThresholds(degraded_ms=150.0,
                                 critical_ms=250.0,
                                 failure_ms=500.0),
        min_confidence=0.6,
    )

    # Simulated latency trace over the session (milliseconds)
    latency_trace = [48, 52, 61, 95, 140, 180, 240, 310, 270, 160, 70]
    print("Latency trace (degradation level per sample):")
    for i, lat in enumerate(latency_trace):
        level = twin.record_latency(float(lat))
        print(f"  t={i:2d}  {lat:4d} ms  ->  {level.value}")

    # Typed hints generated during the session.
    # Numerals get a short TTL: they must be delivered quickly because
    # numbers are hard to retain in working memory.
    twin.generate_hint("policy rate: 3.50%", PrimType.NUMERAL,
                       confidence=0.95, ttl_seconds=4.0, now=0.0)
    twin.generate_hint("year-on-year: 2.1 percent", PrimType.NUMERAL,
                       confidence=0.88, ttl_seconds=4.0, now=1.0)
    # A terminology hint tolerates a longer TTL.
    twin.generate_hint("quantitative tightening", PrimType.TERM,
                       confidence=0.91, ttl_seconds=10.0, now=1.0)
    # A low-confidence hint is filtered out.
    rejected = twin.generate_hint("uncertain phrase", PrimType.CONTEXT,
                                  confidence=0.40, ttl_seconds=8.0,
                                  now=1.0)
    print(f"\nLow-confidence hint rejected: {rejected is None}")
    print(f"Active hints at t=2 s : {len(twin.active_hints(now=2.0))}")
    print(f"Active hints at t=6 s : {len(twin.active_hints(now=6.0))}")

    print("\nSession analytics:")
    for key, value in twin.session_analytics().items():
        print(f"  {key}: {value}")

    print("\n[OK] RSI QoS session example completed.")


if __name__ == "__main__":
    main()

"""Remote interpreting QoS example."""
from usd_nlp import InterpretationTwin, PrimType, QoSConfig

twin = InterpretationTwin(qos=QoSConfig(min_confidence=0.6))
twin.start_session("central_bank_press_conference")
twin.add_hint("basis points", PrimType.TERM, confidence=0.9, latency_ms=120)
twin.add_hint("2.5 percent", PrimType.NUMERAL, confidence=0.95, latency_ms=210)
twin.add_hint("late hint", PrimType.CONTEXT, confidence=0.8, latency_ms=5000)
print(twin.end_session())

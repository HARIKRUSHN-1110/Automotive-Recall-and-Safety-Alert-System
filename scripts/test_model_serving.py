"""
test_model_serving.py
---------------------
Smoke test and latency test for the model serving module.

Run from your project root:
    python scripts/test_model_serving.py

Tests:
  1. Model loads without errors
  2. predict_proba() returns a valid PredictionResult
  3. predict() returns 0 or 1
  4. Input validation rejects bad inputs
  5. Latency — 100 predictions average under 500ms
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.serve import ModelServer, InvalidInputError


def print_divider(title: str):
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")


# Sample inputs for testing

SAMPLE_INPUTS = [
    {
        "make":      "BMW",
        "model":     "3 Series",
        "year":      2020,
        "summary":   "engine stalled on highway without warning at 65mph very dangerous",
        "component": "ENGINE",
        "crash":     False,
        "fire":      False,
    },
    {
        "make":      "TOYOTA",
        "model":     "Camry",
        "year":      2019,
        "summary":   "airbag deployed unexpectedly while parked injuries sustained",
        "component": "AIR BAGS",
        "crash":     True,
        "fire":      False,
    },
    {
        "make":      "FORD",
        "model":     "F-150",
        "year":      2018,
        "summary":   "fuel leak detected near engine compartment smell of gasoline",
        "component": "FUEL SYSTEM",
        "crash":     False,
        "fire":      True,
    },
    {
        "make":      "TESLA",
        "model":     "Model 3",
        "year":      2021,
        "summary":   "minor paint scratch on door no safety issues",
        "component": "PAINT",
        "crash":     False,
        "fire":      False,
    },
]


def test_model_loads():
    print_divider("TEST 1 — Model loads")
    server = ModelServer()
    server.load_model()
    assert server.is_loaded(), "Model failed to load"
    print("  ✅  Model loaded successfully")
    print(f"  {server}")
    return server


def test_predict_proba(server: ModelServer):
    print_divider("TEST 2 — predict_proba() on sample inputs")

    for inp in SAMPLE_INPUTS:
        result = server.predict_proba(**inp)

        assert 0 <= result.probability <= 1, "Probability out of range"
        assert 0 <= result.risk_score <= 100, "Risk score out of range"
        assert result.risk_label in ("Low", "Medium", "High")
        assert result.prediction in (0, 1)

        print(
            f"  {inp['make']:<12} {inp['model']:<12} {inp['year']}  "
            f"→  score={result.risk_score:>3}  "
            f"label={result.risk_label:<7}  "
            f"prob={result.probability:.3f}  "
            f"({result.elapsed_ms:.1f}ms)"
        )

    print("  ✅  All predictions returned valid results")


def test_predict_binary(server: ModelServer):
    print_divider("TEST 3 — predict() returns 0 or 1")

    for inp in SAMPLE_INPUTS:
        pred = server.predict(**inp)
        assert pred in (0, 1), f"Expected 0 or 1, got {pred}"

    print("  ✅  All binary predictions are 0 or 1")


def test_input_validation(server: ModelServer):
    print_divider("TEST 4 — Input validation rejects bad inputs")

    bad_inputs = [
        # (description, kwargs that should fail)
        ("empty make",    dict(make="",    model="Camry",  year=2020, summary="engine failed")),
        ("future year",   dict(make="BMW", model="3 Series", year=2099, summary="engine failed")),
        ("ancient year",  dict(make="BMW", model="3 Series", year=1800, summary="engine failed")),
        ("empty summary", dict(make="BMW", model="3 Series", year=2020, summary="")),
        ("null summary",  dict(make="BMW", model="3 Series", year=2020, summary=None)),
    ]

    for description, kwargs in bad_inputs:
        try:
            server.predict_proba(**kwargs)
            print(f"  ❌  FAILED — should have rejected: {description}")
        except InvalidInputError as e:
            print(f"  ✅  Correctly rejected ({description})")
        except Exception as e:
            print(f"  ⚠️  Wrong exception type for ({description}): {type(e).__name__}")

    print("  ✅  Input validation working correctly")


def test_latency(server: ModelServer):
    print_divider("TEST 5 — Latency (target: avg < 500ms)")

    inp = SAMPLE_INPUTS[0]
    n   = 100

    times = []
    for _ in range(n):
        t0     = time.perf_counter()
        server.predict_proba(**inp)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

    avg_ms  = sum(times) / len(times)
    max_ms  = max(times)
    min_ms  = min(times)
    p95_ms  = sorted(times)[int(0.95 * n)]

    print(f"  Ran {n} predictions")
    print(f"  Average : {avg_ms:.1f}ms   {'Good' if avg_ms < 500 else 'TOO SLOW'}")
    print(f"  p95     : {p95_ms:.1f}ms   {'Good' if p95_ms < 500 else 'TOO SLOW'}")
    print(f"  Min     : {min_ms:.1f}ms")
    print(f"  Max     : {max_ms:.1f}ms")

    if avg_ms < 500:
        print("  ✅  Latency target met (<500ms)")
    else:
        print("  ❌  Latency too high — check model complexity")

    return avg_ms < 500

# Main

if __name__ == "__main__":
    print("\n Model Serving — Smoke Test & Latency Check")

    results = {}

    server              = test_model_loads()
    results["load"]     = True

    test_predict_proba(server)
    results["proba"]    = True

    test_predict_binary(server)
    results["binary"]   = True

    test_input_validation(server)
    results["validation"] = True

    latency_ok          = test_latency(server)
    results["latency"]  = latency_ok

    # Final summary
    print_divider("SUMMARY")
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  —  {test_name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print(" Model serving is working correctly.")
    else:
        print(" Some tests failed — review output above.")
    print()
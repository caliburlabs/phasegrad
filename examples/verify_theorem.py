#!/usr/bin/env python3
"""Reproduce the Phase-Gradient Duality verification table.

Expected output:
      N    residual      an vs fd      tp vs fd
      6    ~1e-17       +1.000000     +1.000000
     10    ~1e-17       +1.000000     +1.000000
     15    ~1e-17       +1.000000     +1.000000
     20    ~1e-16       +1.000000     +1.000000
"""

from phasegrad.verification import run_verification

if __name__ == "__main__":
    print("Phase-Gradient Duality — Theorem Verification\n")
    results = run_verification(sizes=[6, 10, 15, 20], seed=42)
    print("\nAll sizes verified." if all(
        r["best_cos_tp_fd"] > 0.999 for r in results
    ) else "\nSome sizes FAILED.")

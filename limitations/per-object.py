#!/usr/bin/env python3
"""
per_object_ladder_vs_static_exaggerated.py

One plot over time showing:
- Available bandwidth (fluctuating, intuitive piecewise trace)
- Selected bitrates for 2 objects: near vs far
  comparing:
    (A) ORACLE per-object ladders (fine, content-aware ranges)
    (B) STATIC global ladder (coarse shared rungs)

Key idea:
  Static ladder forces both objects onto the same coarse rungs,
  so far object gets silly jumps / wasted bits, and near object gets stuck or pops.
  Oracle ladders are tailored: far has small meaningful rates; near has large meaningful rates.

Requires:
  pip install numpy matplotlib
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import math


def best_combo_under_budget(budget, ladder_far, ladder_near, w_far=4.0, w_near=9.0):
    """Choose (rf, rn) from ladders to maximize weighted log utility under rf+rn<=budget."""
    best_rf, best_rn = 0.0, 0.0
    best_u = -1e18

    for rf in ladder_far:
        for rn in ladder_near:
            if rf + rn <= budget + 1e-9:
                u = w_far * math.log(1.0 + rf) + w_near * math.log(1.0 + rn)
                if u > best_u:
                    best_u = u
                    best_rf, best_rn = rf, rn

    # if budget too small for any rung pair, send nothing
    if best_u < -1e17:
        return 0.0, 0.0
    return best_rf, best_rn


def make_intuitive_bandwidth_trace(T, seed=1):
    """
    Piecewise "intuitive" network trace:
      - warmup moderate
      - ramp up
      - sudden congestion drop
      - recovery
      - oscillatory variability

    Returns Mbps array length T.
    """
    rng = np.random.default_rng(seed)
    bw = np.zeros(T, dtype=float)

    # segment lengths (scaled to T)
    a = int(0.18 * T)
    b = int(0.22 * T)
    c = int(0.12 * T)
    d = int(0.20 * T)
    e = T - (a + b + c + d)

    # 1) moderate, slightly noisy
    t = np.linspace(0, 1, a, endpoint=False)
    bw[:a] = 35 + 4 * np.sin(2 * np.pi * t) + 1.0 * rng.standard_normal(a)

    # 2) ramp up (good conditions)
    t = np.linspace(0, 1, b, endpoint=False)
    bw[a:a+b] = 50 + 35 * t + 2 * np.sin(4 * np.pi * t) + 1.5 * rng.standard_normal(b)

    # 3) sudden congestion drop
    t = np.linspace(0, 1, c, endpoint=False)
    bw[a+b:a+b+c] = 28 - 10 * t + 1.0 * rng.standard_normal(c)

    # 4) recovery ramp
    t = np.linspace(0, 1, d, endpoint=False)
    bw[a+b+c:a+b+c+d] = 32 + 55 * t + 2 * np.sin(2 * np.pi * t) + 1.2 * rng.standard_normal(d)

    # 5) oscillatory variability
    t = np.linspace(0, 1, e, endpoint=False)
    bw[a+b+c+d:] = 95 + 18 * np.sin(2 * np.pi * 1.5 * t) + 8 * np.sin(2 * np.pi * 6 * t) + 2.0 * rng.standard_normal(e)

    # Smooth a bit (so it looks like throughput, not white noise)
    win = 9
    k = np.ones(win) / win
    bw = np.convolve(bw, k, mode="same")

    # Clamp to realistic bounds
    bw = np.clip(bw, 5, 130)
    return bw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=240, help="timesteps")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--w_near", type=float, default=10.0, help="utility weight for near object")
    ap.add_argument("--w_far", type=float, default=1.0, help="utility weight for far object")
    args = ap.parse_args()

    # ---------- EXAGGERATED ORACLE LADDERS (tailored per object) ----------
    # Far object: meaningful rates are tiny; fine steps help avoid waste/popping.
    oracle_far = np.array([1, 3.5, 5, 7, 10, 14, 20.0], dtype=float)
    # Near object: meaningful rates are large; needs fine control at high Mbps.
    oracle_near = np.array([25, 35, 45, 60, 75, 95, 120], dtype=float)

    # ---------- STATIC GLOBAL LADDER (shared and coarse) ----------
    static_global = np.array([1, 20, 40, 60, 80, 100], dtype=float)

    # Intuitive fluctuating throughput
    bw = make_intuitive_bandwidth_trace(args.T, seed=args.seed)

    # Selections over time
    o_far = np.zeros(args.T); o_near = np.zeros(args.T)
    s_far = np.zeros(args.T); s_near = np.zeros(args.T)

    for t in range(args.T):
        B = float(bw[t])

        rf, rn = best_combo_under_budget(B, oracle_far, oracle_near, w_far=args.w_far, w_near=args.w_near)
        o_far[t], o_near[t] = rf, rn

        rf, rn = best_combo_under_budget(B, static_global, static_global, w_far=args.w_far, w_near=args.w_near)
        s_far[t], s_near[t] = rf, rn

    # ---- One plot ----
    x = np.arange(args.T)

    plt.figure()
    plt.plot(x, bw, label="Available bandwidth (Mbps)")
    plt.plot(x, o_near, label="Oracle per-object ladder: Near")
    plt.plot(x, o_far,  label="Oracle per-object ladder: Far")
    plt.plot(x, s_near, label="Static global ladder: Near")
    plt.plot(x, s_far,  label="Static global ladder: Far")

    plt.xlabel("time")
    plt.ylabel("Mbps")
    plt.title("Static ladders are coarse + mismatched; per-object ladders track content needs under bandwidth swings")
    plt.ylim(0, 135)
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.show()


if __name__ == "__main__":
    main()


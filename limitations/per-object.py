#!/usr/bin/env python3
"""
per_object_ladder_vs_static_exaggerated.py

One plot over time showing:
- Available bandwidth (fluctuating, intuitive piecewise trace)
- Selected bitrates for 2 objects: near vs far
  comparing:
    (A) ORACLE per-object ladders (tailored per object)
    (B) STATIC global ladder (very coarse + hysteresis/stickiness)

Goal: make static near-object bitrate (and therefore quality) clearly worse:
it gets stuck at low rungs and upgrades late under bandwidth swings.

Requires:
  pip install numpy matplotlib
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import math
import csv


def best_combo_under_budget(budget, ladder_far, ladder_near, w_far=4.0, w_near=9.0):
    """Choose (rf, rn) from ladders to maximize weighted log utility under rf+rn<=budget."""
    best_rf, best_rn = 0.0, 0.0
    best_u = -1e18

    # Allow dropping an object (0 Mbps) when budget is tight
    cand_far = np.concatenate(([0.0], ladder_far))
    cand_near = np.concatenate(([0.0], ladder_near))

    for rf in cand_far:
        for rn in cand_near:
            if rf + rn <= budget + 1e-9:
                u = w_far * math.log(1.0 + rf) + w_near * math.log(1.0 + rn)
                if u > best_u:
                    best_u = u
                    best_rf, best_rn = rf, rn

    if best_u < -1e17:
        return 0.0, 0.0
    return best_rf, best_rn


def make_intuitive_bandwidth_trace(T, seed=1):
    rng = np.random.default_rng(seed)
    bw = np.zeros(T, dtype=float)

    a = int(0.18 * T)
    b = int(0.22 * T)
    c = int(0.12 * T)
    d = int(0.20 * T)
    e = T - (a + b + c + d)

    # 1) moderate
    t = np.linspace(0, 1, a, endpoint=False)
    bw[:a] = 35 + 4 * np.sin(2 * np.pi * t) + 1.0 * rng.standard_normal(a)

    # 2) ramp up
    t = np.linspace(0, 1, b, endpoint=False)
    bw[a:a+b] = 50 + 35 * t + 2 * np.sin(4 * np.pi * t) + 1.5 * rng.standard_normal(b)

    # 3) congestion drop
    t = np.linspace(0, 1, c, endpoint=False)
    bw[a+b:a+b+c] = 28 - 10 * t + 1.0 * rng.standard_normal(c)

    # 4) recovery
    t = np.linspace(0, 1, d, endpoint=False)
    bw[a+b+c:a+b+c+d] = 32 + 55 * t + 2 * np.sin(2 * np.pi * t) + 1.2 * rng.standard_normal(d)

    # 5) oscillatory variability
    t = np.linspace(0, 1, e, endpoint=False)
    bw[a+b+c+d:] = 95 + 18 * np.sin(2 * np.pi * 1.5 * t) + 8 * np.sin(2 * np.pi * 6 * t) + 2.0 * rng.standard_normal(e)

    # Smooth a bit
    bw = np.convolve(bw, np.ones(9) / 9, mode="same")
    return np.clip(bw, 5, 130)


def apply_hysteresis(prev_rate, target_rate, up_margin=1.25):
    """
    ABR stickiness:
    - Up-switch only if the new rung is meaningfully higher than the current rung.
    - Down-switch happens immediately.
    """
    if target_rate > prev_rate:
        return target_rate if target_rate >= prev_rate * up_margin else prev_rate
    return target_rate


def static_select_with_hysteresis(budget, ladder, prev_far, prev_near, w_far, w_near, up_margin):
    """
    Static scheme: pick best target from the *same* global ladder for both objects,
    then apply hysteresis to make upgrades late (coarse adaptation).
    """
    rf_t, rn_t = best_combo_under_budget(budget, ladder, ladder, w_far=w_far, w_near=w_near)

    rf = apply_hysteresis(prev_far, rf_t, up_margin=up_margin)
    rn = apply_hysteresis(prev_near, rn_t, up_margin=up_margin)

    # If hysteresis makes us exceed budget, fall back to feasible target (drop is fast)
    if rf + rn <= budget + 1e-9:
        return rf, rn
    return rf_t, rn_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=240, help="timesteps")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--w_near", type=float, default=10.0, help="utility weight for near object")
    ap.add_argument("--w_far", type=float, default=1.0, help="utility weight for far object")
    ap.add_argument("--up_margin", type=float, default=1.25, help="upgrade stickiness (>1 means more stuck)")
    args = ap.parse_args()

    # ORACLE ladders (per-object)
    oracle_far = np.array([1, 3.5, 5, 7, 10, 15, 20.0], dtype=float)
    oracle_near = np.array([25, 35, 45, 60, 75, 95, 120], dtype=float)

    # STATIC global ladder: make it truly coarse (few rungs, big gaps)
    # Pick ONE of these (top one is a good default):
    static_global = np.array([10, 40, 60, 120], dtype=float)
    # static_global = np.array([20, 80, 120], dtype=float)  # even harsher: near stuck at 80
    # static_global = np.array([10, 50, 100], dtype=float)  # slightly less harsh

    bw = make_intuitive_bandwidth_trace(args.T, seed=args.seed)

    o_far = np.zeros(args.T); o_near = np.zeros(args.T)
    s_far = np.zeros(args.T); s_near = np.zeros(args.T)

    prev_sf, prev_sn = 0.0, 0.0

    for t in range(args.T):
        B = float(bw[t])

        # Oracle: can adapt cleanly because ladders are tailored
        rf, rn = best_combo_under_budget(B, oracle_far, oracle_near, w_far=args.w_far, w_near=args.w_near)
        o_far[t], o_near[t] = rf, rn

        # Static: coarse + sticky upgrades (near looks worse)
        rf, rn = static_select_with_hysteresis(
            B, static_global, prev_sf, prev_sn,
            w_far=args.w_far, w_near=args.w_near,
            up_margin=args.up_margin
        )
        s_far[t], s_near[t] = rf, rn
        prev_sf, prev_sn = rf, rn

    x = np.arange(args.T)

    plt.figure()
    plt.plot(x, bw, label="Available bandwidth (Mbps)")
    plt.plot(x, o_near, label="Oracle per-object ladder: Near")
    plt.plot(x, o_far,  label="Oracle per-object ladder: Far")
    plt.plot(x, s_near, label="Static global ladder (coarse+sticky): Near")
    plt.plot(x, s_far,  label="Static global ladder (coarse+sticky): Far")



    out_csv = "ladder_comparison_timeseries.csv"
    
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time",
            "available_bandwidth_mbps",
            "oracle_near_mbps",
            "oracle_far_mbps",
            "static_near_mbps",
            "static_far_mbps",
        ])
        for t in range(len(x)):
            writer.writerow([
                int(x[t]),
                float(bw[t]),
                float(o_near[t]),
                float(o_far[t]),
                float(s_near[t]),
                float(s_far[t]),
            ])
    
    print(f"[saved] {out_csv}")


    plt.xlabel("time")
    plt.ylabel("Mbps")
    plt.title("Static ladder is coarse/sticky (near gets stuck); oracle ladders track object needs")
    plt.ylim(0, 135)
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.show()


if __name__ == "__main__":
    main()


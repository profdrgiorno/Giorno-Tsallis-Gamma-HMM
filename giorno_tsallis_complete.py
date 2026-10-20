# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 20:09:10 2026

@author: DRGIORNO
Giorno–Tsallis Dynamic Non-Extensive Entropy Estimator
Complete pipeline: Monte Carlo + Inference / Confidence Intervals

Supports:
  - Training lengths n = 1000, 2000, 3000
  - Large number of replications
  - 6-parameter and 9-parameter models
  - Exact filter → dynamic q_t and dynamic Tsallis entropy
  - Observed-information Wald CIs (HMM-6)
  - Parametric bootstrap CIs for key parameters of HMM-9 (q0, q1, ε)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize
from scipy.special import gammaln, gammainc
from scipy.signal import fftconvolve
from scipy.stats import t as student_t, gamma as gamma_dist, norm
from scipy.integrate import quad
from scipy.linalg import inv

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# True parameters
# ---------------------------------------------------------------------------
QTRUE = np.array([1.12, 1.42])
NUTRUE = (3.0 - QTRUE) / (QTRUE - 1.0)
STRUE = np.array([0.80, 1.25])
PTRUE = np.array([[0.985, 0.015], [0.040, 0.960]])
NTEST = 400
SEED = 20261010

GRID_STEP = 0.025
GRID_LIMIT = 250.0
GRID = np.linspace(-GRID_LIMIT, GRID_LIMIT, int(round(2 * GRID_LIMIT / GRID_STEP)) + 1)
GRID_NEG = GRID <= 0
GRID_G = (-GRID[GRID_NEG]).copy()

# ---------------------------------------------------------------------------
# Core model functions
# ---------------------------------------------------------------------------

@njit(cache=True)
def forward_filter(le: np.ndarray, p00: float, p11: float):
    n = le.shape[0]
    states = np.empty(n)
    score = np.empty(n)
    a = 0.5
    b = 0.5
    logl = 0.0
    for i in range(n):
        if i:
            pa = a * p00 + b * (1.0 - p11)
            pb = a * (1.0 - p00) + b * p11
        else:
            pa = 0.5
            pb = 0.5
        m = le[i, 0] if le[i, 0] > le[i, 1] else le[i, 1]
        e0 = math.exp(le[i, 0] - m)
        e1 = math.exp(le[i, 1] - m)
        den = pa * e0 + pb * e1
        if den <= 0.0 or not np.isfinite(den):
            return -1e100, states, score
        a = pa * e0 / den
        b = pb * e1 / den
        sc = math.log(den) + m
        states[i] = b
        score[i] = sc
        logl += sc
    return logl, states, score


def t_logpdf(y, nu, s):
    v = np.asarray(y)
    const = float(
        gammaln((nu + 1) * 0.5)
        - gammaln(nu * 0.5)
        - 0.5 * np.log(nu * np.pi)
        - np.log(s)
    )
    return const - 0.5 * (nu + 1) * np.log1p((v / s) ** 2 / nu)


def le6(y, a):
    return np.column_stack((t_logpdf(y, a[2], a[4]), t_logpdf(y, a[3], a[5])))


def le9(y, a):
    eps, k, theta = a[6], a[7], a[8]
    if eps <= 0.0:
        return le6(y, a)
    weights = np.zeros(len(GRID))
    g = GRID_G
    weights[GRID_NEG] = (
        gammainc(k, (g + GRID_STEP / 2) / theta)
        - gammainc(k, np.maximum(g - GRID_STEP / 2, 0.0) / theta)
    )
    out = np.empty((len(y), 2))
    for j in range(2):
        direct = t_logpdf(y, a[j + 2], a[j + 4])
        f = np.exp(t_logpdf(GRID, a[j + 2], a[j + 4]))
        conv = fftconvolve(f, weights, mode="same")
        mixture = np.maximum(np.interp(y, GRID, conv), 1e-300)
        out[:, j] = np.logaddexp(np.log1p(-eps) + direct, np.log(eps) + np.log(mixture))
    return out


def decode(x):
    a = np.asarray(x, dtype=float).copy()
    a[2:6] = np.exp(a[2:6])
    if len(a) > 6:
        a[7:9] = np.exp(a[7:9])
    return a


def encode(a):
    x = np.asarray(a, dtype=float).copy()
    x[2:6] = np.log(np.maximum(a[2:6], 1e-8))
    if len(a) > 6:
        x[7:9] = np.log(np.maximum(a[7:9], 1e-8))
    return x


def reorder(a):
    a = np.asarray(a, dtype=float).copy()
    if a[2] < a[3]:
        a[0], a[1] = a[1], a[0]
        a[2], a[3] = a[3], a[2]
        a[4], a[5] = a[5], a[4]
    return a


def bounds6(y):
    sd = float(np.std(y))
    s_lo = max(0.15, sd * 0.17)
    s_hi = max(2.0, sd * 4.0)
    nu_lo = (3 - 1.62) / 0.62
    nu_hi = (3 - 1.03) / 0.03
    return [
        (0.65, 0.998), (0.65, 0.998),
        (np.log(nu_lo), np.log(nu_hi)),
        (np.log(nu_lo), np.log(nu_hi)),
        (np.log(s_lo), np.log(s_hi)),
        (np.log(s_lo), np.log(s_hi)),
    ]


def objective(x, y, use_jump):
    a = decode(x)
    emission = le9(y, a) if use_jump else le6(y, a)
    ll, _, _ = forward_filter(emission, float(a[0]), float(a[1]))
    return -ll


def fit6(y):
    sd = float(np.std(y))
    starts = [
        np.array([0.965, 0.955, 18.0, 3.8, max(0.3, 0.65 * sd), max(0.6, 1.3 * sd)]),
        np.array([0.970, 0.940, 7.0, 2.8, max(0.3, 0.70 * sd), max(0.6, 1.2 * sd)]),
    ]
    results = []
    for s in starts:
        r = minimize(objective, encode(s), args=(y, False), method="L-BFGS-B",
                     bounds=bounds6(y), options={"maxiter": 160, "ftol": 1e-9, "gtol": 1e-5})
        results.append(r)
    best = min(results, key=lambda z: z.fun)
    a = reorder(decode(best.x))
    ll = -objective(encode(a), y, False)
    return a, ll, bool(best.success)


def fit9(y, a6_init):
    starts = []
    for ep, k, th in ((0.03, 2.0, 3.0), (0.05, 1.5, 3.5), (0.02, 2.5, 2.5)):
        starts.append(np.hstack((a6_init.copy(), [ep, k, th])))
    bounds = bounds6(y) + [(1e-5, 0.15), (np.log(0.55), np.log(7.0)), (np.log(0.2), np.log(8.0))]
    results = []
    for s in starts:
        r = minimize(objective, encode(s), args=(y, True), method="L-BFGS-B",
                     bounds=bounds, options={"maxiter": 140, "ftol": 1e-8, "gtol": 2e-5})
        results.append(r)
    best = min(results, key=lambda z: z.fun)
    a = reorder(decode(best.x))
    ll = -objective(encode(a), y, True)
    return a, ll, bool(best.success)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(scenario: str, n: int, rep: int):
    rng = np.random.default_rng(
        np.random.SeedSequence([SEED,
                                {"correct": 0, "gaussian_contamination": 1, "negative_gamma_jumps": 2}[scenario],
                                n, rep])
    )
    z = np.empty(n + NTEST, dtype=np.int8)
    y = np.empty(n + NTEST)
    jump = np.zeros(n + NTEST, dtype=np.int8)
    z[0] = int(rng.random() > 0.7272727272727273)
    for i in range(n + NTEST):
        if i:
            z[i] = int(rng.random() >= PTRUE[z[i - 1], 0])
        yi = rng.standard_t(NUTRUE[z[i]]) * STRUE[z[i]]
        if scenario == "gaussian_contamination" and rng.random() < 0.03:
            jump[i] = 1
            yi += rng.normal(0.0, 5.0)
        elif scenario == "negative_gamma_jumps" and rng.random() < 0.03:
            jump[i] = 1
            yi -= rng.gamma(shape=2.0, scale=3.0)
        y[i] = yi
    return y, z, jump


# ---------------------------------------------------------------------------
# Dynamic quantities + Tsallis entropy
# ---------------------------------------------------------------------------

def tsallis_entropy(p, q):
    p = np.clip(np.asarray(p, dtype=float), 1e-300, 1.0)
    if abs(q - 1.0) < 1e-12:
        return -float(np.sum(p * np.log(p)))
    return float((1.0 - np.sum(p ** q)) / (q - 1.0))


def scores_and_track(y, z, n, a, use_jump):
    le = le9(y, a) if use_jump else le6(y, a)
    _, posterior, scores = forward_filter(le, float(a[0]), float(a[1]))
    q_hat = (a[2:4] + 3.0) / (a[2:4] + 1.0)
    q_t = (1.0 - posterior[n:]) * q_hat[0] + posterior[n:] * q_hat[1]
    q_rmse = float(np.sqrt(np.mean((q_t - QTRUE[z[n:]]) ** 2)))
    acc = float(np.mean((posterior[n:] > 0.5) == z[n:]))
    dyn_ent = np.mean([tsallis_entropy([1 - p, p], qt) for p, qt in zip(posterior[n:], q_t)])
    return float(np.mean(scores[n:])), q_rmse, acc, float(dyn_ent)


# ---------------------------------------------------------------------------
# Inference module (Wald + parametric bootstrap)
# ---------------------------------------------------------------------------

def numerical_hessian(func, x, eps=1e-5):
    """Simple central-difference Hessian."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    H = np.zeros((n, n))
    f0 = func(x)
    for i in range(n):
        xp = x.copy(); xm = x.copy()
        xp[i] += eps; xm[i] -= eps
        H[i, i] = (func(xp) - 2 * f0 + func(xm)) / (eps ** 2)
        for j in range(i + 1, n):
            xpp = x.copy(); xmm = x.copy()
            xpm = x.copy(); xmp = x.copy()
            xpp[i] += eps; xpp[j] += eps
            xmm[i] -= eps; xmm[j] -= eps
            xpm[i] += eps; xpm[j] -= eps
            xmp[i] -= eps; xmp[j] += eps
            H[i, j] = H[j, i] = (func(xpp) - func(xpm) - func(xmp) + func(xmm)) / (4 * eps ** 2)
    return H


def wald_ci_6(y, a6, alpha=0.05):
    """Observed-information Wald 95% CI for the 6-parameter model (on original scale)."""
    def negll(x):
        return objective(x, y, False)

    x = encode(a6)
    try:
        H = numerical_hessian(negll, x)
        cov = inv(H)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
        zcrit = norm.ppf(1 - alpha / 2)

        # Transform back to original scale for reporting
        # parameters: p00, p11, nu0, nu1, s0, s1  (then convert nu → q)
        q0 = (a6[2] + 3) / (a6[2] + 1)
        q1 = (a6[3] + 3) / (a6[3] + 1)

        # Approximate SE for q via delta method (nu → q)
        dnu_dq = -2 / (q0 - 1)**2          # rough; we use simulation SE later
        # For simplicity we report CI on the estimated q using the numerical SE of nu
        # and a simple transformation
        se_q0 = se[2] * abs(-2 / (a6[2] + 1)**2)   # delta method
        se_q1 = se[3] * abs(-2 / (a6[3] + 1)**2)

        return {
            "q0_hat": q0, "q0_se": se_q0,
            "q0_ci_low": q0 - zcrit * se_q0, "q0_ci_high": q0 + zcrit * se_q0,
            "q1_hat": q1, "q1_se": se_q1,
            "q1_ci_low": q1 - zcrit * se_q1, "q1_ci_high": q1 + zcrit * se_q1,
            "hessian_ok": True
        }
    except Exception:
        return {"hessian_ok": False}


def parametric_bootstrap_9(y, a9, n_boot=200, alpha=0.05):
    """Simple parametric bootstrap for q0, q1, ε of the 9-parameter model.
    Uses the fitted model to generate bootstrap samples and re-estimates.
    Limited n_boot for speed; increase for final paper.
    """
    n = len(y)
    q0_list, q1_list, eps_list = [], [], []

    # Generate bootstrap paths from the fitted 9-parameter model
    rng = np.random.default_rng(42)
    p00, p11 = a9[0], a9[1]
    nu0, nu1 = a9[2], a9[3]
    s0, s1 = a9[4], a9[5]
    eps, k, theta = a9[6], a9[7], a9[8]

    for b in range(n_boot):
        # simulate one path under the fitted model
        z = np.empty(n, dtype=np.int8)
        yb = np.empty(n)
        z[0] = 0 if rng.random() < 0.5 else 1
        for t in range(n):
            if t:
                z[t] = 0 if rng.random() < (p00 if z[t-1] == 0 else (1 - p11)) else 1
            yi = rng.standard_t(nu0 if z[t] == 0 else nu1) * (s0 if z[t] == 0 else s1)
            if rng.random() < eps:
                yi -= rng.gamma(shape=k, scale=theta)
            yb[t] = yi

        try:
            a6b, _, _ = fit6(yb)
            a9b, _, ok = fit9(yb, a6b)
            if ok:
                q0b = (a9b[2] + 3) / (a9b[2] + 1)
                q1b = (a9b[3] + 3) / (a9b[3] + 1)
                q0_list.append(q0b)
                q1_list.append(q1b)
                eps_list.append(a9b[6])
        except Exception:
            continue

    def ci(arr):
        if len(arr) < 5:
            return np.nan, np.nan, np.nan
        arr = np.asarray(arr)
        return float(np.mean(arr)), float(np.quantile(arr, alpha/2)), float(np.quantile(arr, 1-alpha/2))

    q0_m, q0_lo, q0_hi = ci(q0_list)
    q1_m, q1_lo, q1_hi = ci(q1_list)
    eps_m, eps_lo, eps_hi = ci(eps_list)

    return {
        "boot_q0_mean": q0_m, "boot_q0_low": q0_lo, "boot_q0_high": q0_hi,
        "boot_q1_mean": q1_m, "boot_q1_low": q1_lo, "boot_q1_high": q1_hi,
        "boot_eps_mean": eps_m, "boot_eps_low": eps_lo, "boot_eps_high": eps_hi,
        "n_boot_success": len(q0_list)
    }


# ---------------------------------------------------------------------------
# Single replication (with optional inference)
# ---------------------------------------------------------------------------

def calculate(scenario: str, n: int, rep: int, do_inference: bool = False) -> Dict:
    data, z, jump = simulate(scenario, n, rep)

    a6, ll6, ok6 = fit6(data[:n])
    a9, ll9, ok9 = fit9(data[:n], a6)

    sc6, qrmse6, acc6, ent6 = scores_and_track(data, z, n, a6, False)
    sc9, qrmse9, acc9, ent9 = scores_and_track(data, z, n, a9, True)

    q6 = (a6[2:4] + 3.0) / (a6[2:4] + 1.0)
    q9 = (a9[2:4] + 3.0) / (a9[2:4] + 1.0)

    row = {
        "scenario": scenario, "n": n, "rep": rep,
        "q0_direct": q6[0], "q1_direct": q6[1],
        "q0_jump": q9[0], "q1_jump": q9[1],
        "q0_sqerr_direct": (q6[0] - QTRUE[0])**2,
        "q1_sqerr_direct": (q6[1] - QTRUE[1])**2,
        "q0_sqerr_jump": (q9[0] - QTRUE[0])**2,
        "q1_sqerr_jump": (q9[1] - QTRUE[1])**2,
        "epsilon": a9[6], "k": a9[7], "theta": a9[8],
        "logscore_direct": sc6, "logscore_jump": sc9,
        "logscore_gain": sc9 - sc6,
        "filter_rmse_direct": qrmse6, "filter_rmse_jump": qrmse9,
        "dyn_entropy_direct": ent6, "dyn_entropy_jump": ent9,
        "success_direct": ok6, "success_jump": ok9,
        "k_bound": (a9[7] < 0.56) or (a9[7] > 6.9),
    }

    if do_inference and ok6:
        wald = wald_ci_6(data[:n], a6)
        row.update({f"wald_{k}": v for k, v in wald.items()})

    if do_inference and ok9 and scenario == "negative_gamma_jumps":
        # Bootstrap only on the correctly specified generator (expensive)
        boot = parametric_bootstrap_9(data[:n], a9, n_boot=30)
        row.update(boot)

    return row


# ---------------------------------------------------------------------------
# Parallel runner
# ---------------------------------------------------------------------------

def run_one(scenario, n, start_rep, reps, out_dir, do_inference):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    out_file = out_dir / f"{scenario}_n{n}_reps{start_rep}-{start_rep+reps-1}.csv"

    rows = []
    if out_file.exists():
        rows = pd.read_csv(out_file).to_dict("records")
    completed = {int(r["rep"]) for r in rows}

    for rep in range(start_rep, start_rep + reps):
        if rep in completed:
            continue
        t0 = time.monotonic()
        try:
            row = calculate(scenario, n, rep, do_inference=do_inference)
            rows.append(row)
        except Exception as e:
            rows.append({"scenario": scenario, "n": n, "rep": rep, "error": str(e)})
        if (rep - start_rep + 1) % 5 == 0 or rep == start_rep + reps - 1:
            tmp = out_file.with_suffix(".tmp")
            pd.DataFrame(rows).to_csv(tmp, index=False)
            os.replace(tmp, out_file)
            print(f"{scenario} n={n} rep={rep} ({rep-start_rep+1}/{reps}) "
                  f"{time.monotonic()-t0:.1f}s", flush=True)
    return f"{scenario}:{n} → {len(rows)} records"


# ---------------------------------------------------------------------------
# Summary + coverage
# ---------------------------------------------------------------------------

def make_summaries(out_dir: Path):
    files = list(out_dir.glob("*_n*_reps*.csv"))
    if not files:
        print("No result files found.")
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "error" in df.columns:
        df = df[df["error"].isna()]

    print("\n========== RMSE of regime q ==========")
    for (sc, n), g in df.groupby(["scenario", "n"]):
        print(f"{sc:22s} n={n:4d}  q1 6/9 = {np.sqrt(g.q1_sqerr_direct.mean()):.4f}/"
              f"{np.sqrt(g.q1_sqerr_jump.mean()):.4f}   "
              f"q0 6/9 = {np.sqrt(g.q0_sqerr_direct.mean()):.4f}/"
              f"{np.sqrt(g.q0_sqerr_jump.mean()):.4f}  (N={len(g)})")

    print("\n========== Log-score gain & filtered-q ==========")
    for (sc, n), g in df.groupby(["scenario", "n"]):
        print(f"{sc:22s} n={n:4d}  Δlogscore = {g.logscore_gain.mean():+.5f}   "
              f"filter-RMSE 6/9 = {g.filter_rmse_direct.mean():.4f}/{g.filter_rmse_jump.mean():.4f}")

    # Wald coverage for HMM-6 (when available)
    if "wald_q1_ci_low" in df.columns:
        print("\n========== Wald 95% coverage (HMM-6) ==========")
        for (sc, n), g in df.groupby(["scenario", "n"]):
            g = g.dropna(subset=["wald_q1_ci_low"])
            if len(g) == 0:
                continue
            cov_q1 = np.mean((g.wald_q1_ci_low <= QTRUE[1]) & (g.wald_q1_ci_high >= QTRUE[1]))
            cov_q0 = np.mean((g.wald_q0_ci_low <= QTRUE[0]) & (g.wald_q0_ci_high >= QTRUE[0]))
            print(f"{sc:22s} n={n:4d}  q1 coverage = {100*cov_q1:.1f}%   q0 coverage = {100*cov_q0:.1f}%  (N={len(g)})")

    df.to_csv(out_dir / "all_replications_combined.csv", index=False)
    print(f"\nCombined file: {out_dir / 'all_replications_combined.csv'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Giorno-Tsallis complete pipeline")
    ap.add_argument("--start-rep", type=int, default=0)
    ap.add_argument("--reps", type=int, default=100, help="Number of replications")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", type=str, default="giorno_tsallis_results")
    ap.add_argument("--scenario", type=str, default="all",
                    choices=["all", "correct", "gaussian_contamination", "negative_gamma_jumps"])
    ap.add_argument("--n", type=str, default="all",
                    choices=["all", "1000", "2000", "3000"])
    ap.add_argument("--inference", action="store_true",
                    help="Compute Wald CIs (HMM-6) and bootstrap CIs (HMM-9)")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(exist_ok=True, parents=True)

    if args.summarize_only:
        make_summaries(out_dir)
        return

    scenarios = (["correct", "gaussian_contamination", "negative_gamma_jumps"]
                 if args.scenario == "all" else [args.scenario])
    sizes = [1000, 2000, 3000] if args.n == "all" else [int(args.n)]

    configs = [(sc, n, args.start_rep, args.reps, str(out_dir), args.inference)
               for sc in scenarios for n in sizes]

    print(f"Running {len(configs)} configurations | reps {args.start_rep}..{args.start_rep+args.reps-1} "
          f"| inference={args.inference} | workers={args.workers}")

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(run_one, *cfg) for cfg in configs]
        for fut in concurrent.futures.as_completed(futs):
            print("Finished:", fut.result(), flush=True)

    make_summaries(out_dir)
    print("DONE")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()

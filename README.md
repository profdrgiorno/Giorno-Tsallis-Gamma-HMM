# Giorno–Tsallis | Paper 1.2

### Dynamic Entropic Tail-State Estimation under Asymmetric Gamma Jumps

**Companion research-code and simulation-data repository** for the manuscript by **J. Giorno, PhD**, *Dynamic Entropic Tail-State Estimation under Asymmetric Gamma Jumps* (working manuscript, 15 October 2026; intended for the *Journal of Econometrics*, **not represented as accepted or published**).

[**Read the current paper (PDF)**](paper/Giorno_Tsallis_Paper_1_2_Latest_2026-10-15.pdf) · [Methods and reproducibility](reproducibility/REPRODUCIBILITY.md) · [Original source README](docs/README_original_2026-10-13.md)

> **Research attribution:** J. Giorno directs the research program and manuscript. OpenAI ChatGPT assisted with research discussion, coding, analysis and editing. The project name does not imply participation or endorsement by C. Tsallis.

## Research question

Can a two-state q-Gaussian (Student-t) hidden Markov model with a distinct negative Gamma-jump mechanism distinguish heavy-tail regime changes from rare asymmetric disturbances? This repository preserves the simulations comparing **HMM-6**, a **shared-location HMM-7**, and the **reflected-Gamma-jump HMM-9**. It does **not** claim universal HMM-9 superiority, global nine-parameter identification, or validated full-parameter confidence-interval coverage.

## Models

| Model | Parameters | Observation mechanism |
|---|---|---|
| HMM-6 | 6 | Two centered Student-t/q-Gaussian regime densities |
| HMM-7 | 7 | HMM-6 plus a common fitted location μ |
| HMM-9 | 9 | HMM-6 plus independent negative Gamma jumps with probability ε, shape k, scale θ |

For state j, `ν_j=(3−q_j)/(q_j−1)`, with `1<q_j<5/3`. HMM-9 models `r_t=s_{K_t} T_{ν_{K_t},t} − B_t G_t`, where `B_t~Bernoulli(ε)` and `G_t~Gamma(k, scale θ)`; its emission density is `(1−ε)f_j(r)+ε∫_0^∞ f_j(r+z) Gamma(z;k,θ) dz`. Filtered `q` is a posterior-weighted *tail index*, **not** a directly measured Tsallis entropy functional.

## What is included

| Component | Location | Status |
|---|---|---|
| Latest supplied manuscript (21 pages) | `paper/` | Supplied by author; replaces 13 October manuscript |
| Original 2,700-path HMM-6/HMM-9 matched archive | `data/raw/all_replications_combined.csv` | Included; nine cells × 300 |
| HMM-7 paired location comparison | `data/raw/hmm7_location_replications.csv` | Included; nine cells × 300 |
| HMM-9 Wald inference experiment | `data/raw/hmm9_wald_inference_replications.csv` | Included; separate 1,000 paths |
| Original simulation/benchmark source | `src/` | Included unchanged from 13 October archive |
| Summary CSVs and paper figure | `data/summary/`, `results/figures/` | Included |
| Input integrity manifest, audit and checks | `reproducibility/`, `tests/` | Included |

**The 2,700 matched paths and the 1,000 inference paths are different experiments. Do not add them to claim 3,700 matched forecasting replications.** The historical HMM-6 Wald experiment described in the manuscript is not necessarily represented by complete raw data here. The older six-cell 500-replication study is a separate experiment and must not be pooled into this archive.

## Monte Carlo design

| Data-generating mechanism | Train n | Independent paths per cell | Chronological test observations |
|---|---|---:|---:|
| Clean Student-t | 1,000 / 2,000 / 3,000 | 300 | 400 |
| Gaussian contamination | 1,000 / 2,000 / 3,000 | 300 | 400 |
| Negative Gamma jumps | 1,000 / 2,000 / 3,000 | 300 | 400 |

Shared truth: `p00=.985`, `p11=.960`, `q0=1.12`, `q1=1.42`, `s0=.8`, `s1=1.25`. Contamination uses `N(0,5²)` at probability `.03`; negative Gamma shocks use `Gamma(k=2,θ=3)` at probability `.03`.

## Selected paper results (as reported in the manuscript)

At `n=3,000` under correctly specified negative Gamma jumps, q1 RMSE is **0.1978 (HMM-6)**, **0.1976 (HMM-7)**, and **0.0663 (HMM-9)**. Mean held-out log-score improvement for HMM-9 relative to HMM-7 is **+0.01830** per observation, with a paired Monte Carlo 95% interval **[0.01712, 0.01948]**, and HMM-9 wins **291/300** paths. Under clean observations the simpler models generally recover q1 more precisely. These are simulation-specific comparisons, **not structural parameter confidence intervals**.

The separate HMM-9 Wald study has 500 paths at each `n=1,000` and `n=3,000`; only **287/500** and **428/500** were eligible for the stated Hessian-based analysis. Conditional joint nine-dimensional coverage was **83.3%** and **89.7%**, below nominal 95%. Thus the paper **does not establish routine nine-parameter Wald validity**.

## Quick start (Windows / macOS / Linux)

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python reproducibility/audit_archive.py
```

To run the original simulation implementation on a **small new pilot** (not to overwrite the archived data):

```bash
python giorno_tsallis_complete_grok.py --scenario negative_gamma_jumps --n 1000 --reps 2 --workers 1 --out-dir scratch/pilot
```

To recompute the HMM-7 benchmark on the archived HMM-6/HMM-9 paths, the original scripts are also copied to the repository root because the benchmark imports the main module by name:

```bash
python gt_hmm7_location_benchmark.py --moon-csv data/raw/all_replications_combined.csv --scenario negative_gamma_jumps --n 1000 --reps 2 --quick-pilot --out-dir scratch/location_smoke
```

The `--quick-pilot` option changes optimization effort: **never pool smoke-test output with the published results**. A complete independent rerun is computationally expensive and has not been performed during packaging.

## Research boundaries

- HMM-9 uses FFT-discretized convolution; numerical approximation and optimizer flags require sensitivity checks.
- A common-location HMM-7 is a useful robustness benchmark, not an exhaustive comparison against all skewed distributions.
- Theoretical model-family non-equivalence is **not** full nine-parameter global identification.
- Intervals for Monte Carlo mean score differences are not parameter-level confidence intervals.
- Boundary `ε=0` leaves jump shape and scale unidentified.
- Physical applications, dynamic Gamma shape and economic welfare/control remain future work or separate papers.

## Citation

Use [CITATION.cff](CITATION.cff). The attached paper is a working manuscript; update the citation after publication. The code and data are research artifacts and have not been independently rerun end-to-end in this packaging pass.

## License and data policy

See [LICENSE](LICENSE). This repository includes original research code and simulation results. The manuscript remains under the author's copyright and is not automatically covered by the code license.

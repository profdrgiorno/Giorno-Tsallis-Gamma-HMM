# Giorno–Tsallis Paper 1.2: Focused Location-Adjusted HMM Benchmark

Date: 13 October 2026. Author: J. Giorno, PhD. AI-assisted numerical implementation and manuscript editing.

## Scientific question

Does HMM-9 outperform a symmetric q-Gaussian HMM simply because it accommodates a nonzero global location shift, or because it can model the additional one-sided disturbance?

Models:
- HMM-6: centered Student-t HMM with p00, p11, nu0, nu1, s0, s1.
- HMM-7: HMM-6 plus a single shared location mu. The emission densities are f_j(y-mu). Neither conditional state gets its own mean.
- HMM-9: HMM-6 with independent reflected-Gamma jumps, including epsilon, k and theta.
- True-generator oracle: parameters and emissions known from simulation. This is not a fitted comparison model.

## Empirical design

The `all_replications_combined(2).csv` input contains 2,700 matched MOON replications (300 independent simulation paths per cell) across clean Student-t, additive Gaussian contamination, and negative Gamma jumps at training n=1000,2000,3000. The 400 held-out observations per path are evaluated chronologically. Models share the simulated path and start the filter at the same 0.5/0.5 prior. HMM-7 is fitted on only the training segment; all test scores use the sequential HMM predictive likelihood. HMM-6 and HMM-9 metrics are from the original MOON archive; they are not counted as newly independent simulations.

The true negative-Gamma generator uses epsilon=.03, k=2, theta=3; its unconditional extra shock mean is -epsilon*k*theta = -0.18. The conditional Student-t components retain symmetric, centered shapes. HMM-7 is estimated via scipy L-BFGS-B, 3 starts, 160 iteration maximum and a predeclared symmetric location bound +/- max(1, min(4,2*sd_training)). No hidden state or jump labels are supplied to optimization. All 2,700 HMM-7 fits reported successful optimizer flags, but the flag alone does not establish a global maximum.

The Gaussian-contamination oracle uses a properly normalized FFT convolution of Student-t and Normal(0,5^2) densities. It was compared against direct adaptive numerical integration at representative observations, with log-density differences on the order of 1e-14. This oracle is only a known-parameter benchmark, not a substitute for confidence interval validation.

## Main finding

At n=3000 with true negative Gamma jumps, HMM-7 q1 RMSE = 0.197580 versus HMM-9 = 0.066284, and the average paired HMM-9 minus HMM-7 held-out log score is +0.018300 (Monte Carlo 95% interval approximately [0.017123,0.019477]); HMM-9 wins 291 of 300 replications. HMM-7 location mean is approximately -0.020667 across paths, not the true unconditional shift -0.18. The location adjustment does not explain the HMM-9 gain in this design.

The source data also show that HMM-9 loses precision for clean Student-t q1 relative to the simpler model. Under Gaussian contamination the Gamma extension is misspecified even where it produces modest positive predictive gains.

## Theoretical scope

Proposition 3 in the manuscript proves that a nondegenerate independent one-sided reflected-Gamma shock added to symmetric Student-t regime mixtures cannot have a marginal law equal to a Student-t HMM with only a common location shift. The proof uses positivity of the Student-t (Gaussian-scale-mixture) characteristic function and the jump variable's one-sided bounded support and negative unbounded tail. This is MODEL-FAMILY NON-EQUIVALENCE, not global nine-parameter identification or a proof of regular Wald coverage.

## Reproduce on Windows

The following three files must be in the same directory:

1. `giorno_tsallis_complete_grok.py` (original simulation/HMM source)
2. `all_replications_combined(2).csv` (original 2,700-record MOON archive)
3. `gt_hmm7_location_benchmark.py` (new HMM-7 fitting and benchmark)

Recommended Python packages: numpy, pandas, scipy, numba (see requirements.txt).

```
python -u gt_hmm7_location_benchmark.py --moon-csv "all_replications_combined(2).csv" --scenario all --n all --reps 300 --out-dir results_location
```

The program checkpoints every fitted replication to a CSV. Repeating it with the same output directory skips already completed (scenario,n,rep) records. A single configuration, e.g. `--scenario negative_gamma_jumps --n 3000`, can be run separately. For a smoke test use `--reps 5 --quick-pilot --out-dir results_smoke` (the one-start smoke test must NOT be pooled into final results).

Outputs: `all_location_benchmark_replications.csv` and `location_benchmark_summary.csv`. They were exported under the filenames `GT_Location_Complete_2700_Replications.csv` and `GT_Location_Complete_2700_Summary.csv` for the paper.

## Evidence boundaries

- All simulation-path intervals are paired Monte Carlo intervals for mean differences; they are not structural parameter confidence intervals.
- The HMM-9 Wald diagnostics (separate 500 paths each at n=1000,3000) are supplied for auditing; their near-boundary and joint coverage limitations remain.
- No HMM-9-to-optimal-control causal welfare gain has been validated in this package; companion Paper 1.1 addresses optimal decisions.
- Dynamic Gamma shape k_t and unknown structural breaks are explicit future extensions.
- Real-data validation, full nine-parameter identification, and boundary-valid inference remain outside the proven contributions.

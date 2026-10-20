# Reproducibility and validation

1. Install `requirements.txt`; execute `python -m pytest -q` and `python reproducibility/audit_archive.py`.
2. Raw archives are immutable research inputs; do not overwrite them with a new simulation.
3. Run new Monte Carlo into `scratch/` with explicit scenario, n, start-rep, number of reps, and worker count.
4. HMM-7 script imports `giorno_tsallis_complete_grok.py` by module name; the two original scripts are copied into repository root for compatible execution.
5. The original HMM-7 benchmark was reported with three L-BFGS-B starts. `--quick-pilot` uses a reduced procedure and must not be combined with final runs.
6. Compare results using paired `(scenario,n,rep)` keys, preserve chronological 400-observation test windows, and keep the separate Wald-inference paths separate.
7. Check optimizer success, boundary frequencies, FFT/quadrature differences, missing and nonfinite fields before statistical interpretation.
8. Keep version, source code, dependencies, raw outputs, and SHA256 hashes together for an independently rerunnable release.

**Not verified in this packaging step:** numerical rerun of 2,700 fits, convergence of every optimization, source code's exact seed equivalence with each archived row, and all manuscript statistics. Archive checks test presence, uniqueness, numeric sanity and selected cross-file matches, not complete end-to-end reproduction.

**Paper correspondence:** manuscript supplied 20 October 2026, internally marked working draft 15 October 2026, 21 pages. Manuscript Table 1 (nine-cell design), Tables 2–4 (HMM-6/HMM-9 performance), Table 6 (Wald), and Table 7 (HMM-7) are the relevant comparisons.

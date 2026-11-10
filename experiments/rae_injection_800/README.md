# 800-case RAE/Feast benchmark and 32-configuration ablation

The benchmark generates synthetic cases from a fixed seed and does not read CPI target data. The generator, RAE implementation, Feast comparison, and 2^5 module-gated ablation are included. Each named ablation disables the corresponding validation computation.

## Environment

Python 3.11 is recommended. Install `requirements/rae_injection.txt` from the repository root.

## Run in a clean workspace

From the repository root, move into this experiment folder. The benchmark writes generated `cases/` and outputs below that folder; the archived outputs in `reference_results/` are separate. Then run:

```bash
cd experiments/rae_injection_800
python run_feast_comparison.py
python run_computational_ablation.py --root . --out results/computational_ablation_rerun
```

For a short smoke run, set `RAE_CASE_LIMIT=10` for the first command. The complete ablation requires all 800 generated cases.

The reference report includes the benchmark comparison scope: Feast is a point-in-time feature retrieval baseline, not a complete target, key, or score auditor. The 32-arm run uses 800 frozen synthetic cases and reports configuration summaries, module-specific detection, factorial effects, interactions, and runtime checks.

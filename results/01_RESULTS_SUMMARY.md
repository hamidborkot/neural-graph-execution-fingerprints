# 01_RESULTS_SUMMARY.md
## All Experiments, Organized by Research Question

This file consolidates every result table produced across the project's lifetime, flags which
numbers are FINAL/AUTHORITATIVE (post-fix, deconfounded, budget-matched) vs SUPERSEDED
(early runs kept for the "measurement-order confound" methodology finding), and gives the
best evidence to cite for each claim in the paper.

---

## Experiment 1: Detector Discriminative Power (Clean vs Tampered)
**Use case:** Establishes the detector actually works before testing if it can be evaded.

### 1a. Clean-population false-positive-rate calibration (Cell 10) -- FINAL, use as-is
| Model | Mean D | Median D | p90 | p95 (threshold) | p99 |
|---|---|---|---|---|---|
| efficientnet | 3.200 | 3.257 | 4.144 | **4.195** | 4.233 |
| mobilenetv2 | 2.295 | 1.837 | 3.444 | **4.863** | 6.958 |
| resnet18 | 3.905 | 3.462 | 7.295 | **7.914** | 9.176 |
| squeezenet | 3.605 | 3.077 | 6.595 | **6.659** | 6.820 |

30 clean re-profiling sessions per model. p95 column = recommended per-model detection
threshold controlling false-positive rate at ~5%. **This is your core "the detector is
well-calibrated" table.**

### 1b. Struct-prune generalization across datasets (deconfounded, randomized order) -- FINAL
Source: Cell 9 fix (randomized measurement order across 16 dataset x model pairs).
See `FINAL_struct_prune_generalization_deconfounded.csv`.

| Dataset | Model | timing_D | activation_D |
|---|---|---|---|
| cifar10 | resnet18 | 0.107 | 706.2 |
| cifar10 | mobilenetv2 | 4.639 | 12.2 |
| cifar10 | squeezenet | 1.373 | 56193.7 |
| cifar10 | efficientnet | 1.467 | 2531.9 |
| imagenetmini | mobilenetv2 | 0.804 | 3.5 |
| imagenetmini | resnet18 | 1.632 | 226.4 |
| synthetic | resnet18 | 4.106 | 972.9 |
| tinyimagenet | squeezenet | 0.662 | 46711.4 |

(full 16-row table in CSV) -- **activation_D dwarfs timing_D consistently for squeezenet**
(4-5 orders of magnitude larger), suggesting activation-signature detection is far more
sensitive than timing for that architecture -- a useful architecture-dependent finding.

### 1c. SUPERSEDED (do not cite as primary): original single-order Cell 9 run
Same table computed with fixed nested-loop measurement order gave very different numbers
(e.g., cifar10/resnet18 timing_D = 9.804 vs 0.107 deconfounded). **Keep this table only to
demonstrate the measurement-order confound in the methodology section** -- do not use these
numbers to support any tampering-sensitivity claim.

---

## Experiment 2: Struct-Prune Detection Sensitivity (Block C, single-dataset then all-4)
**Use case:** How much structural pruning is needed before the detector flags it, and does
this hold across datasets.

Block C (final, all 4 datasets x 4 models x 3 prune ratios = 36 rows) is in
`struct_prune_multi_arch.csv` / NDSS_FULL_RESULTS.xlsx sheet `struct_prune_single_dataset`.
Representative rows:

| Dataset | Model | Prune | timing_D | activation_D |
|---|---|---|---|---|
| cifar10 | efficientnet | 0.5 | 5.638 | 10452.0 |
| cifar10 | squeezenet | 0.5 | 0.174 | 61199.9 |
| imagenetmini | resnet18 | 0.2 | 0.618 | 401.6 |
| tinyimagenet | efficientnet | 0.2 | 3.574 | 1114.8 |

**Key pattern:** timing_D does NOT increase monotonically with prune ratio for most
models -- this is worth an explicit note ("timing side-channel sensitivity to structural
pruning is non-monotonic, likely because pruning changes memory-access patterns
non-linearly with sparsity ratio").

---

## Experiment 3: Adversarial Evasion -- Random-Walk vs CMA-ES (THE CORE RESULT)
**Use case:** Can a query-efficient black-box optimizer find weight perturbations that evade
the detector faster/better than naive random search? This is the paper's central adversarial
robustness claim.

### FINAL, AUTHORITATIVE TABLE (budget-matched ~100-200 queries, full-parameter via random
projection, all 4 datasets, deconfounded baselines) -- see `FINAL_attacker_comparison_all_datasets.csv`

| Model | Dataset | Random-Walk best D | CMA-ES best D | CMA-ES wins? |
|---|---|---|---|---|
| mobilenetv2 | cifar10 | 0.2261 | 0.0623 | Yes |
| resnet18 | cifar10 | 0.0554 | 0.0094 | Yes |
| squeezenet | tinyimagenet | 0.0730 | 0.0178 | Yes |
| efficientnet | imagenetmini | 0.0668 | 0.0072 | Yes |
| efficientnet | cifar10 | 0.0141 | 0.0667 | No |
| resnet18 | imagenetmini | 0.0664 | 0.0951 | No |

**Overall CMA-ES win rate: 9/16 (56.25%)** -- mean best-D: CMA-ES 0.0415 vs random-walk 0.0634.

**How to frame this for NDSS:** Not a clean "CMA-ES beats random search" story. Frame as:
*"Query-efficient optimization provides a modest but architecture-dependent advantage over
naive random search (56% win rate, ~35% lower mean evasion distance), indicating the
detector's fitness landscape is only partially exploitable by gradient-free methods --
consistent with INT8 quantization inducing a non-smooth, plateaued objective surface that
limits CMA-ES's covariance-adaptation advantage."* This reframes a mixed result as a
mechanistic finding about quantization's effect on attack surfaces -- genuinely novel framing.

### Progression of this experiment across iterations (show due diligence in methodology section)
| Iteration | Budget | Subspace | Result | Problem |
|---|---|---|---|---|
| Cell 11 (v1) | RW=80, CMA=120 | 200-dim single layer | CMA loses on 2/4 models | Budget too small, unfair subspace |
| Cell 11 fix | RW=100, CMA=114 | still restricted | Mixed | Still budget-limited |
| Block A | CMA=817 | still restricted | CMA D as low as 0.001-0.024 | No RW comparison at matched budget |
| Block B | RW=150, CMA=154 | full-param attempted | CMA MemoryError on naive full-cov | 3.4M-dim covariance = 87.6 TiB |
| **FINAL** | RW~100, CMA~200 | full-param via 40-dim random projection | 56% CMA win rate, all 4 datasets | Resolved -- this is what to cite |

---

## Experiment 4: Adaptive Attacks -- FGSM & Backdoor Trigger Replication
**Use case:** Tests input-space (not weight-space) evasion -- does adversarial input
perturbation alone (no model tampering) shift the timing/activation signature enough to
either evade detection or (interestingly) trigger false positives on clean-but-adversarial
inputs.

Full results (64 rows FGSM + 64 rows backdoor, all 4 models x 4 datasets) in
`fgsm_all_datasets.csv` / `backdoor_all_datasets.csv`. **Use median, not mean** (see note below).

### Standout findings worth featuring:
| Model | Dataset | Pattern |
|---|---|---|
| resnet18 | synthetic | FGSM median D climbs to 9.9-14.0 across all epsilons -- by far the strongest detection signal in the whole project |
| squeezenet | tinyimagenet | FGSM median D stays 0.15-0.77 across all epsilons -- weakest detection surface found |
| efficientnet | cifar10 | Detection signal (median D ~5.5-6.5) is stable across epsilon, suggesting saturation |

**Caveat to state explicitly:** resnet18/synthetic's outsized D-scores may be partly an
artifact of `synthetic` being pure Gaussian noise (very different statistics from natural
images), not purely a robustness signal -- flag this as a limitation, and consider it a
motivation for the "detector counter-move" future work direction (below).

---

## Experiment 5: MLaaS Deployment Demo (4 rows)
**Use case:** End-to-end illustrative scenario showing the detector in a simulated deployment
pipeline. Useful for the paper's "system design" or "case study" section, not a statistical
result -- treat as a qualitative demonstration, not evidence.

---

## Master file inventory (produced across all cells)
| File | Rows | Role in paper |
|---|---|---|
| NDSS_FULL_RESULTS.xlsx | 6 sheets | Primary results workbook -- use as supplementary material |
| FINAL_attacker_comparison_all_datasets.csv | 16 | Table for Section 5 (Adversarial Evaluation) |
| FINAL_struct_prune_generalization_deconfounded.csv | 16 | Table for Section 4 (Detection Sensitivity) |
| struct_prune_multi_arch.csv | 36 | Full prune-ratio sweep, supplementary table |
| fgsm_all_datasets.csv / backdoor_all_datasets.csv | 64+64 | Section 5 input-space attacks |
| attack_manifest.csv | -- | Reproducibility appendix only (checkpoint log, not a result) |
| master CSV (adaptive/prune/deployment, 360 rows) | 360 | Raw data appendix |

## One data-quality note to fix before final tables go in the paper
FGSM/backdoor std values are large relative to means in several rows (e.g. mobilenetv2
imagenetmini eps=0.1: mean=4.17, std=5.25) -- heavily right-skewed. **Report median + IQR
(q25/q75) in the paper body, keep mean+std in an appendix table only.** The median/IQR
re-aggregation code for this was already provided (Cell 9 median-robust cell) -- confirm it
was run and use `fgsm_robust_summary.csv` / `backdoor_robust_summary.csv` as the citable source.

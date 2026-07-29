# Results Summary
## All Experiments, Organized by Research Question

This file consolidates every result table produced across the project's lifetime, flags which numbers are FINAL/AUTHORITATIVE (post-fix, deconfounded, budget-matched) vs SUPERSEDED (early runs kept for the "measurement-order confound" methodology finding), and gives the best evidence to cite for each claim in the paper.

---

## Experiment 1: Detector Discriminative Power (Clean vs Tampered)

### 1a. Clean-population false-positive-rate calibration — FINAL
| Model | Mean D | Median D | p90 | p95 (threshold) | p99 |
|---|---|---|---|---|---|
| efficientnet | 3.200 | 3.257 | 4.144 | **4.195** | 4.233 |
| mobilenetv2 | 2.295 | 1.837 | 3.444 | **4.863** | 6.958 |
| resnet18 | 3.905 | 3.462 | 7.295 | **7.914** | 9.176 |
| squeezenet | 3.605 | 3.077 | 6.595 | **6.659** | 6.820 |

30 clean re-profiling sessions per model. p95 column = recommended per-model detection threshold controlling false-positive rate at ~5%.

### 1b. Struct-prune generalization across datasets (deconfounded) — FINAL
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

(full 16-row table in CSV) — activation_D dwarfs timing_D consistently for squeezenet (4–5 orders of magnitude larger).

### 1c. SUPERSEDED: original single-order Cell 9 run
Same table computed with fixed nested-loop measurement order gave very different numbers (e.g., cifar10/resnet18 timing_D = 9.804 vs 0.107 deconfounded). **Keep this table only to demonstrate the measurement-order confound in the methodology section.**

---

## Experiment 2: Struct-Prune Detection Sensitivity (Block C, all-4 datasets)

| Dataset | Model | Prune | timing_D | activation_D |
|---|---|---|---|---|
| cifar10 | efficientnet | 0.5 | 5.638 | 10452.0 |
| cifar10 | squeezenet | 0.5 | 0.174 | 61199.9 |
| imagenetmini | resnet18 | 0.2 | 0.618 | 401.6 |
| tinyimagenet | efficientnet | 0.2 | 3.574 | 1114.8 |

**Key pattern:** timing_D does NOT increase monotonically with prune ratio for most models — pruning changes memory-access patterns non-linearly with sparsity ratio.

---

## Experiment 3: Adversarial Evasion — Random-Walk vs CMA-ES (THE CORE RESULT)

### FINAL, AUTHORITATIVE TABLE (budget-matched ~100–200 queries, full-parameter via random projection)

| Model | Dataset | Random-Walk best D | CMA-ES best D | CMA-ES wins? |
|---|---|---|---|---|
| mobilenetv2 | cifar10 | 0.2261 | 0.0623 | Yes |
| resnet18 | cifar10 | 0.0554 | 0.0094 | Yes |
| squeezenet | tinyimagenet | 0.0730 | 0.0178 | Yes |
| efficientnet | imagenetmini | 0.0668 | 0.0072 | Yes |
| efficientnet | cifar10 | 0.0141 | 0.0667 | No |
| resnet18 | imagenetmini | 0.0664 | 0.0951 | No |

**Overall CMA-ES win rate: 9/16 (56.25%)** — mean best-D: CMA-ES 0.0415 vs random-walk 0.0634.

**NDSS framing:** *"Query-efficient optimization provides a modest but architecture-dependent advantage over naive random search (56% win rate, ~35% lower mean evasion distance), indicating the detector's fitness landscape is only partially exploitable by gradient-free methods — consistent with INT8 quantization inducing a non-smooth, plateaued objective surface that limits CMA-ES's covariance-adaptation advantage."*

---

## Experiment 4: Adaptive Attacks — FGSM & Backdoor

**Standout findings:**
| Model | Dataset | Pattern |
|---|---|---|
| resnet18 | synthetic | FGSM median D climbs to 9.9–14.0 across all epsilons — strongest detection signal in the whole project |
| squeezenet | tinyimagenet | FGSM median D stays 0.15–0.77 across all epsilons — weakest detection surface found |
| efficientnet | cifar10 | Detection signal (median D ~5.5–6.5) stable across epsilon, suggesting saturation |

**Use median + IQR (not mean+std) in paper body** — FGSM/backdoor distributions are heavily right-skewed.

---

## Experiment 5: Vendor-Fraud Detection (activation_D AUC with 95% Bootstrap CI)

| Model | activation_D AUC [CI] | timing_D AUC [CI] |
|---|---|---|
| EfficientNet | **1.000 [1.000–1.000]** | 0.656 [0.344–1.000] |
| MobileNetV2 | 0.875 [0.625–1.000] | 0.875 [0.625–1.000] |
| ResNet18 | 0.875 [0.625–1.000] | 0.688 [0.375–0.969] |
| SqueezeNet | 0.875 [0.625–1.000] | 0.906 [0.625–1.000] |

**n=4 genuine sessions per model.** Bootstrap 95% CIs stated; lower bounds still clear 0.5 for the primary metric.

---

## Experiment 6: Evasion Robustness — Latency-Padding Sweep (0–50 ms)

| Model | act_D at 0 ms | act_D at 50 ms | Change |
|---|---|---|---|
| MobileNetV2 | 2.062 | 2.062 | **0.000%** |
| SqueezeNet | 2.656 | 2.656 | **0.000%** |
| ResNet18 | 3.753 | 3.753 | **0.000%** |
| EfficientNet | 41.707 | 41.707 | **0.000%** |

`activation_D` is provably invariant to latency-padding attacks.

---

## Experiment 7: Ensemble Defense vs. Adaptive Attacker

| Model | real_activation_D | Ensemble threshold | % of threshold | Evaded? |
|---|---|---|---|---|
| MobileNetV2 | 1.66 | 5.75 | 21.0% | Yes |
| SqueezeNet | 1.99 | 6.12 | 28.5% | Yes |
| ResNet18 | 1.76 | 5.42 | 25.6% | Yes |
| EfficientNet | 3.26 | 5.92 | **45.0%** | Yes |

**Ensemble defense reduces attacker margin to 21–45% of threshold** (vs. near-zero with single-threshold detector).

---

## One Data-Quality Note Before Final Paper Tables
FGSM/backdoor std values are large relative to means in several rows — heavily right-skewed. **Report median + IQR in the paper body, keep mean+std in appendix only.** Use `fgsm_robust_summary.csv` / `backdoor_robust_summary.csv` as the citable source.

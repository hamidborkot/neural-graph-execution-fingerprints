# Week 3 — Complete Experiment Log & Results Archive
**Project:** Neural Graph Execution Fingerprints — Multi-Channel MLaaS Tamper Detection  
**Purpose:** Full, unabridged inventory of every experiment, cell, and result table produced during Week 3, organized in execution order. Superseded/early-iteration runs are kept and explicitly labeled (not deleted) per the project's own methodology notes on the measurement-order confound.

---

## Cell 4 — Baseline Self-Consistency Validation
All four architectures passed self-consistency validation: timing D-scores between 0.30 and 3.70 (below the 5.0 threshold), and activation D = 0.0000 exactly (deterministic given fixed weights — expected, not a bug). Ridge-regularized covariance fix confirmed working.

---

## Cell 5 — Struct-Prune Results (Single Dataset, Interpretable Pass)
| Model | Prune 0.1 timing_D | Prune 0.5 timing_D | Activation_D range |
|---|---|---|---|
| mobilenetv2 | 1.32 ± 1.07 | 1.82 ± 0.45 | 2.0–7.9 |
| squeezenet | 2.17 ± 1.62 | 5.24 ± 2.41 | 18,507–30,613 |
| resnet18 | 2.45 ± 1.45 | 0.81 ± 0.07 | 229–401 |
| efficientnet | 3.70 ± 2.13 | 0.45 ± 0.34 | 270–7,895 |

SqueezeNet's activation channel is dramatically more sensitive to pruning than MobileNetV2's — a cross-architecture finding.

---

## Cell 6 — Adaptive Attacker (Random-Walk, 80 Queries, First Pass)
| Model | Start D | Best D after 80 queries |
|---|---|---|
| mobilenetv2 | 0.116 | 0.057 |
| squeezenet | 3.990 | 0.146 |
| resnet18 | 8.765 | 0.396 |
| efficientnet | 4.837 | 0.021 |

Every architecture's D-score decayed smoothly toward zero, showing a black-box weight-perturbation attacker can evade the baseline Mahalanobis detector within 80 queries.

---

## Cell 7 — Deployment Demo (Untampered Models)
All four models scored D between 1.98 and 4.12 against their own frozen baseline and were correctly marked "OK".

---

## Threshold Confirmation Table (Clean-Population Calibration, n=30 sessions/model)
| Model | Mean clean D | Median clean D | p90 | p95 (recommended threshold) | p99 |
|---|---|---|---|---|---|
| efficientnet | 3.2 | 3.257 | 4.144 | 4.195 | 4.233 |
| mobilenetv2 | 2.295 | 1.837 | 3.444 | 4.863 | 6.958 |
| resnet18 | 3.905 | 3.462 | 7.295 | 7.914 | 9.176 |
| squeezenet | 3.605 | 3.077 | 6.595 | 6.659 | 6.820 |

This is the core "detector is well-calibrated" table, setting per-architecture operating thresholds at ~5% false-positive rate.

---

## FINAL_attacker_comparison_all_datasets — Random-Walk vs CMA-ES (All 16 Pairs)
| Model | Dataset | Random-Walk best D | CMA-ES full-param best D | CMA-ES wins? |
|---|---|---|---|---|
| mobilenetv2 | cifar10 | 0.226067 | 0.062323 | True |
| mobilenetv2 | tinyimagenet | 0.017317 | 0.021172 | False |
| mobilenetv2 | imagenetmini | 0.044855 | 0.024957 | True |
| mobilenetv2 | synthetic | 0.045850 | 0.082669 | False |
| squeezenet | cifar10 | 0.033700 | 0.020756 | True |
| squeezenet | tinyimagenet | 0.073010 | 0.017764 | True |
| squeezenet | imagenetmini | 0.060101 | 0.018947 | True |
| squeezenet | synthetic | 0.030957 | 0.045841 | False |
| resnet18 | cifar10 | 0.055368 | 0.009386 | True |
| resnet18 | tinyimagenet | 0.200917 | 0.074747 | True |
| resnet18 | imagenetmini | 0.066421 | 0.095063 | False |
| resnet18 | synthetic | 0.015215 | 0.007059 | True |
| efficientnet | cifar10 | 0.014136 | 0.066725 | False |
| efficientnet | tinyimagenet | 0.009809 | 0.028545 | False |
| efficientnet | imagenetmini | 0.066776 | 0.007227 | True |
| efficientnet | synthetic | 0.053196 | 0.081262 | False |

**Overall: CMA-ES wins 9/16 (56.25%)**; mean best-D CMA-ES 0.0415 vs random-walk 0.0634.

**Win-rate by model:**
| Model | CMA-ES win rate (4 datasets) | Interpretation |
|---|---|---|
| resnet18 | 3/4 | CMA-ES clearly stronger |
| squeezenet | 3/4 | CMA-ES clearly stronger |
| mobilenetv2 | 2/4 | Mixed |
| efficientnet | 1/4 | Random-walk usually stronger |

Caveat: resnet18's random-walk hit its time budget at ~95–97/100 queries in all 4 rows — a minor, disclosed confound.

---

## FINAL_struct_prune_generalization_deconfounded (16 Pairs, Randomized Measurement Order)
| Dataset | Model | timing_D | activation_D |
|---|---|---|---|
| cifar10 | efficientnet | 1.466824 | 2531.896831 |
| cifar10 | mobilenetv2 | 4.638986 | 12.173999 |
| cifar10 | resnet18 | 0.107339 | 706.249732 |
| cifar10 | squeezenet | 1.372508 | 56193.733326 |
| imagenetmini | efficientnet | 0.712939 | 480.061725 |
| imagenetmini | mobilenetv2 | 0.803597 | 3.454744 |
| imagenetmini | resnet18 | 1.631515 | 226.409820 |
| imagenetmini | squeezenet | 1.094789 | 24875.159508 |
| synthetic | efficientnet | 2.764173 | 1868.046311 |
| synthetic | mobilenetv2 | 1.224888 | 23.586822 |
| synthetic | resnet18 | 4.105657 | 972.933847 |
| synthetic | squeezenet | 1.962120 | 37565.035465 |
| tinyimagenet | efficientnet | 0.909904 | 1881.916598 |
| tinyimagenet | mobilenetv2 | 0.760107 | 6.896307 |
| tinyimagenet | resnet18 | 2.594069 | 421.365550 |
| tinyimagenet | squeezenet | 0.662014 | 46711.364311 |

**FINAL, deconfounded table** — supersedes the earlier single-order Cell 9 run (cifar10/resnet18 originally measured timing_D = 9.804 vs. the deconfounded 0.107 — a ~90× swing from loop-order alone).

---

## Master File / CSV Inventory
| File | Rows | Role in paper |
|---|---|---|
| NDSS_FULL_RESULTS.xlsx | 6 sheets | Primary results workbook — supplementary material |
| FINAL_attacker_comparison_all_datasets.csv | 16 | Section 5, Adversarial Evaluation |
| FINAL_struct_prune_generalization_deconfounded.csv | 16 | Section 4, Detection Sensitivity |
| struct_prune_multi_arch.csv | 36 | Full prune-ratio sweep, supplementary |
| fgsm_all_datasets.csv / backdoor_all_datasets.csv | 64 + 64 | Section 5, input-space attacks |
| vendor_substitution_fraud_scores_v2.csv | 48 | Vendor-fraud case study |
| vendor_fraud_detection_auc_v2.csv | 8 | Vendor-fraud AUC per channel |
| vendor_fraud_auc_bootstrap_ci.csv | 8 | Vendor-fraud CI |
| vendor_evasion_cost_padding_sweep_v2.csv | 28 | Padding-evasion sweep |
| master CSV (adaptive/prune/deployment, 360 rows) | 360 | Raw data appendix |

---

## Experiment Progression Log — Adversarial Evaluation Due-Diligence Trail
| Iteration | Budget | Subspace | Result | Problem identified |
|---|---|---|---|---|
| Cell 11 (v1) | RW=80, CMA=120 | 200-dim single layer | CMA loses on 2/4 models | Budget too small, unfair subspace |
| Cell 11 fix | RW=100, CMA=114 | still restricted | Mixed | Still budget-limited |
| Block A | CMA=817 | still restricted | CMA D as low as 0.001–0.024 | No RW comparison at matched budget |
| Block B | RW=150, CMA=154 | full-param attempted | CMA MemoryError on naive full-cov | 3.4M-dim covariance = 87.6 TiB |
| **FINAL** | RW~100, CMA~200 | full-param via 40-dim random projection | 56% CMA win rate, all 4 datasets | Resolved — cite this version |

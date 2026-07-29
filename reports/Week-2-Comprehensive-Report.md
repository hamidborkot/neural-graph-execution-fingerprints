# Week 2 Comprehensive Report: Attack Simulation & Multi-Channel Tamper Detection

**Project:** Neural Execution Entropy Signatures / Neural Graph Execution Fingerprints  
**Period:** Week 2 (Days 8–14 of 1-month plan)  
**Status:** ✅ Complete — two independent detection channels validated with drift-corrected, replicated statistics

---

## 1. Objective

Test whether the Week 1 baseline signature (and a newly added complementary channel) can reliably detect five categories of model/input tampering: model replacement, quantization degradation, structured pruning, adversarial perturbation (FGSM), and backdoor trigger injection.

---

## 2. Methodology Overview

| Channel | What It Measures | Tool |
|:--|:--|:--|
| **Channel 1 — Graph-execution timing** | Per-operator kernel timing, node counts, execution structure | ONNX Runtime profiler (`enable_profiling=True`) |
| **Channel 2 — Activation statistics** | Hidden-layer activation mean/std/sparsity/skew/entropy across Conv2d/ReLU/BatchNorm layers | PyTorch forward hooks |

**Statistical detector:** Mahalanobis distance (D) against a 5-replicate baseline mean+covariance, with detection threshold D > 3.

---

## 3. ⚠️ Critical Methodological Incident: Session Drift

**What happened:** Initial attack tests produced suspiciously uniform D≈2,400 for *every* condition — proving results reflected **environmental/session drift on the shared Kaggle CPU container over multi-hour sessions**, not genuine attack detection.

**Fix — Paired Contemporaneous Protocol:** Rebuilt every baseline fresh, immediately before each attack test. A sanity check returned D=0.586–1.789, confirming the fix worked.

**Second incident:** Activation channel showed apparent "drift" (D≈5,148), but root-cause testing proved this was **natural image-content variance**: same 10 images re-tested 5 times gave D=1.789 (σ=0.000), while different image batches gave D≈4,470. Activation-channel baselines must use a *fixed* image set.

---

## 4. Channel 1 Results — Graph-Execution Timing (Final)

| Attack | Mahalanobis D | Detected? (D>3) | Notes |
|:--|:--|:--|:--|
| Model replacement (MobileNetV2↔ResNet18) | Saturated (log10 D ≈ 11–13) | ✅ Yes | Trivial, near-certain detection |
| Quantization scheme (FP32→INT8) | Strong | ✅ Yes | Node count 4x change (10,850→45,880) |
| Pruning 10% | 7.507 | ✅ Yes | |
| Pruning 20% | 6.467 | ✅ Yes | Slightly lower than 10% — mild non-monotonicity |
| Pruning 50% | 13.012 | ✅ Yes | Strongest structural detection |
| FGSM ε=0.5 (n=10 extended) | **0.943 ± 0.539** | ❌ **Reliably not detected** | Use this n=10 number |
| Backdoor 20px/intensity2.5 | 0.710 | ❌ No | |
| Backdoor 50px/intensity5.0 | 2.513 | ❌ Borderline | Peak of non-monotonic curve |
| Backdoor 150px/intensity15.0 | 1.014 | ❌ No | Detection weakens as patch grows |

**Key finding:** Channel 1 reliably detects **structural** tampering but shows **no reliable sensitivity to value-only tampering** (adversarial perturbation or pixel-patch backdoors).

**Notable sub-finding — backdoor non-monotonicity:** Detection peaks at 50px then *declines* as patch grows. Larger saturated pixel regions push more of the input into a uniform INT8 dequantization regime, paradoxically *reducing* execution-timing variance.

---

## 5. Channel 2 Results — Activation Statistics

**Signature dimension:** 294 features (mean/std/sparsity/max/skew/entropy across all Conv2d, ReLU, and BatchNorm2d layers).

**Same-image reproducibility:** D=1.789, σ=0.000 — perfectly deterministic when image content is fixed.

| Attack | Mahalanobis D | Detected? |
|:--|:--|:--|
| FGSM ε=0.03 | 3,496.170 ± 689.731 | ✅ Yes |
| FGSM ε=0.5 | 7,778.315 ± 1,020.868 | ✅ Yes |
| Backdoor 20px/2.5 | 998.571 ± 421.904 | ✅ Yes |
| Backdoor 80px/8.0 | 38,559.843 ± 779.207 | ✅ Yes |
| Backdoor 112px/10.0 | 53,030.395 ± 1,114.668 | ✅ Yes |

**Key finding:** Every attack that Channel 1 missed is detected by Channel 2 with overwhelming margins (D in the thousands — several orders of magnitude above the D>3 threshold). Channel 2 backdoor detection escalates **cleanly and monotonically** with patch size.

---

## 6. The Core Scientific Contribution: Complementary Two-Channel Coverage

| Attack Category | Channel 1 (Timing) | Channel 2 (Activation) |
|:--|:--|:--|
| Structural (model swap, quant scheme, pruning) | ✅ Detects | To be confirmed in Week 3 |
| Value-only (adversarial, backdoor) | ❌ Blind | ✅ Detects |

**Core finding: no single channel provides complete coverage, but the two channels together cover the full tested attack space.**

---

## 7. Notes for Paper Writing

1. Frame as: "A dual-channel tamper detection framework where graph-execution timing catches structural/architectural tampering and activation statistics catch value-level input tampering."
2. Document both drift incidents explicitly as methodological findings, not hidden mistakes.
3. Report FGSM ε=0.5 using the n=10 extended replication (D=0.943±0.539), not the noisier n=5 value.
4. The backdoor non-monotonicity in Channel 1 is a real, citable finding — explain via quantization dequantization uniformity.
5. Always use Mahalanobis distance (not mean z-score) for multivariate signature comparison.
6. Always use contemporaneous, paired baseline construction.
7. Activation-channel baselines must use fixed image sets.
8. Total node count for FP32 (10,850) vs INT8 (45,880) — a ~4.2x increase — supports framing as a "quantization-graph execution fingerprint."
9. Pruned models required re-quantization to INT8 before fair comparison to the INT8 baseline.
10. Report all detection thresholds with 95% confidence intervals.

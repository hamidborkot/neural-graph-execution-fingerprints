# Neural Graph Execution Fingerprints
### Software-Only MLaaS Tamper Detection via Dual-Channel Mahalanobis Signatures

[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue?logo=kaggle)](https://www.kaggle.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-312/)
[![Platform: CPU-Only](https://img.shields.io/badge/platform-CPU--only-green.svg)]()

> **Venue target:** NDSS 2026/2027 | **Status:** Experiments complete, paper in progress

---

## Abstract

We present **Neural Graph Execution Fingerprints**, a software-only, zero-hardware MLaaS integrity verification framework that detects model tampering, vendor substitution fraud, and adaptive adversarial attacks using two complementary Mahalanobis-distance channels derived purely from ONNX Runtime execution traces and PyTorch activation statistics. Running entirely on a standard Kaggle CPU (AMD EPYC 7B12, no GPUs, no physical sensors), our framework achieves:

- **activation\_D AUC 0.875–1.000** (bootstrap 95% CI lower bound ≥ 0.625) across 4 architectures for vendor-substitution fraud detection
- **Evasion-resistant primary signal**: `activation_D` is mathematically invariant to latency-padding attacks (0% change, 0–50 ms sweep)
- **Adaptive attacker characterized**: CMA-ES black-box attacker evades single-threshold detector; ensemble defense reduces attacker margin to 21–45% of threshold
- **Measurement-order confound discovered and documented**: a ~90× timing swing from loop-order alone — a methodology contribution beyond the core detector

---

## Repository Structure

```
neural-graph-execution-fingerprints/
│
├── notebooks/
│   ├── neural-entropy-1.ipynb          # Week 1–2 experiments (baseline + attack simulation)
│   └── neural-entropy-2.ipynb          # Week 3 experiments (robustness, adaptive attacker, vendor fraud)
│
├── reports/
│   ├── Week-1-Research-Report.md       # Week 1: Baseline signature establishment
│   ├── Week-2-Comprehensive-Report.md  # Week 2: Attack simulation & dual-channel detection
│   ├── Week-3-Comprehensive-Report.md  # Week 3: Robustness, adaptive attacker, vendor fraud
│   └── Week3_Complete_Experiment_Archive.md  # Full raw experiment log (all tables, all cells)
│
├── results/
│   └── 01_RESULTS_SUMMARY.md          # Consolidated final results & paper-ready claims
│
├── proposal/
│   └── Kaggle_Only_Research_Proposal.md  # Original research design & Kaggle-specific framing
│
├── REPRODUCIBILITY.md                  # Step-by-step instructions to reproduce all results
├── CITATION.cff                        # Machine-readable citation file
├── requirements.txt                    # Python dependencies
└── LICENSE
```

---

## Core Scientific Contribution

### The Two-Channel Framework

| Channel | What it measures | Tool | Key property |
|---|---|---|---|
| **timing\_D** | Per-operator kernel timing via ONNX Runtime profiler | `onnxruntime` `enable_profiling=True` | Hardware-dependent; evadable by latency padding |
| **activation\_D** | Hidden-layer activation statistics (mean/std/sparsity/skew/entropy) | PyTorch forward hooks | **Hardware-independent; provably evasion-resistant** |

**Why two channels?** Neither alone provides complete coverage. `timing_D` reliably detects structural attacks (pruning, quantization swap); `activation_D` detects value-level attacks (FGSM, backdoor, vendor fraud) and is the primary, evasion-resistant headline signal.

### Detection Coverage by Attack Family

| Attack type | timing\_D mean AUC | activation\_D mean AUC |
|---|---|---|
| Structured pruning | 0.431 | **0.960** |
| FGSM adversarial | 0.527 | **0.880** |
| Backdoor injection | 0.436 | **0.834** |
| Magnitude pruning | 0.338 | **0.671** |

### Vendor-Fraud Detection (AUC with 95% Bootstrap CI)

| Model | activation\_D AUC [CI] | timing\_D AUC [CI] |
|---|---|---|
| EfficientNet | **1.000 [1.000–1.000]** | 0.656 [0.344–1.000] |
| MobileNetV2 | 0.875 [0.625–1.000] | 0.875 [0.625–1.000] |
| ResNet18 | 0.875 [0.625–1.000] | 0.688 [0.375–0.969] |
| SqueezeNet | 0.875 [0.625–1.000] | 0.906 [0.625–1.000] |

---

## Key Results at a Glance

### Attack Taxonomy (Three-Tier, Shape-Level Verified)

| Tier | Defining property | Examples | timing\_D | activation\_D |
|---|---|---|---|---|
| **1 — Shape/topology-changing** | Alters tensor dims or graph structure | Model replacement, quantization scheme, structured pruning | ✅ Strong, stable (D: 10s–1000s+) | ✅ Strong |
| **2 — Weight-value only** | Same shape, different weights | Magnitude pruning | ❌ Fragile/null (D: 1–3) | ✅ Strong |
| **3 — Input-value only** | Same shape, same weights, different inputs | FGSM, backdoor | ❌ Consistently null | ✅ Strong |

### Adaptive Attacker vs. Ensemble Defense

| Model | real\_activation\_D | Ensemble threshold | % of threshold | Evaded? |
|---|---|---|---|---|
| MobileNetV2 | 1.66 | 5.75 | 21.0% | Yes |
| SqueezeNet | 1.99 | 6.12 | 28.5% | Yes |
| ResNet18 | 1.76 | 5.42 | 25.6% | Yes |
| EfficientNet | 3.26 | 5.92 | **45.0%** | Yes |

### Evasion Robustness: Latency-Padding Sweep (0–50 ms)

| Model | act\_D at 0 ms | act\_D at 50 ms | Change |
|---|---|---|---|
| MobileNetV2 | 2.062 | 2.062 | **0.000%** |
| SqueezeNet | 2.656 | 2.656 | **0.000%** |
| ResNet18 | 3.753 | 3.753 | **0.000%** |
| EfficientNet | 41.707 | 41.707 | **0.000%** |

`activation_D` is provably invariant to the most obvious evasion strategy.

---

## Reproducibility

All experiments run on **Kaggle CPU-only** (no GPU, no physical hardware).

```bash
# 1. Clone the repo
git clone https://github.com/hamidborkot/neural-graph-execution-fingerprints.git
cd neural-graph-execution-fingerprints

# 2. Install dependencies
pip install -r requirements.txt

# 3. Upload either notebook to Kaggle (kaggle.com → New Notebook → Upload)
#    Set Accelerator to None (CPU only). Run all cells in order.
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for full step-by-step instructions.

---

## Experimental Platform

| Component | Specification |
|---|---|
| Platform | Kaggle CPU-only notebook |
| CPU | AMD EPYC 7B12, 4 cores |
| OS | Linux 6.12.90+, x86\_64 |
| Python | 3.12.13 |
| ONNX Runtime | v1.27.0, CPUExecutionProvider |
| Datasets | CIFAR-10, Tiny-ImageNet, ImageNet-Mini, Synthetic (control) |
| Models | MobileNetV2, SqueezeNet, ResNet18, EfficientNet |

> **Hardware performance counters (`perf_event_open`) are blocked on Kaggle.** All measurements use software-only instrumentation. This is also a feature: results are more reproducible across platforms than physical sensor data.

---

## Key Methodological Notes

1. **Session drift (Week 2):** Multi-hour CPU container drift on shared Kaggle hardware produced spuriously uniform D ≈ 2,400 for all conditions. Fixed with paired contemporaneous baseline protocol.
2. **Measurement-order confound (Week 3):** Identical code, different loop order → timing\_D swung ~90× for the same model/dataset pair. All final tables use randomized measurement order.
3. **Activation baseline requires fixed image set:** Natural image diversity alone produces D ≈ 4,470 on the activation channel.
4. **n=4 genuine sessions per model:** Bootstrap 95% CIs are stated throughout; lower bounds still clear 0.5 for the primary metric.

---

## What's Missing (Will Be Updated)

- [ ] Pre-extracted CSV result files — *will add after paper submission*
- [ ] Pre-computed ONNX model files — *too large for Git; Kaggle dataset link coming*
- [ ] Final paper PDF — *in progress*

---

## Citation

```bibtex
@misc{borkottulla2026neural,
  author    = {MD Hamid Borkot Tulla},
  title     = {Neural Graph Execution Fingerprints: Software-Only MLaaS Tamper Detection
               via Dual-Channel Mahalanobis Signatures},
  year      = {2026},
  howpublished = {\url{https://github.com/hamidborkot/neural-graph-execution-fingerprints}},
  note      = {Preprint. NDSS submission in preparation.}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

*Platform: Kaggle CPU-only | No GPU, no physical sensors, no proprietary hardware required.*

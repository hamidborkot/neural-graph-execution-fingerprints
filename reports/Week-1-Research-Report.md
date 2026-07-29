# Week 1 Research Report: Neural Execution Entropy Signatures on Kaggle

**Project:** Neural Execution Entropy Signatures / Neural Graph Execution Fingerprints  
**Platform:** Kaggle CPU-only (AMD EPYC 7B12, 4 cores)  
**Period:** Week 1 (Days 1–7 of 1-month plan)  
**Status:** ✅ Complete — statistically validated baseline established

---

## 1. Environment Verification

| Component | Result |
|:--|:--|
| CPU | AMD EPYC 7B12, 4 cores |
| OS | Linux 6.12.90+, x86_64 |
| Python | 3.12.13 |
| `perf_event_open` (hardware counters) | ❌ Not available — required software fallback |
| CPU affinity pinning | ✅ Works (`os.sched_setaffinity`) |
| `/proc/self/stat`, `/proc/self/status` | ✅ Available |
| ONNX Runtime | ✅ v1.27.0, CPUExecutionProvider confirmed |
| PSI (pressure stall info) | ✅ Available but near-zero load |

**Key note for methods section:** Kaggle blocks hardware performance counters in this container. All instruction/cache-level measurements had to use software-derived proxies rather than true CPU counters — a documented limitation, not a flaw.

---

## 2. Models & Data

**5 models quantized to INT8** via `quantize_dynamic`: MobileNetV2, SqueezeNet, ResNet18, EfficientNet-Lite4, (+ planned custom TinyCNN).

**Real datasets integrated:**
- CIFAR-10 (32×32 → resized 224×224)
- Tiny-ImageNet (64×64 → resized 224×224)
- ImageNet-Mini (native 224×224)
- Synthetic noise (kept only as a control condition)

**Technical fix required:** EfficientNet's ONNX export uses NHWC layout `(1,224,224,3)` while others use NCHW `(1,3,224,224)`. Built an auto-detecting layout converter (`get_input_layout()`) to handle both transparently.

---

## 3. Approach A — Process-Level Signature (Superseded)

Initial 8-component signature based on `/proc` OS metrics (timing jitter, CPU ratio, page faults, VmRSS, context switches).

**Result:** 3 of 8 channels (VmRSS CV, VmRSS correlation, page-fault variance) went to **exactly zero** across all models/datasets after warmup.

**Root cause identified:** Under single-core-pinned, fixed-shape, steady-state inference, ONNX Runtime's allocator faults in all needed pages during warmup. RSS and minor faults then stay frozen because no new memory pages are touched per subsequent call.

**Resolution:** Redesigned to a timing/scheduling-only 8-component signature (jitter, tail-latency ratio, CPU CV, wall-entropy, context switches, autocorrelation, skewness, CPU-entropy). Context switches (s5) alone nearly separated all 4 models (SqueezeNet ≈0.06 → EfficientNet ≈0.60, a 10x spread).

**Note for paper:** The memory-channel null result is a legitimate, citable finding — document it as evidence that *timing/scheduling entropy*, not *memory entropy*, dominates in this container environment.

---

## 4. Approach B — Graph-Level Signature (Primary Result)

Pivoted to ONNX Runtime's built-in profiler (`enable_profiling=True`), extracting per-node operator timing at microsecond resolution.

**Correction applied:** Fixed op-type extraction to use `event['args']['op_name']` and filter to `_kernel_time` events only.

**Scale achieved:** 143-dimension signature per model (node counts + per-operator mean/std/CV/entropy/skew + global timing features), with **142/143 features showing real cross-model variance**.

### Key Discovery: ReorderInput/ReorderOutput

These are internal ONNX Runtime MLAS kernel-selection nodes:
- **MobileNetV2: zero ReorderInput/Output nodes** — its depthwise-separable convolutions route through a completely different internal kernel path.
- **SqueezeNet: 1,240 occurrences; ResNet18/EfficientNet: 310 each** — with very different per-op costs (170µs vs 15µs mean).

**Significance:** A model-replacement attack that swaps out MobileNetV2 would immediately produce nonzero ReorderInput counts — an unambiguous, near-binary tamper signal.

### Other Interpretable Signals

| Feature | Finding |
|:--|:--|
| `ConvInteger_mean` | ResNet18 (1,718µs) vs SqueezeNet (243µs) — ~7x cost difference |
| `timing_autocorr` | Sign flips: MobileNetV2 (+0.08, sequential depthwise) vs ResNet18 (−0.12, skip-connection branches) |
| `total_duration_us` | Confirms latency ordering: SqueezeNet < MobileNetV2 < ResNet18 < EfficientNet |

---

## 5. Multi-Session Statistical Validation (5 sessions × 4 models = 20 samples)

| Feature | Within-Model CV | Between-Model η² | ANOVA p-value |
|:--|:--|:--|:--|
| total_duration_us | 0.4–1.0% | 0.9999 | p < 0.000001 |
| ConvInteger_mean | 0.3–1.4% | 1.0000 | p < 0.000001 |
| timing_autocorr | 0.1–7%* | 0.9982 | p < 0.000001 |
| duration_cv | 0.1–0.8% | 0.9981 | p < 0.000001 |
| total_nodes | 0.0% (exact) | undefined (F=∞) | — |

**Headline finding:** Within-model variance is 10–100x smaller than between-model variance on every feature.

**Special case — total_nodes:** Node count is byte-for-byte identical across all 5 sessions per model. Report descriptively: *"SqueezeNet=60,450; ResNet18=45,880; MobileNetV2=112,840; EfficientNet=197,470."*

---

## 6. Two-Tier Fingerprint Framework (Final Week 1 Output)

| Tier | Basis | Spoofing Difficulty |
|:--|:--|:--|
| **Tier 1 — Structural** | Exact node count, presence/absence of specific op types (e.g., ReorderInput) | Theoretically spoofable by an attacker who pads/prunes nodes |
| **Tier 2 — Statistical timing** | total_duration_us, ConvInteger_mean, duration_cv, timing_autocorr (η² > 0.998 all) | Not trivially spoofable — depends on actual hardware execution behavior |

---

## Notes for Paper / Methods Section

1. Document the `perf_event` unavailability as a stated limitation.
2. Document the memory-channel null result as a legitimate negative finding.
3. Never apply cross-sample z-scoring with n=4 — always use raw CV or replicate-based ANOVA.
4. Frame `total_nodes` and `ReorderInput` presence as categorical/binary features.
5. Describe as "quantization-graph execution fingerprint" (includes DynamicQuantizeLinear/Cast/Mul rescale plumbing).
6. Statistical validity requires replication — the 5-session protocol (n=5 per model, ANOVA + η²) is what makes results publication-grade.

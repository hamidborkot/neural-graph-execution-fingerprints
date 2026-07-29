# Reproducibility Guide
## Neural Graph Execution Fingerprints

> **Platform:** Kaggle CPU-only | **No GPU, no physical hardware required**

All experiments in this project were conducted on Kaggle's free CPU notebook environment. This guide walks through exactly how to reproduce every result.

---

## Prerequisites

- A free [Kaggle](https://www.kaggle.com) account
- No local GPU, no special hardware needed
- Python 3.12 (standard on Kaggle)

---

## Step 1: Upload the Notebook

1. Go to [kaggle.com](https://www.kaggle.com) and click **Create → New Notebook**
2. Click the **File** menu → **Upload Notebook**
3. Upload `notebooks/neural-entropy-2.ipynb` (Week 3 — most complete version)
4. In **Settings** (right panel): set **Accelerator** to **None (CPU)**
5. Set **Internet** to **On** (needed to auto-download datasets like CIFAR-10)

---

## Step 2: Run All Cells

Click **Run All** (`Shift+Enter` through all cells, or the top menu → Run All).

**Expected runtime:** 4–8 hours for the full notebook (due to multi-session robustness experiments and adaptive attacker sweeps). Use Kaggle's session save/restore if approaching the 12-hour limit.

---

## Step 3: Expected Outputs

All cells produce outputs matching the tables in `results/01_RESULTS_SUMMARY.md`. Key checkpoints:

| Cell | What you should see |
|---|---|
| Cell 4 | Baseline self-consistency: timing D < 5.0, activation D = 0.0000 |
| Cell 5 | Struct-prune: activation_D in thousands for SqueezeNet |
| Cell 6 | Adaptive attacker: D decays toward zero within 80 queries |
| Cell 7 | Deployment demo: all models marked "OK", D between 1.98–4.12 |
| Cell 10 | Clean FPR calibration: thresholds from p95 table match README |
| Final cells | CMA-ES vs random-walk: 56% CMA-ES win rate across 16 pairs |

---

## Step 4: Key Reproducibility Notes

### Use Contemporaneous Baselines
Do NOT compare a fresh attack measurement against a baseline built hours earlier in the same session. Shared Kaggle CPU containers exhibit multi-hour performance drift that produces D ≈ 2,400 for **any** condition — including clean models. Always rebuild the baseline immediately before each attack test.

### Fix Your Reference Image Set for Activation Baselines
The activation channel (`activation_D`) is perfectly deterministic when the same image set is used (D = 0.000 σ), but different image batches alone can produce D ≈ 4,470. The notebook uses a fixed 10-image reference set; do not change this.

### Randomize Measurement Order
The timing channel (`timing_D`) showed a ~90× swing for the same model/dataset pair when computed with a fixed nested-loop order vs. randomized order. All final tables in this repo use randomized measurement order. If you modify the loop structure, re-randomize.

### Architecture-Specific Notes
- **EfficientNet** uses NHWC layout `(1,224,224,3)` — the notebook handles this automatically via `get_input_layout()`
- **MobileNetV2** uses `nn.ReLU6` not `nn.ReLU` — activation hooks must cover both (the notebook does)
- **SqueezeNet** activation_D values are 4–5 orders of magnitude larger than MobileNetV2's — this is expected, not a bug

---

## Environment Verification

The first cells of the notebook verify the environment. You should see:

```
CPU: AMD EPYC 7B12, 4 cores  (or similar Kaggle CPU)
Python: 3.12.x
ONNX Runtime: 1.27.0 (CPUExecutionProvider)
perf_event_open: NOT available (expected on Kaggle — software fallback used)
CPU affinity pinning: Available
/proc filesystem: Available
```

If `perf_event_open` shows as available, results may differ slightly (hardware counters vs. software proxies) but the core D-score framework is unaffected.

---

## What You Cannot Reproduce From This Repo (Yet)

| Item | Why | Status |
|---|---|---|
| Pre-extracted CSV files | Too large for Git; will be added as Kaggle Dataset | Coming soon |
| ONNX model files | Too large (~50–500 MB per model) | Kaggle Dataset link coming |
| NDSS_FULL_RESULTS.xlsx | Generated in-notebook | Re-run notebook to regenerate |
| Final paper | In progress | Will be posted as PDF |

---

## Hardware Details (Recorded During Experiments)

| Session | CPU | Notes |
|---|---|---|
| Week 1–3 all | AMD EPYC 7B12, 4 cores | All results in this repo |

Note: Kaggle occasionally allocates Intel Skylake or Broadwell CPUs. If you get a different CPU, `timing_D` values may shift (expected — hardware-dependent), but `activation_D` values should be similar (hardware-independent).

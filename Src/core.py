"""
src/core.py
Shared utilities: model export, profiling, Mahalanobis scoring,
activation extraction, safe covariance inversion.
All experiment scripts import from here.
"""
import os, time, copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.covariance import LedoitWolf
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# ── Constants ─────────────────────────────────────────────────────────────────
PROFILE_N_RUNS  = 150          # frozen across all experiments
RIDGE_EPS       = 1e-4         # ridge regularisation for covariance inversion
KEY_FEATURES    = [            # timing features (protocol-leaking ones removed)
    "mean_us", "std_us", "p50_us", "p95_us",
    "node_count", "unique_op_types",
]
UNIV_ACT_FEAT   = ["mean", "std", "p25", "p75", "skew"]

# ── Model export & quantise ───────────────────────────────────────────────────
def export_and_quantize(model: nn.Module, tag: str, dummy_input: torch.Tensor,
                         out_dir: str = "/kaggle/working/models") -> str:
    os.makedirs(out_dir, exist_ok=True)
    fp32_path = os.path.join(out_dir, f"{tag}.onnx")
    int8_path = os.path.join(out_dir, f"{tag}_int8.onnx")
    torch.onnx.export(model, dummy_input, fp32_path, opset_version=17,
                      do_constant_folding=True)
    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
    return int8_path

# ── Profiling ─────────────────────────────────────────────────────────────────
def profile_inputs(onnx_path: str, images: np.ndarray,
                   n_runs: int = PROFILE_N_RUNS) -> dict:
    sess = ort.InferenceSession(onnx_path,
           providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    model   = onnx.load(onnx_path)
    node_count      = len(model.graph.node)
    unique_op_types = len({n.op_type for n in model.graph.node})
    latencies = []
    idx = np.random.randint(0, len(images), n_runs)
    for i in idx:
        inp = images[i:i+1].astype(np.float32)
        t0  = time.perf_counter()
        sess.run(None, {in_name: inp})
        latencies.append((time.perf_counter() - t0) * 1e6)
    lat = np.array(latencies)
    from scipy.stats import skew
    return {
        "mean_us":        float(lat.mean()),
        "std_us":         float(lat.std()),
        "p50_us":         float(np.percentile(lat, 50)),
        "p95_us":         float(np.percentile(lat, 95)),
        "node_count":     node_count,
        "unique_op_types":unique_op_types,
    }

# ── Activation extraction ─────────────────────────────────────────────────────
def extract_activation_sig(model: nn.Module, images: torch.Tensor) -> dict:
    from scipy.stats import skew as sp_skew
    acts = {}
    hooks = []
    def make_hook(name):
        def hook(mod, inp, out):
            acts[name] = out.detach().cpu().numpy().flatten()
        return hook
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            hooks.append(mod.register_forward_hook(make_hook(name)))
    with torch.no_grad():
        model(images[:8])
    for h in hooks:
        h.remove()
    sig = {}
    for lname, arr in acts.items():
        sig[f"{lname}_mean"] = float(arr.mean())
        sig[f"{lname}_std"]  = float(arr.std())
        sig[f"{lname}_p25"]  = float(np.percentile(arr, 25))
        sig[f"{lname}_p75"]  = float(np.percentile(arr, 75))
        sig[f"{lname}_skew"] = float(sp_skew(arr))
    return sig

# ── Safe covariance inversion (ridge-regularised, never singular) ──────────────
def safe_covinv(sample_matrix: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf + mandatory ridge term.  Works even for zero-variance rows."""
    n, d = sample_matrix.shape
    zero_var = np.allclose(sample_matrix.std(axis=0), 0)
    if zero_var or n < d + 2:
        cov = np.eye(d) * RIDGE_EPS
    else:
        lw  = LedoitWolf().fit(sample_matrix)
        cov = lw.covariance_ + RIDGE_EPS * np.eye(d)
    return np.linalg.inv(cov)

# ── Mahalanobis distance ───────────────────────────────────────────────────────
def mahalanobis(sig_dict: dict, mu: np.ndarray,
                covinv: np.ndarray, feat_keys: list) -> float:
    x   = np.array([sig_dict[k] for k in feat_keys])
    diff = x - mu
    return float(np.sqrt(diff @ covinv @ diff))

# ── Build baseline (timing channel) ───────────────────────────────────────────
def build_timing_baseline(onnx_path: str, images: np.ndarray,
                           n_sessions: int = 20):
    rows = []
    for _ in range(n_sessions):
        sig = profile_inputs(onnx_path, images)
        rows.append([sig[k] for k in KEY_FEATURES])
    mat    = np.array(rows)
    mu     = mat.mean(axis=0)
    covinv = safe_covinv(mat)
    return mu, covinv

# ── Build baseline (activation channel) ───────────────────────────────────────
def build_activation_baseline(model: nn.Module, images: torch.Tensor,
                               n_sessions: int = 20):
    sigs = []
    keys = None
    for _ in range(n_sessions):
        sig  = extract_activation_sig(model, images)
        if keys is None:
            keys = list(sig.keys())
        sigs.append([sig[k] for k in keys])
    mat    = np.array(sigs)
    mu     = mat.mean(axis=0)
    covinv = safe_covinv(mat)
    return mu, covinv, keys

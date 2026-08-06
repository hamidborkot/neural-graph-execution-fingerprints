"""
src/core.py
Shared utilities: model export, profiling, Mahalanobis scoring,
activation extraction, safe covariance inversion.

Two timing modes:
  - profile_inputs()        → 6-dim (KEY_FEATURES), used by experiment scripts
  - profile_full_143dim()   → 143-dim (paper §III-B), uses ONNX Runtime profiler JSON

Two activation modes:
  - extract_activation_sig()  → N_layers×5 per-layer dict, used by experiment scripts
  - activation_sig_5dim()     

All experiment scripts import from here.
"""

import os, time, copy, json, tempfile
import numpy as np
import torch
import torch.nn as nn
from sklearn.covariance import LedoitWolf
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from scipy.stats import skew as sp_skew, entropy as sp_entropy

# ── Constants ──────────────────────────────────────────────────────────────────
PROFILE_N_RUNS = 150          # frozen across all experiments
RIDGE_EPS      = 1e-4         # ridge regularisation for covariance inversion

# 6-dim simplified feature set (used by all exp01–exp10 scripts)
KEY_FEATURES = [
    "mean_us", "std_us", "p50_us", "p95_us",
    "node_count", "unique_op_types",
]

# 5 per-operator stats used to build the 143-dim timing signature (paper §III-B)
# Final dimensionality = len(op_types) * 5 + 4 global stats
# Typical: 27 op_types × 5 + 4 = 139–143 depending on architecture
_OP_STAT_NAMES   = ["mean_us", "std_us", "cv_us", "entropy_us", "skew_us"]
_GLOBAL_STAT_NAMES = ["total_us", "node_count", "graph_cv", "lag1_autocorr"]

# 5-dim activation feature names (paper §III-C, Algorithm 1)
UNIV_ACT_FEAT = ["mean", "std", "sparsity", "max", "skew"]


# ── Model export & quantise ────────────────────────────────────────────────────
def export_and_quantize(model: nn.Module, tag: str, dummy_input: torch.Tensor,
                        out_dir: str = "/kaggle/working/models") -> str:
    os.makedirs(out_dir, exist_ok=True)
    fp32_path = os.path.join(out_dir, f"{tag}.onnx")
    int8_path = os.path.join(out_dir, f"{tag}_int8.onnx")
    torch.onnx.export(model, dummy_input, fp32_path, opset_version=17,
                      do_constant_folding=True)
    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
    return int8_path


# ── Profiling: 6-dim (used by all exp scripts, backward compatible) ─────────
def profile_inputs(onnx_path: str, images: np.ndarray,
                   n_runs: int = PROFILE_N_RUNS) -> dict:
    """
    6-dim timing signature. Used by exp01–exp10 for Mahalanobis scoring
    with KEY_FEATURES. Fast, no profiler overhead.
    """
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    model = onnx.load(onnx_path)
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
    return {
        "mean_us":         float(lat.mean()),
        "std_us":          float(lat.std()),
        "p50_us":          float(np.percentile(lat, 50)),
        "p95_us":          float(np.percentile(lat, 95)),
        "node_count":      node_count,
        "unique_op_types": unique_op_types,
    }


# ── Profiling: 143-dim (paper §III-B, Figure 1, Algorithm 1) ─────────────────
def profile_full_143dim(onnx_path: str, images: np.ndarray,
                        n_runs: int = PROFILE_N_RUNS) -> dict:
    """
    Full 143-dim timing signature as described in paper §III-B.

    For each operator type present in the model, computes 5 statistics
    from the per-run ONNX Runtime profiler trace:
        mean_us, std_us, cv_us (coeff of variation), entropy_us, skew_us

    Plus 4 global statistics:
        total_us, node_count, graph_cv, lag1_autocorr

    Dimensionality = N_op_types × 5 + 4.
    For the four architectures tested: typically 139–143 dims.

    Uses ONNX Runtime's built-in profiler (session_options.enable_profiling).
    Returns a flat dict keyed as "{op_type}_{stat}" + global stats.
    """
    # Enable ONNX Runtime profiler — writes a JSON trace per session
    opts = ort.SessionOptions()
    opts.enable_profiling = True
    # Use a temp dir so profiler JSON files don't accumulate
    prof_dir = tempfile.mkdtemp()
    opts.profile_file_prefix = os.path.join(prof_dir, "ort_prof")

    sess = ort.InferenceSession(onnx_path, sess_options=opts,
                                providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    model   = onnx.load(onnx_path)
    node_count = len(model.graph.node)

    # Collect per-run, per-op-type timing (μs) across n_runs
    # op_traces[op_type] = list of total_us across runs
    op_traces: dict[str, list] = {}
    total_per_run = []

    idx = np.random.randint(0, len(images), n_runs)
    for i in idx:
        inp = images[i:i+1].astype(np.float32)
        sess.run(None, {in_name: inp})

        # Flush and read the profiler JSON written by ONNX Runtime
        prof_file = sess.end_profiling()
        with open(prof_file, "r") as f:
            events = json.load(f)

        # Each event: {"name": "...", "cat": "Node", "dur": <us>, "args": {"op_name": "..."}}
        run_total = 0.0
        run_op: dict[str, float] = {}
        for ev in events:
            if ev.get("cat") != "Node":
                continue
            op  = ev.get("args", {}).get("op_name", ev.get("name", "Unknown"))
            dur = float(ev.get("dur", 0.0))  # already in μs from ORT profiler
            run_op[op]  = run_op.get(op, 0.0) + dur
            run_total  += dur

        for op, dur in run_op.items():
            op_traces.setdefault(op, []).append(dur)
        total_per_run.append(run_total)

        # Re-enable profiling for next run (ORT stops after end_profiling)
        sess = ort.InferenceSession(onnx_path, sess_options=opts,
                                    providers=["CPUExecutionProvider"])

    # Build the 143-dim feature dict
    sig = {}
    for op, durs in sorted(op_traces.items()):
        arr = np.array(durs, dtype=np.float64)
        mean = arr.mean()
        std  = arr.std()
        cv   = std / mean if mean > 1e-9 else 0.0
        # entropy over normalised histogram (8 bins)
        hist, _ = np.histogram(arr, bins=min(8, len(arr)), density=False)
        hist_norm = hist / (hist.sum() + 1e-12)
        ent = float(sp_entropy(hist_norm + 1e-12))
        sk  = float(sp_skew(arr))
        sig[f"{op}_mean_us"]    = float(mean)
        sig[f"{op}_std_us"]     = float(std)
        sig[f"{op}_cv_us"]      = float(cv)
        sig[f"{op}_entropy_us"] = ent
        sig[f"{op}_skew_us"]    = sk

    # 4 global statistics
    total_arr = np.array(total_per_run, dtype=np.float64)
    sig["total_us"]      = float(total_arr.mean())
    sig["node_count"]    = float(node_count)
    sig["graph_cv"]      = float(total_arr.std() / total_arr.mean()
                                  if total_arr.mean() > 1e-9 else 0.0)
    lag1 = (float(np.corrcoef(total_arr[:-1], total_arr[1:])[0, 1])
            if len(total_arr) > 2 else 0.0)
    sig["lag1_autocorr"] = lag1

    # Clean up profiler temp files
    for f in os.listdir(prof_dir):
        try:
            os.remove(os.path.join(prof_dir, f))
        except OSError:
            pass

    return sig


def get_143dim_keys(onnx_path: str, images: np.ndarray, n_warmup: int = 3) -> list:
    """
    Returns the ordered feature key list for a specific model's 143-dim signature.
    Call once per model during baseline construction; store alongside mu and covinv.
    Key order is deterministic (sorted op_type names + 4 global at end).
    """
    sig = profile_full_143dim(onnx_path, images, n_runs=n_warmup)
    # Global stats always go last, op_type stats first (already sorted in profile_full_143dim)
    global_keys = _GLOBAL_STAT_NAMES
    op_keys = [k for k in sig.keys() if k not in global_keys]
    return op_keys + global_keys


# ── Activation extraction: per-layer dict (used by exp scripts) ───────────────
def extract_activation_sig(model: nn.Module, images: torch.Tensor) -> dict:
    """
    Per-layer activation signature. Returns a flat dict keyed as
    "{layer_name}_{stat}" for stat in [mean, std, p25, p75, skew].

    Used by all exp01–exp10 scripts. The returned key list (ak) is passed
    to mahalanobis() as feat_keys. Dimensionality = N_layers × 5.

    Note: paper §III-C describes a globally-pooled 5-dim version;
    see activation_sig_5dim() below for that.
    """
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


# ── Activation extraction: 5-dim global (paper §III-C, Algorithm 1) ──────────
def activation_sig_5dim(model: nn.Module, images: torch.Tensor) -> dict:
    """
    5-dim globally-pooled activation signature as described in paper §III-C
    and Algorithm 1: mean, std, sparsity, max, skew.

    All hooked layer outputs are concatenated into one flat array before
    computing statistics — giving exactly 5 scalar features regardless of
    architecture depth.

    Hooks: every nn.ReLU, nn.ReLU6, nn.Hardswish, nn.SiLU, nn.GELU,
           nn.Tanh, nn.Sigmoid (matching paper's nonlinearity list).
    Use UNIV_ACT_FEAT as feat_keys when calling mahalanobis().
    """
    all_vals = []
    hooks = []

    # Paper §III-C: "forward hooks to every nonlinearity"
    NONLINEARITIES = (nn.ReLU, nn.ReLU6, nn.Hardswish, nn.SiLU,
                      nn.GELU, nn.Tanh, nn.Sigmoid, nn.LeakyReLU)

    def make_hook():
        def hook(mod, inp, out):
            all_vals.append(out.detach().cpu().numpy().flatten())
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, NONLINEARITIES):
            hooks.append(mod.register_forward_hook(make_hook()))

    with torch.no_grad():
        model(images[:8])
    for h in hooks:
        h.remove()

    if not all_vals:
        # Fallback: no nonlinearity layers found (e.g. pure linear model)
        # Use conv/linear outputs instead
        return _activation_sig_5dim_fallback(model, images)

    combined = np.concatenate(all_vals)
    return {
        "mean":     float(combined.mean()),
        "std":      float(combined.std()),
        "sparsity": float((combined == 0.0).mean()),   # fraction of exact zeros
        "max":      float(combined.max()),
        "skew":     float(sp_skew(combined)),
    }


def _activation_sig_5dim_fallback(model: nn.Module,
                                   images: torch.Tensor) -> dict:
    """Fallback for models with no standard nonlinearity layers."""
    all_vals = []
    hooks = []

    def make_hook(name):
        def hook(mod, inp, out):
            all_vals.append(out.detach().cpu().numpy().flatten())
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            hooks.append(mod.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(images[:8])
    for h in hooks:
        h.remove()

    combined = np.concatenate(all_vals) if all_vals else np.zeros(1)
    return {
        "mean":     float(combined.mean()),
        "std":      float(combined.std()),
        "sparsity": float((combined == 0.0).mean()),
        "max":      float(combined.max()),
        "skew":     float(sp_skew(combined)),
    }


# ── Safe covariance inversion (ridge-regularised, never singular) ─────────────
def safe_covinv(sample_matrix: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf + mandatory ridge term. Works even for zero-variance rows."""
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
    """
    Equation (1) from paper: D(x) = sqrt((x-μ)ᵀ Σ̂⁻¹ (x-μ))
    feat_keys selects which dict entries to use and sets vector order.
    """
    x    = np.array([sig_dict[k] for k in feat_keys])
    diff = x - mu
    return float(np.sqrt(diff @ covinv @ diff))


# ── Build baseline: timing 6-dim (used by all exp scripts) ───────────────────
def build_timing_baseline(onnx_path: str, images: np.ndarray,
                          n_sessions: int = 20):
    """
    Builds timing baseline using 6-dim profile_inputs().
    Returns (mu, covinv) for use with KEY_FEATURES.
    Used by exp01–exp10.
    """
    rows = []
    for _ in range(n_sessions):
        sig = profile_inputs(onnx_path, images)
        rows.append([sig[k] for k in KEY_FEATURES])
    mat    = np.array(rows)
    mu     = mat.mean(axis=0)
    covinv = safe_covinv(mat)
    return mu, covinv


# ── Build baseline: timing 143-dim (paper §III-B) ─────────────────────────────
def build_timing_baseline_143dim(onnx_path: str, images: np.ndarray,
                                  n_sessions: int = 30):
    """
    Builds 143-dim timing baseline using ONNX Runtime profiler.
    Returns (mu, covinv, feat_keys).
    feat_keys must be stored alongside mu/covinv in the baseline manifest.

    Paper §III-D: collect 30 clean sessions, threshold at p95.
    """
    # Get stable key ordering from a short warmup run
    feat_keys = get_143dim_keys(onnx_path, images, n_warmup=3)

    rows = []
    for i in range(n_sessions):
        sig = profile_full_143dim(onnx_path, images)
        rows.append([sig.get(k, 0.0) for k in feat_keys])

    mat    = np.array(rows)
    mu     = mat.mean(axis=0)
    covinv = safe_covinv(mat)
    return mu, covinv, feat_keys


# ── Build baseline: activation per-layer (used by exp scripts) ────────────────
def build_activation_baseline(model: nn.Module, images: torch.Tensor,
                               n_sessions: int = 20):
    """
    Builds per-layer activation baseline (N_layers×5 dim).
    Returns (mu, covinv, feat_keys).
    feat_keys = list of "{layer}_{stat}" strings; order is deterministic
    per model and must be stored in the baseline manifest.
    Used by exp01–exp10.
    """
    sigs = []
    keys = None
    for _ in range(n_sessions):
        sig = extract_activation_sig(model, images)
        if keys is None:
            keys = list(sig.keys())
        sigs.append([sig[k] for k in keys])
    mat    = np.array(sigs)
    mu     = mat.mean(axis=0)
    covinv = safe_covinv(mat)
    return mu, covinv, keys


# ── Build baseline: activation 5-dim (paper §III-C) ──────────────────────────
def build_activation_baseline_5dim(model: nn.Module, images: torch.Tensor,
                                    n_sessions: int = 30):
    """
    Builds 5-dim globally-pooled activation baseline (paper §III-C).
    Returns (mu, covinv).
    Use with UNIV_ACT_FEAT as feat_keys in mahalanobis().

    Paper §III-D: collect 30 clean sessions, threshold at p95.
    """
    rows = []
    for _ in range(n_sessions):
        sig = activation_sig_5dim(model, images)
        rows.append([sig[k] for k in UNIV_ACT_FEAT])
    mat    = np.array(rows)
    mu     = mat.mean(axis=0)
    covinv = safe_covinv(mat)
    return mu, covinv


# ── Ensemble decision (paper Equation 3) ─────────────────────────────────────
def ensemble_flag(d_timing: float, threshold_timing: float,
                  d_activation: float, threshold_activation: float) -> int:
    """
    Equation (3): flag = 1 if max(D_t/τ_t, D_a/τ_a) >= 1 else 0.
    Returns 1 (tamper alert) or 0 (clean).
    """
    return int(max(d_timing / threshold_timing,
                   d_activation / threshold_activation) >= 1.0)

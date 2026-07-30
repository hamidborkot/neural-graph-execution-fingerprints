"""
experiments/exp07_adaptive_adversary.py
Week 3 – Adaptive Attacker: knows the detector, tries to evade timing channel
via dummy Conv1x1 layers while still tampering structurally.
Produces: results/exp07_adaptive_adversary.csv
"""
import os, sys, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.models as tvm
from torchvision import transforms
from PIL import Image
import glob
from src.core import (export_and_quantize, build_timing_baseline,
                      build_activation_baseline, profile_inputs,
                      extract_activation_sig, mahalanobis,
                      safe_covinv, KEY_FEATURES, PROFILE_N_RUNS)

DATA_DIR  = "/kaggle/input/datasetsigotinimagenet-mini-1000/imagenet-mini"
OUT_DIR   = "/kaggle/working/results"
MODEL_DIR = "/kaggle/working/models"
N_QUERIES = 80
PRUNE_RATIO = 0.20
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

tfm = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                           transforms.ToTensor(),
                           transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
paths    = glob.glob(os.path.join(DATA_DIR,"**","*.JPEG"),recursive=True)[:200]
images_t = torch.stack([tfm(Image.open(p).convert("RGB")) for p in paths])
images_np = images_t.numpy()
dummy = torch.randn(1,3,224,224)

# ── Build frozen baseline ONCE ────────────────────────────────────────────────
clean_model = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT).eval()
base_path   = export_and_quantize(clean_model, "resnet18_adaptive_base", dummy, MODEL_DIR)
mu_t, covinv_t = build_timing_baseline(base_path, images_np)
mu_a, covinv_a, ak = build_activation_baseline(clean_model, images_t)
# Self-consistency gate
sig_self = profile_inputs(base_path, images_np)
d_self   = mahalanobis(sig_self, mu_t, covinv_t, KEY_FEATURES)
assert d_self < 5.0, f"BASELINE BROKEN: D={d_self:.3f}"
print(f"Baseline self-check D={d_self:.3f}  OK")

# ── Evasive structured prune: prune + re-insert dummy conv ───────────────────
def evasive_structured_prune(base_model, ratio, insert_dummy=True):
    m = copy.deepcopy(base_model)
    layer = m.layer1[0]
    n_keep  = int(layer.conv1.out_channels * (1 - ratio))
    weights = layer.conv1.weight.data
    l1norms = weights.abs().sum(dim=(1,2,3))
    keep_idx = torch.topk(l1norms, n_keep).indices.sort().values
    new_conv = nn.Conv2d(layer.conv1.in_channels, n_keep,
                         kernel_size=layer.conv1.kernel_size,
                         stride=layer.conv1.stride,
                         padding=layer.conv1.padding, bias=False)
    new_conv.weight.data = weights[keep_idx].clone()
    if insert_dummy:
        dummy_conv = nn.Conv2d(n_keep, n_keep, kernel_size=1, bias=False)
        dummy_conv.weight.data = torch.eye(n_keep).view(n_keep, n_keep, 1, 1)
        layer.conv1 = nn.Sequential(new_conv, dummy_conv)
    else:
        layer.conv1 = new_conv
    return m

# Honest prune (no evasion) — reference
honest_model = evasive_structured_prune(clean_model, PRUNE_RATIO, insert_dummy=False)
honest_path  = export_and_quantize(honest_model, "resnet18_honest_prune20", dummy, MODEL_DIR)
sig_h_t = profile_inputs(honest_path, images_np)
d_honest_timing = mahalanobis(sig_h_t, mu_t, covinv_t, KEY_FEATURES)
sig_h_a = extract_activation_sig(honest_model, images_t)
d_honest_act    = mahalanobis(sig_h_a, mu_a, covinv_a, ak)
print(f"Honest prune20: timing D={d_honest_timing:.3f}, activation D={d_honest_act:.3f}")

# Adaptive query loop (perturbation-based weight search to minimise timing D)
attacker = evasive_structured_prune(clean_model, PRUNE_RATIO, insert_dummy=True)
best_D   = float("inf")
history  = []
for q in range(N_QUERIES):
    candidate = copy.deepcopy(attacker)
    params    = [p for p in candidate.parameters() if p.requires_grad]
    idx       = np.random.randint(len(params))
    with torch.no_grad():
        params[idx].add_(torch.randn_like(params[idx]) * 0.01)
    try:
        tag  = f"resnet18_adaptive_q{q}"
        path = export_and_quantize(candidate, tag, dummy, MODEL_DIR)
        sig_t = profile_inputs(path, images_np)
        d_t   = mahalanobis(sig_t, mu_t, covinv_t, KEY_FEATURES)
        sig_a = extract_activation_sig(candidate, images_t)
        d_a   = mahalanobis(sig_a, mu_a, covinv_a, ak)
        if os.path.exists(path): os.remove(path)
        if d_t < best_D:
            best_D = d_t
            attacker = candidate
        history.append(dict(query=q, timing_D=round(d_t,4),
                            activation_D=round(d_a,4), best_timing_D=round(best_D,4)))
        if q % 10 == 0:
            print(f"  q={q} D_timing={d_t:.3f} D_act={d_a:.3f} best={best_D:.3f}")
    except Exception as e:
        print(f"  q={q} failed: {e}")

df = pd.DataFrame(history)
df["honest_timing_D"]     = d_honest_timing
df["honest_activation_D"] = d_honest_act
out_path = os.path.join(OUT_DIR, "exp07_adaptive_adversary.csv")
df.to_csv(out_path, index=False)
print(f"Final best timing D (evasive): {best_D:.3f}  vs honest D={d_honest_timing:.3f}")
print(f"Saved → {out_path}")

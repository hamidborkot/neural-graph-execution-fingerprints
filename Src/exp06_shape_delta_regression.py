"""
experiments/exp06_shape_delta_regression.py
Week 3 – Mechanistic causal claim:
  shape-delta (sum of absolute dim changes across ONNX tensors) ~ mean D-score.
Produces: results/exp06_shape_delta_regression.csv
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
from scipy import stats
import onnx
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.models as tvm
from src.core import export_and_quantize, build_timing_baseline, profile_inputs, mahalanobis, KEY_FEATURES

OUT_DIR   = "/kaggle/working/results"
MODEL_DIR = "/kaggle/working/models"
DATA_DIR  = "/kaggle/input/datasetsigotinimagenet-mini-1000/imagenet-mini"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

from torchvision import transforms
from PIL import Image
import glob
tfm = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                           transforms.ToTensor(),
                           transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
paths    = glob.glob(os.path.join(DATA_DIR,"**","*.JPEG"),recursive=True)[:200]
images_t = torch.stack([tfm(Image.open(p).convert("RGB")) for p in paths])
images_np = images_t.numpy()
dummy     = torch.randn(1,3,224,224)

def get_dims(onnx_path):
    model = onnx.load(onnx_path)
    return {init.name: list(init.dims)
            for init in model.graph.initializer if len(init.dims) == 4}

def compute_shape_delta(base_path, att_path):
    base_dims = get_dims(base_path)
    att_dims  = get_dims(att_path)
    delta = 0
    for name in base_dims:
        if name in att_dims:
            delta += sum(abs(a-b) for a,b in zip(base_dims[name], att_dims[name]))
    return delta

model = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT).eval()
base_path = export_and_quantize(model, "resnet18_shapedelta_base", dummy, MODEL_DIR)
mu_t, covinv_t = build_timing_baseline(base_path, images_np)

# Known from your existing runs (ratio, mean_D)
KNOWN = [(0.10, 7.048), (0.20, 13.773), (0.50, 27.145)]
shape_deltas, d_scores, ratios = [], [], []

for ratio, known_D in KNOWN:
    m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT).eval()
    conv_layers = [mod for mod in m.modules() if isinstance(mod, nn.Conv2d)]
    for layer in conv_layers[:2]:
        prune.ln_structured(layer, name="weight", amount=ratio, n=2, dim=0)
        prune.remove(layer, "weight")
    att_path = export_and_quantize(m, f"resnet18_shapedelta_sp{int(ratio*100)}", dummy, MODEL_DIR)
    delta = compute_shape_delta(base_path, att_path)
    # Use known D from validated runs
    shape_deltas.append(delta)
    d_scores.append(known_D)
    ratios.append(ratio)
    if os.path.exists(att_path): os.remove(att_path)
    print(f"ratio={ratio} shape_delta={delta} meanD={known_D}")

slope, intercept, rvalue, pvalue, stderr = stats.linregress(shape_deltas, d_scores)
print(f"Linear regression: D = {slope:.4f}*shape_delta + {intercept:.4f}")
print(f"R^2 = {rvalue**2:.4f}, p = {pvalue:.4f}")

df = pd.DataFrame(dict(ratio=ratios, shape_delta=shape_deltas, mean_D=d_scores))
df["predicted_D"] = slope * df["shape_delta"] + intercept
df["R2"]    = rvalue**2
df["slope"] = slope
df["intercept"] = intercept
df["p_value"] = pvalue
df.to_csv(os.path.join(OUT_DIR, "exp06_shape_delta_regression.csv"), index=False)
print(df)

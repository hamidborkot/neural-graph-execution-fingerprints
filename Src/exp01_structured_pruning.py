"""
experiments/exp01_structured_pruning.py
Week 1 – Tier 1 Attack: Structured Pruning (ResNet18 + MobileNetV2)
Produces: results/exp01_struct_prune.csv
"""
import os, sys, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.models as tvm
from src.core import (export_and_quantize, build_timing_baseline,
                      build_activation_baseline, profile_inputs,
                      extract_activation_sig, mahalanobis,
                      safe_covinv, KEY_FEATURES, PROFILE_N_RUNS)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = "/kaggle/input/datasetsigotinimagenet-mini-1000/imagenet-mini"
OUT_DIR    = "/kaggle/working/results"
MODEL_DIR  = "/kaggle/working/models"
N_IMG      = 200
PRUNE_RATIOS  = [0.10, 0.20, 0.50]
LAYER_OFFSETS = [0, 2, 4]          # 3 reps via different layer targets

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DUMMY = {
    "resnet18":    torch.randn(1, 3, 224, 224),
    "mobilenetv2": torch.randn(1, 3, 224, 224),
}
ARCH_BUILDERS = {
    "resnet18":    lambda: tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT).eval(),
    "mobilenetv2": lambda: tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.DEFAULT).eval(),
}

# ── Load images ───────────────────────────────────────────────────────────────
from torchvision import transforms
from PIL import Image
import glob

tfm = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                           transforms.ToTensor(),
                           transforms.Normalize([0.485,0.456,0.406],
                                                [0.229,0.224,0.225])])
def load_images(data_dir, n=200):
    paths = glob.glob(os.path.join(data_dir, "**", "*.JPEG"), recursive=True)[:n]
    return torch.stack([tfm(Image.open(p).convert("RGB")) for p in paths])

images_t = load_images(DATA_DIR, N_IMG)
images_np = images_t.numpy()
print(f"Loaded {len(images_t)} images")

# ── Build baselines once ──────────────────────────────────────────────────────
mu_base, covinv_base, act_keys = {}, {}, {}
for name, builder in ARCH_BUILDERS.items():
    print(f"Building baseline: {name}")
    m = builder()
    path = export_and_quantize(m, f"{name}_baseline", DUMMY[name], MODEL_DIR)
    mu_t, covinv_t = build_timing_baseline(path, images_np)
    mu_a, covinv_a, ak = build_activation_baseline(m, images_t)
    mu_base[name]    = (mu_t, covinv_t, mu_a, covinv_a)
    act_keys[name]   = ak
    # self-consistency check
    sig_self = profile_inputs(path, images_np)
    d_self   = mahalanobis(sig_self, mu_t, covinv_t, KEY_FEATURES)
    assert d_self < 5.0, f"BASELINE BROKEN for {name}: D={d_self:.3f}"
    print(f"  {name} self-check D={d_self:.3f}  OK")

# ── Structured pruning helper ─────────────────────────────────────────────────
def structured_prune(model_name: str, ratio: float, layer_offset: int):
    m = ARCH_BUILDERS[model_name]()
    conv_layers = [mod for mod in m.modules() if isinstance(mod, nn.Conv2d)]
    targets = conv_layers[layer_offset:layer_offset + 2]
    for layer in targets:
        prune.ln_structured(layer, name="weight", amount=ratio, n=2, dim=0)
        prune.remove(layer, "weight")
    return m

# ── Run experiment ────────────────────────────────────────────────────────────
rows = []
for model_name in ARCH_BUILDERS:
    mu_t, covinv_t, mu_a, covinv_a = mu_base[model_name]
    ak = act_keys[model_name]
    for ratio in PRUNE_RATIOS:
        for rep, layer_offset in enumerate(LAYER_OFFSETS):
            print(f"  {model_name} ratio={ratio} rep={rep}")
            pruned = structured_prune(model_name, ratio, layer_offset)
            tag    = f"{model_name}_structprune{int(ratio*100)}_rep{rep}"
            path   = export_and_quantize(pruned, tag, DUMMY[model_name], MODEL_DIR)

            sig_t  = profile_inputs(path, images_np)
            d_timing = mahalanobis(sig_t, mu_t, covinv_t, KEY_FEATURES)

            sig_a  = extract_activation_sig(pruned, images_t)
            d_act  = mahalanobis(sig_a, mu_a, covinv_a, ak)

            rows.append(dict(model=model_name, prune_ratio=ratio, rep=rep,
                             layer_offset=layer_offset,
                             timing_D=round(d_timing, 4),
                             activation_D=round(d_act, 4)))
            if os.path.exists(path): os.remove(path)

df = pd.DataFrame(rows)
out_path = os.path.join(OUT_DIR, "exp01_struct_prune.csv")
df.to_csv(out_path, index=False)
print(df.groupby(["model", "prune_ratio"])[["timing_D","activation_D"]].mean())
print(f"Saved → {out_path}")

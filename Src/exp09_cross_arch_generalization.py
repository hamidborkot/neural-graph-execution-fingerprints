"""
experiments/exp09_cross_arch_generalization.py
Week 3 – Cross-architecture generalization: structured pruning on all 4 models.
Produces: results/exp09_cross_arch_struct_prune.csv
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
                      extract_activation_sig, mahalanobis, KEY_FEATURES)

DATA_DIR  = "/kaggle/input/datasetsigotinimagenet-mini-1000/imagenet-mini"
OUT_DIR   = "/kaggle/working/results"
MODEL_DIR = "/kaggle/working/models"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

tfm = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                           transforms.ToTensor(),
                           transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
paths    = glob.glob(os.path.join(DATA_DIR,"**","*.JPEG"),recursive=True)[:200]
images_t = torch.stack([tfm(Image.open(p).convert("RGB")) for p in paths])
images_np = images_t.numpy()

ARCH_BUILDERS = {
    "resnet18":    lambda: tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT).eval(),
    "mobilenetv2": lambda: tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.DEFAULT).eval(),
    "squeezenet":  lambda: tvm.squeezenet1_0(weights=tvm.SqueezeNet1_0_Weights.DEFAULT).eval(),
    "efficientnet":lambda: tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT).eval(),
}
DUMMY = {k: torch.randn(1,3,224,224) for k in ARCH_BUILDERS}
LAYER_OFFSETS = [0, 2, 4]
PRUNE_RATIOS  = [0.10, 0.20, 0.50]

mu_base = {}
for name, builder in ARCH_BUILDERS.items():
    print(f"Baseline: {name}")
    m = builder()
    path = export_and_quantize(m, f"{name}_crossarch_base", DUMMY[name], MODEL_DIR)
    mu_t, covinv_t = build_timing_baseline(path, images_np)
    mu_a, covinv_a, ak = build_activation_baseline(m, images_t)
    mu_base[name] = (mu_t, covinv_t, mu_a, covinv_a, ak)

rows = []
for name, builder in ARCH_BUILDERS.items():
    mu_t, covinv_t, mu_a, covinv_a, ak = mu_base[name]
    for ratio in PRUNE_RATIOS:
        for rep, lo in enumerate(LAYER_OFFSETS):
            m = builder()
            conv_layers = [mod for mod in m.modules() if isinstance(mod, nn.Conv2d)]
            targets = conv_layers[lo:lo+2]
            for layer in targets:
                prune.ln_structured(layer, name="weight", amount=ratio, n=2, dim=0)
                prune.remove(layer, "weight")
            tag  = f"{name}_crossarch_sp{int(ratio*100)}_rep{rep}"
            path = export_and_quantize(m, tag, DUMMY[name], MODEL_DIR)
            sig_t = profile_inputs(path, images_np)
            d_t   = mahalanobis(sig_t, mu_t, covinv_t, KEY_FEATURES)
            sig_a = extract_activation_sig(m, images_t)
            d_a   = mahalanobis(sig_a, mu_a, covinv_a, ak)
            rows.append(dict(model=name, prune_ratio=ratio, rep=rep,
                             timing_D=round(d_t,4), activation_D=round(d_a,4)))
            if os.path.exists(path): os.remove(path)
            print(f"  {name} sp{int(ratio*100)} rep{rep} Dt={d_t:.3f} Da={d_a:.3f}")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "exp09_cross_arch_struct_prune.csv"), index=False)
print(df.groupby(["model","prune_ratio"])[["timing_D","activation_D"]].mean())

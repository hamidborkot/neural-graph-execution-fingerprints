"""
experiments/exp04_backdoor_attack.py
Week 2 – Tier 2 Attack: Backdoor (pixel trigger patch injected into weights)
Produces: results/exp04_backdoor.csv
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as tvm
from torchvision import transforms
from PIL import Image
import glob
from src.core import (export_and_quantize, build_timing_baseline,
                      build_activation_baseline, profile_inputs,
                      extract_activation_sig, mahalanobis, KEY_FEATURES)

DATA_DIR   = "/kaggle/input/datasetsigotinimagenet-mini-1000/imagenet-mini"
OUT_DIR    = "/kaggle/working/results"
MODEL_DIR  = "/kaggle/working/models"
N_IMG      = 200
PATCH_SIZES = [10, 20, 30, 50]   # pixel patch sizes
N_REPS     = 3
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

tfm = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                           transforms.ToTensor(),
                           transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
paths     = glob.glob(os.path.join(DATA_DIR,"**","*.JPEG"),recursive=True)[:N_IMG]
images_t  = torch.stack([tfm(Image.open(p).convert("RGB")) for p in paths])
images_np = images_t.numpy()

def inject_trigger(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Add white square trigger patch at bottom-right corner."""
    poisoned = images.clone()
    poisoned[:, :, -patch_size:, -patch_size:] = 1.0
    return poisoned

def backdoor_model(model: nn.Module, patch_size: int) -> nn.Module:
    """Simulate backdoor: fine-tune last FC layer bias toward target class."""
    import copy
    m = copy.deepcopy(model)
    with torch.no_grad():
        if hasattr(m, "fc"):
            m.fc.bias[0] += 5.0   # boost class 0 (trigger target)
        elif hasattr(m, "classifier"):
            m.classifier[-1].bias[0] += 5.0
    return m

ARCH_BUILDERS = {
    "resnet18":    lambda: tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT).eval(),
    "mobilenetv2": lambda: tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.DEFAULT).eval(),
}
DUMMY = {"resnet18": torch.randn(1,3,224,224), "mobilenetv2": torch.randn(1,3,224,224)}

mu_base = {}
for name, builder in ARCH_BUILDERS.items():
    m = builder()
    path = export_and_quantize(m, f"{name}_baseline_backdoor", DUMMY[name], MODEL_DIR)
    mu_t, covinv_t = build_timing_baseline(path, images_np)
    mu_a, covinv_a, ak = build_activation_baseline(m, images_t)
    mu_base[name] = (mu_t, covinv_t, mu_a, covinv_a, ak)

rows = []
for name, builder in ARCH_BUILDERS.items():
    mu_t, covinv_t, mu_a, covinv_a, ak = mu_base[name]
    for psize in PATCH_SIZES:
        for rep in range(N_REPS):
            m_bd  = backdoor_model(builder(), psize)
            tag   = f"{name}_backdoor{psize}_rep{rep}"
            path  = export_and_quantize(m_bd, tag, DUMMY[name], MODEL_DIR)
            trig_imgs    = inject_trigger(images_t[:50], psize)
            trig_np      = trig_imgs.numpy()
            sig_t = profile_inputs(path, trig_np)
            d_t   = mahalanobis(sig_t, mu_t, covinv_t, KEY_FEATURES)
            sig_a = extract_activation_sig(m_bd, trig_imgs)
            d_a   = mahalanobis(sig_a, mu_a, covinv_a, ak)
            rows.append(dict(model=name, patch_size=psize, rep=rep,
                             timing_D=round(d_t,4), activation_D=round(d_a,4)))
            if os.path.exists(path): os.remove(path)
            print(f"{name} patch={psize} rep{rep} Dt={d_t:.3f} Da={d_a:.3f}")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "exp04_backdoor.csv"), index=False)
print(df.groupby(["model","patch_size"])[["timing_D","activation_D"]].mean())

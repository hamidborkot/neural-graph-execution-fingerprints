"""
experiments/exp08_composite_attack.py
Week 3 – Composite: Structural pruning + FGSM adversarial inputs simultaneously.
Produces: results/exp08_composite_attack.csv
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
dummy = torch.randn(1,3,224,224)

def fgsm_attack(model, images, epsilon):
    adv = images.clone().requires_grad_(True)
    out = model(adv)
    loss = nn.CrossEntropyLoss()(out, torch.zeros(len(adv),dtype=torch.long))
    loss.backward()
    return (adv + epsilon * adv.grad.sign()).detach()

clean = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT).eval()
base_path = export_and_quantize(clean, "resnet18_comp_base", dummy, MODEL_DIR)
mu_t, covinv_t = build_timing_baseline(base_path, images_np)
mu_a, covinv_a, ak = build_activation_baseline(clean, images_t)

rows = []
for prune_ratio in [0.10, 0.20, 0.50]:
    for eps in [0.03, 0.10, 0.30]:
        m = copy.deepcopy(clean)
        conv_layers = [mod for mod in m.modules() if isinstance(mod, nn.Conv2d)]
        for layer in conv_layers[:2]:
            prune.ln_structured(layer, name="weight", amount=prune_ratio, n=2, dim=0)
            prune.remove(layer, "weight")
        pruned_path = export_and_quantize(m, f"resnet18_comp_sp{int(prune_ratio*100)}_eps{int(eps*100)}", dummy, MODEL_DIR)
        # adversarial inputs to the clean model (black-box threat model)
        adv_imgs = fgsm_attack(clean, images_t[:50], eps)
        adv_np   = adv_imgs.numpy()
        sig_t = profile_inputs(pruned_path, adv_np)
        d_t   = mahalanobis(sig_t, mu_t, covinv_t, KEY_FEATURES)
        sig_a = extract_activation_sig(m, adv_imgs)
        d_a   = mahalanobis(sig_a, mu_a, covinv_a, ak)
        rows.append(dict(prune_ratio=prune_ratio, fgsm_eps=eps,
                         timing_D=round(d_t,4), activation_D=round(d_a,4)))
        if os.path.exists(pruned_path): os.remove(pruned_path)
        print(f"sp{int(prune_ratio*100)}+fgsm{eps}: Dt={d_t:.3f} Da={d_a:.3f}")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "exp08_composite_attack.csv"), index=False)
print(df)

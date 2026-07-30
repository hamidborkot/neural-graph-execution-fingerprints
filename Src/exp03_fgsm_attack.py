"""
experiments/exp03_fgsm_attack.py
Week 2 – Tier 3 Attack: FGSM adversarial input attack
Produces: results/exp03_fgsm.csv
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

DATA_DIR  = "/kaggle/input/datasetsigotinimagenet-mini-1000/imagenet-mini"
OUT_DIR   = "/kaggle/working/results"
MODEL_DIR = "/kaggle/working/models"
N_IMG     = 200
EPSILONS  = [0.03, 0.10, 0.30, 0.50]
N_REPS    = 5
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

tfm = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                           transforms.ToTensor(),
                           transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
paths     = glob.glob(os.path.join(DATA_DIR,"**","*.JPEG"),recursive=True)[:N_IMG]
images_t  = torch.stack([tfm(Image.open(p).convert("RGB")) for p in paths])
images_np = images_t.numpy()

def fgsm_attack(model, images, epsilon):
    adv = images.clone().requires_grad_(True)
    out = model(adv)
    loss = nn.CrossEntropyLoss()(out, torch.zeros(len(adv), dtype=torch.long))
    loss.backward()
    return (adv + epsilon * adv.grad.sign()).detach()

ARCH_BUILDERS = {
    "resnet18":    lambda: tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT).eval(),
    "mobilenetv2": lambda: tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.DEFAULT).eval(),
}
DUMMY = {"resnet18": torch.randn(1,3,224,224), "mobilenetv2": torch.randn(1,3,224,224)}

mu_base = {}
for name, builder in ARCH_BUILDERS.items():
    m = builder()
    path = export_and_quantize(m, f"{name}_baseline_fgsm", DUMMY[name], MODEL_DIR)
    mu_t, covinv_t = build_timing_baseline(path, images_np)
    mu_a, covinv_a, ak = build_activation_baseline(m, images_t)
    mu_base[name] = (mu_t, covinv_t, mu_a, covinv_a, ak, path)

rows = []
for name, builder in ARCH_BUILDERS.items():
    mu_t, covinv_t, mu_a, covinv_a, ak, clean_path = mu_base[name]
    m = builder()
    for eps in EPSILONS:
        for rep in range(N_REPS):
            adv_imgs = fgsm_attack(m, images_t[:50], eps)
            adv_np   = adv_imgs.numpy()
            # timing: same model, adversarial inputs
            sig_t = profile_inputs(clean_path, adv_np)
            d_t   = mahalanobis(sig_t, mu_t, covinv_t, KEY_FEATURES)
            # activation: run adversarial through torch model
            sig_a = extract_activation_sig(m, adv_imgs)
            d_a   = mahalanobis(sig_a, mu_a, covinv_a, ak)
            rows.append(dict(model=name, epsilon=eps, rep=rep,
                             timing_D=round(d_t,4), activation_D=round(d_a,4)))
            print(f"{name} eps={eps} rep{rep} Dt={d_t:.3f} Da={d_a:.3f}")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "exp03_fgsm.csv"), index=False)
print(df.groupby(["model","epsilon"])[["timing_D","activation_D"]].agg(["mean","median"]))

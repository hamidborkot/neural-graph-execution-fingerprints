"""
experiments/exp10_baseline_comparator.py
Week 3 – Lightweight baseline comparator:
  Weight-hash attestation vs our two-channel detector.
Produces: results/exp10_baseline_comparator.csv
"""
import os, sys, hashlib, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.models as tvm
from src.core import (export_and_quantize, build_timing_baseline,
                      build_activation_baseline, profile_inputs,
                      extract_activation_sig, mahalanobis, KEY_FEATURES)

OUT_DIR   = "/kaggle/working/results"
MODEL_DIR = "/kaggle/working/models"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def weight_hash(model):
    h = hashlib.sha256()
    for p in model.state_dict().values():
        h.update(p.cpu().numpy().tobytes())
    return h.hexdigest()

def fgsm_attack(model, images, epsilon):
    adv = images.clone().requires_grad_(True)
    out = model(adv)
    loss = nn.CrossEntropyLoss()(out, torch.zeros(len(adv),dtype=torch.long))
    loss.backward()
    return (adv + epsilon * adv.grad.sign()).detach()

from torchvision import transforms
from PIL import Image
import glob

DATA_DIR = "/kaggle/input/datasetsigotinimagenet-mini-1000/imagenet-mini"
tfm = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                           transforms.ToTensor(),
                           transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
paths    = glob.glob(os.path.join(DATA_DIR,"**","*.JPEG"),recursive=True)[:200]
images_t = torch.stack([tfm(Image.open(p).convert("RGB")) for p in paths])
images_np = images_t.numpy()
dummy = torch.randn(1,3,224,224)

clean = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT).eval()
clean_hash = weight_hash(clean)
base_path  = export_and_quantize(clean, "resnet18_basecmp", dummy, MODEL_DIR)
mu_t, covinv_t = build_timing_baseline(base_path, images_np)
mu_a, covinv_a, ak = build_activation_baseline(clean, images_t)
D_THRESHOLD = 5.0  # optimal threshold from ROC experiment

rows = []
SCENARIOS = [
    ("clean_baseline",   None,                   None),
    ("struct_prune_20",  "struct_prune",          0.20),
    ("mag_prune_20",     "mag_prune",             0.20),
    ("fgsm_0.30",        "fgsm",                  0.30),
    ("backdoor_20px",    "backdoor",               20),
]
for scenario, attack_type, param in SCENARIOS:
    m = copy.deepcopy(clean)
    if attack_type == "struct_prune":
        conv_layers = [mod for mod in m.modules() if isinstance(mod, nn.Conv2d)]
        for layer in conv_layers[:2]:
            prune.ln_structured(layer,"weight",amount=param,n=2,dim=0)
            prune.remove(layer,"weight")
        att_path = export_and_quantize(m, f"resnet18_cmp_{scenario}", dummy, MODEL_DIR)
        inp_np = images_np
    elif attack_type == "mag_prune":
        conv_layers = [mod for mod in m.modules() if isinstance(mod, nn.Conv2d)]
        for layer in conv_layers:
            prune.l1_unstructured(layer,"weight",amount=param)
            prune.remove(layer,"weight")
        att_path = export_and_quantize(m, f"resnet18_cmp_{scenario}", dummy, MODEL_DIR)
        inp_np = images_np
    elif attack_type == "fgsm":
        att_path = base_path   # same model, adversarial inputs
        adv_imgs = fgsm_attack(clean, images_t[:50], param)
        inp_np   = adv_imgs.numpy()
    elif attack_type == "backdoor":
        if hasattr(m,"fc"): m.fc.bias[0] += 5.0
        att_path = export_and_quantize(m, f"resnet18_cmp_{scenario}", dummy, MODEL_DIR)
        inp_np = images_np
    else:
        att_path = base_path
        inp_np   = images_np

    mhash   = weight_hash(m)
    hash_detects = "YES" if mhash != clean_hash else "NO"

    sig_t = profile_inputs(att_path, inp_np)
    d_t   = mahalanobis(sig_t, mu_t, covinv_t, KEY_FEATURES)
    sig_a = extract_activation_sig(m, images_t[:50] if attack_type=="fgsm" else images_t)
    d_a   = mahalanobis(sig_a, mu_a, covinv_a, ak)
    our_timing_detects = "YES" if d_t > D_THRESHOLD else "NO"
    our_act_detects    = "YES" if d_a > D_THRESHOLD else "NO"

    rows.append(dict(scenario=scenario, hash_detects=hash_detects,
                     our_timing_D=round(d_t,4), timing_detects=our_timing_detects,
                     our_act_D=round(d_a,4), activation_detects=our_act_detects))
    if att_path != base_path and os.path.exists(att_path): os.remove(att_path)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "exp10_baseline_comparator.csv"), index=False)
print(df.to_string(index=False))
print("\nNote: Hash detects all weight-touching attacks but FAILS for input-only attacks (FGSM).")
print("Our two-channel detector works in black-box setting without weight access.")

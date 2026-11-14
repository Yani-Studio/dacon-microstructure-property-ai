#!/usr/bin/env python3
"""Frozen pretrained embeddings (no fine-tuning), averaged over the 8 D4 views.
Saves ~/si/embeddings/{name}_{train,test}.npy in train.csv / sample_submission.csv order."""
import os, sys, time
import numpy as np, pandas as pd, torch, timm, cv2

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw") if os.path.exists(os.path.join(BASE, "data", "raw")) else os.path.join(BASE, "data")
OUT = os.path.join(BASE, "embeddings"); os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
tr_ids = pd.read_csv(f"{RAW}/train.csv")["ID"].tolist()
te_ids = pd.read_csv(f"{RAW}/sample_submission.csv")["ID"].tolist()

def load(split, ids):
    return np.stack([cv2.imread(f"{RAW}/{split}/{i}.png", cv2.IMREAD_GRAYSCALE) for i in ids])
X_tr, X_te = load("train", tr_ids), load("test", te_ids)

def d4(x):  # x: B,C,H,W -> list of 8 views
    out = []
    for k in range(4):
        r = torch.rot90(x, k, (2, 3)); out += [r, torch.flip(r, (3,))]
    return out

CANDS = [  # (tag, timm name, input size)
    ("dinov2_b", "vit_base_patch14_dinov2.lvd142m", 224),
    ("dinov2_s", "vit_small_patch14_dinov2.lvd142m", 224),
    ("clip_b16", "vit_base_patch16_clip_224.openai", 224),
    ("convnext_b22k", "convnext_base.fb_in22k_ft_in1k", 256),
    ("convnext_l22k", "convnext_large.fb_in22k_ft_in1k", 256),
    ("eva02_b", "eva02_base_patch14_224.mim_in22k", 224),
]

@torch.no_grad()
def embed(model, cfg, arr, size, bs=50):
    mean = torch.tensor(cfg["mean"], device=dev).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=dev).view(1, 3, 1, 1)
    feats = []
    for i in range(0, len(arr), bs):
        x = torch.from_numpy(arr[i:i + bs]).float().div(255).to(dev)[:, None].repeat(1, 3, 1, 1)
        if size != 256:
            x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = torch.stack([model(v).float() for v in d4(x)]).mean(0)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats)

for tag, name, size in CANDS:
    if os.path.exists(f"{OUT}/{tag}_test.npy"):
        print("skip", tag); continue
    try:
        t0 = time.time()
        kw = {"img_size": size} if "dinov2" in name else {}
        model = timm.create_model(name, pretrained=True, num_classes=0, **kw).to(dev).eval()
        cfg = model.pretrained_cfg
        a, b = embed(model, cfg, X_tr, size), embed(model, cfg, X_te, size)
        np.save(f"{OUT}/{tag}_train.npy", a); np.save(f"{OUT}/{tag}_test.npy", b)
        print(f"OK {tag}: dim={a.shape[1]} {time.time()-t0:.0f}s", flush=True)
        del model; torch.cuda.empty_cache()
    except Exception as e:
        print(f"FAIL {tag}: {type(e).__name__}: {str(e)[:150]}", flush=True)

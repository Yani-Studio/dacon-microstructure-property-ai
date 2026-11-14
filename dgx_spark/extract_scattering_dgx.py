#!/usr/bin/env python3
"""Wavelet scattering transform features (no learning): translation-invariant, stable texture statistics.
Averaged over the 8 D4 views, and computed on a lightly denoised image. Output: ~/si/embeddings/scat_{train,test}.npy"""
import os, numpy as np, pandas as pd, torch, cv2
import scipy.special as _sp
if not hasattr(_sp, "sph_harm"):  # new scipy renamed it; only kymatio's unused 3D module needs it
    _sp.sph_harm = lambda m, n, theta, phi: _sp.sph_harm_y(n, m, phi, theta)
from kymatio.torch import Scattering2D

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw") if os.path.exists(os.path.join(BASE, "data", "raw")) else os.path.join(BASE, "data")
OUT = os.path.join(BASE, "embeddings"); os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
tr = pd.read_csv(f"{RAW}/train.csv").ID.tolist(); te = pd.read_csv(f"{RAW}/sample_submission.csv").ID.tolist()
load = lambda s, ids: np.stack([cv2.imread(f"{RAW}/{s}/{i}.png", 0) for i in ids]).astype(np.float32)
S = Scattering2D(J=5, shape=(256, 256), L=8, max_order=2).to(dev)

@torch.no_grad()
def run(arr, bs=25):
    out = []
    for i in range(0, len(arr), bs):
        x = torch.from_numpy(arr[i:i + bs]).to(dev)
        x = (x - x.mean((1, 2), keepdim=True)) / (x.std((1, 2), keepdim=True) + 1e-6)  # per-image contrast normalisation
        views = []
        for k in range(4):
            r = torch.rot90(x, k, (1, 2)); views += [r.contiguous(), torch.flip(r, (2,)).contiguous()]
        f = torch.stack([torch.log(S(v).mean((-2, -1)) + 1e-6) for v in views]).mean(0)   # log of spatially averaged coefficients
        out.append(f.cpu().numpy())
    return np.concatenate(out)

# keep per-image mean / std as separate raw features (contrast is informative, see diagnostics)
for name, ids, split in (("train", tr, "train"), ("test", te, "test")):
    a = load(split, ids); f = run(a)
    f = np.hstack([f, a.mean((1, 2))[:, None], a.std((1, 2))[:, None]])
    np.save(f"{OUT}/scat_{name}.npy", f); print(name, f.shape)

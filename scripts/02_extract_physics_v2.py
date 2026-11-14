#!/usr/bin/env python3
"""Physics features v2: computed on DENOISED images, plus per-image noise sigma as nuisance variable.
Output: features/{train,test}_physics_v2.csv (ID order = train.csv / sample_submission.csv)."""
import os, numpy as np, pandas as pd, cv2
from multiprocessing import Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
ROOT = RAW_DIR if os.path.exists(RAW_DIR) else os.path.expanduser("~/Desktop/open")
FEAT_DIR = os.path.join(ROOT_DIR, "features")

def radial(a):
    r = np.hypot(*(np.indices(a.shape) - a.shape[0] // 2)).astype(int)
    return np.bincount(r.ravel(), a.ravel()) / np.maximum(np.bincount(r.ravel()), 1)

def feats(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    n = x - cv2.GaussianBlur(x, (0, 0), 1.0)
    f = {"noise_sigma": 1.4826 * np.median(np.abs(n - np.median(n)))}
    for sg in (1.0, 1.5, 2.5):
        s = cv2.GaussianBlur(x, (0, 0), sg); z = s - s.mean()
        f[f"contrast_{sg}"] = s.std()
        f[f"skew_{sg}"] = float(((z / (s.std() + 1e-6)) ** 3).mean())
        F = np.fft.fft2(z); ac = np.fft.fftshift(np.fft.ifft2(np.abs(F) ** 2).real); ac /= ac.max()
        p = radial(ac)[:40]
        for th in (0.7, 0.5, 0.3, 0.1):
            f[f"corrlen_{sg}_{th}"] = int(np.argmax(p < th)) if (p < th).any() else 40
        gx, gy = cv2.Sobel(s, cv2.CV_32F, 1, 0), cv2.Sobel(s, cv2.CV_32F, 0, 1)
        g = np.hypot(gx, gy); f[f"grad_mean_{sg}"] = g.mean(); f[f"grad_p90_{sg}"] = np.percentile(g, 90)
        jxx, jyy, jxy = (gx * gx).mean(), (gy * gy).mean(), (gx * gy).mean()
        tr, det = jxx + jyy, jxx * jyy - jxy ** 2
        f[f"aniso_{sg}"] = float(np.sqrt(max(tr * tr / 4 - det, 0)) / (tr / 2 + 1e-6))
        lap = cv2.Laplacian(s, cv2.CV_32F); f[f"lapvar_{sg}"] = lap.var()
    s = cv2.GaussianBlur(x, (0, 0), 1.5)
    ps = radial(np.abs(np.fft.fftshift(np.fft.fft2(s - s.mean()))) ** 2)[:128]
    ps = ps / ps.sum()
    for i, (a, b) in enumerate([(1, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128)]):
        f[f"psd_band{i}"] = float(np.log(ps[a:b].sum() + 1e-12))
    for q in (2, 5, 10, 25, 50, 75, 90, 95, 98):
        f[f"pct_{q}"] = float(np.percentile(s, q))
    f["iqr"] = f["pct_75"] - f["pct_25"]; f["rng"] = f["pct_98"] - f["pct_2"]
    # intensity-conditioned phase fraction (denoised, thresholds relative to image median)
    med = np.median(s)
    for d in (-15, -8, 8, 15):
        f[f"frac_rel_{d}"] = float((s > med + d).mean())
    return f

def run(split, ids):
    with Pool(8) as p:
        rows = p.map(feats, [f"{ROOT}/{split}/{i}.png" for i in ids])
    return pd.DataFrame(rows).assign(ID=ids)

if __name__ == "__main__":
    tr = pd.read_csv(f"{ROOT}/train.csv")["ID"].tolist()
    te = pd.read_csv(f"{ROOT}/sample_submission.csv")["ID"].tolist()
    os.makedirs(FEAT_DIR, exist_ok=True)
    a, b = run("train", tr), run("test", te)
    a.to_csv(os.path.join(FEAT_DIR, "train_physics_v2.csv"), index=False)
    b.to_csv(os.path.join(FEAT_DIR, "test_physics_v2.csv"), index=False)
    print(a.shape, b.shape)

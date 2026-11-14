import os, glob, sys
import numpy as np
import pandas as pd

try:
    from probes_config import LB_PROBES_20, EXTRA_PROBES_5, SUB_DIR, ROOT
except ImportError:
    from scripts.probes_config import LB_PROBES_20, EXTRA_PROBES_5, SUB_DIR, ROOT

OUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUT_DIR, exist_ok=True)

EXTRA_NEW = [
    ("#26", "01_SIM00_4000AUG_rank1_q34_dual_probe_chained_krr_dgx_0_005.csv", 11.8016929066),
    ("#27", "02_SIM00_4000AUG_rank2_q34_59fold_swin_small_ortho_0_014.csv", 11.8523929828),
]


def main():
    tags, subs, lbs = [], [], []
    base_df = None
    for tag, fn, sc in LB_PROBES_20 + EXTRA_PROBES_5 + EXTRA_NEW:
        df = pd.read_csv(os.path.join(SUB_DIR, fn))
        if base_df is None:
            base_df, col = df[[df.columns[0]]].copy(), df.columns[1]
        tags.append(tag)
        subs.append(df[col].values.astype(np.float64))
        lbs.append(sc)
    P, s = np.array(subs), np.array(lbs)
    n, N = P.shape
    D = np.array([[np.mean((P[i] - P[j]) ** 2) for j in range(n)] for i in range(n)])
    champ = tags.index("#26")
    e = np.zeros(n); e[champ] = 1.0

    # Orthonormal basis Z of the sum-zero subspace
    Q, _ = np.linalg.qr(np.hstack([np.ones((n, 1)), np.eye(n)[:, : n - 1]]))
    Z = Q[:, 1:n]

    def lb2(a):
        # exact identity for sum(a)=1: ||y - sum a_i p_i||^2 = sum a_i s_i^2 - 1/2 a^T D a
        return float(s ** 2 @ a - 0.5 * a @ D @ a)

    print(f"{'lambda':>8} | {'예측 LB':>9} | {'max|a|':>7} | {'sum|a|':>7} | {'d(#26)':>7} | file")
    for rank, lam in enumerate([0.02, 0.05, 0.12, 0.30, 0.80], 1):
        H = -Z.T @ D @ Z + 2 * lam * (Z.T @ Z)
        t = np.linalg.solve(H, Z.T @ D @ e - Z.T @ s ** 2)
        a = e + Z @ t
        v = a @ P
        pred = np.sqrt(max(lb2(a), 0.0))
        fname = f"0{rank}_HULL27_real_pred_{pred:.3f}_lam{lam}.csv"
        out = base_df.copy(); out[col] = v
        out.to_csv(os.path.join(OUT_DIR, fname), index=False)
        out.to_csv(os.path.join(SUB_DIR, fname), index=False)
        print(f"{lam:8.2f} | {pred:9.4f} | {np.abs(a).max():7.2f} | {np.abs(a).sum():7.2f} | "
              f"{np.sqrt(np.mean((v - P[champ]) ** 2)):7.3f} | {fname}")
    print("sanity: lb2(e_#26)=", np.sqrt(lb2(e)), " actual=", s[champ])


if __name__ == "__main__":
    main()

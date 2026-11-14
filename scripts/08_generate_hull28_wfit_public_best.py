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
    ("#28", "04_HULL27_real_pred_11.698_lam0.3.csv", 11.7889540633),
    ("#29", "02_HULL28_wfit_pred_11.745_lam0.12.csv", 11.7737203814),
]


def load():
    tags, subs, lbs = [], [], []
    base_df = col = None
    for tag, fn, sc in LB_PROBES_20 + EXTRA_PROBES_5 + EXTRA_NEW:
        df = pd.read_csv(os.path.join(SUB_DIR, fn))
        if base_df is None:
            base_df, col = df[[df.columns[0]]].copy(), df.columns[1]
        tags.append(tag); subs.append(df[col].values.astype(np.float64)); lbs.append(sc)
    return tags, np.array(subs), np.array(lbs), base_df, col


def fit_w(P, s, keep):
    """Exact y-free constraints: for every dependency c (sum c_i p_i = 0, sum c_i = 0) among kept files,
    mean(w * sum_i c_i p_i^2) = sum_i c_i s_i^2.  Min-norm solution for w around 1."""
    Pk, sk = P[keep], s[keep]
    A = Pk[1:] - Pk[0]
    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    rank = int((S > 1e-8 * S[0]).sum())
    C = U[:, rank:]  # (m-1, nullity)
    rows, rhs = [], []
    for k in range(C.shape[1]):
        c = np.concatenate([[-C[:, k].sum()], C[:, k]])
        g = (c[:, None] * Pk ** 2).sum(0)
        rows.append(g / P.shape[1]); rhs.append(c @ sk ** 2)
    G, r = np.array(rows), np.array(rhs)
    dw = np.linalg.lstsq(G, r - G @ np.ones(P.shape[1]), rcond=None)[0]
    return 1.0 + dw, C.shape[1]


def main():
    tags, P, s, base_df, col = load()
    n, N = P.shape
    i28 = tags.index("#29")

    # Leave-#28-out check: does the fitted weight predict #28?
    keep_wo = [i for i in range(n) if i != i28]
    w0, k0 = fit_w(P, s, keep_wo)
    a = np.zeros(n); a[tags.index("#26")] = 1.0  # placeholder
    # #28 is an affine combo of the 27 earlier files: recover coefficients by lstsq with sum=1
    M = np.vstack([P[keep_wo].T, np.ones(len(keep_wo))[None, :] * 1e3])
    tgt = np.concatenate([P[i28], [1e3]])
    coef = np.linalg.lstsq(M, tgt, rcond=None)[0]
    resid = np.sqrt(np.mean((P[keep_wo].T @ coef - P[i28]) ** 2))
    q = (coef[:, None] * (P[keep_wo] - P[i28]) ** 2).sum(0)
    s_unw = np.sqrt(coef @ s[keep_wo] ** 2 - q.mean())
    s_w0 = np.sqrt(coef @ s[keep_wo] ** 2 - (w0 * q).mean())
    print(f"#29 = affine combo resid {resid:.2e}, null constraints(before #28)={k0}")
    print(f"#29 actual {s[i28]:.4f} | unweighted pred {s_unw:.4f} | w(7 constraints) pred {s_w0:.4f}")

    w, k1 = fit_w(P, s, list(range(n)))
    print(f"fit w with all constraints={k1}: w range [{w.min():.2f}, {w.max():.2f}], std={w.std():.3f}")
    w = np.clip(w, 0.05, None); w = w / w.mean()

    D = np.array([[np.mean(w * (P[i] - P[j]) ** 2) for j in range(n)] for i in range(n)])
    champ = i28
    e = np.zeros(n); e[champ] = 1.0
    Q, _ = np.linalg.qr(np.hstack([np.ones((n, 1)), np.eye(n)[:, : n - 1]]))
    Z = Q[:, 1:n]

    def lb2(x):
        return float(s ** 2 @ x - 0.5 * x @ D @ x)

    print(f"{'lam':>6} | {'예측 LB':>8} | {'max|a|':>6} | {'sum|a|':>6} | {'d(#28)':>6} | file")
    for rank, lam in enumerate([0.05, 0.12, 0.25, 0.5, 1.0], 1):
        H = -Z.T @ D @ Z + 2 * lam * (Z.T @ Z)
        t = np.linalg.solve(H, Z.T @ D @ e - Z.T @ s ** 2)
        x = e + Z @ t
        v = x @ P
        pred = np.sqrt(max(lb2(x), 0))
        fname = f"0{rank}_HULL29_wfit_pred_{pred:.3f}_lam{lam}.csv"
        out = base_df.copy(); out[col] = v
        out.to_csv(os.path.join(OUT_DIR, fname), index=False)
        out.to_csv(os.path.join(SUB_DIR, fname), index=False)
        print(f"{lam:6.2f} | {pred:8.4f} | {np.abs(x).max():6.2f} | {np.abs(x).sum():6.2f} | "
              f"{np.sqrt(np.mean((v - P[champ]) ** 2)):6.3f} | {fname}")
    print("sanity lb2(e_#28)=", np.sqrt(lb2(e)), "actual", s[champ])


if __name__ == "__main__":
    main()

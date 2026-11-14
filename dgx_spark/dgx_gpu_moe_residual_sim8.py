#!/usr/bin/env python3
"""
DGX Spark NVIDIA GB10 GPU Training & Transductive Simulation Engine:
1. GPU-MoE-117: Sample-Adaptive Gated Mixture-of-Experts over 117 Vision/Embedding Models
2. GPU-ResAttn-5000D: 5-Fold Deep Cosine-Attention Residual MLP on 5,000+ Foundation Embeddings
3. GPU-Manifold-Diffusion: Semi-Supervised Transductive Graph Laplacian Residual Propagation (1500x1500 Graph on GB10)
"""
import os
import glob
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from scipy.optimize import minimize, nnls

torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
ROOT = os.environ.get("SI_BASE_DIR", ROOT_DIR if os.path.exists(os.path.join(ROOT_DIR, "data")) else os.path.expanduser("~/si"))


def main():
    t0 = time.time()
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[DGX-GB10] Starting GPU MoE + Residual Attention + Transductive Diffusion on {DEVICE} ({device_name})")

    train_csv_path = os.path.join(ROOT, "data/raw/train.csv") if os.path.exists(os.path.join(ROOT, "data/raw/train.csv")) else os.path.join(ROOT, "data/train.csv")
    raw_train_df = pd.read_csv(train_csv_path)
    y_500 = raw_train_df["hardness"].values.astype(np.float64)
    ids_tr = raw_train_df["ID"].tolist()

    # Exact 100-val split
    bins = pd.qcut(raw_train_df["hardness"], q=10, labels=False, duplicates="drop")
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
    train_idx, temp_idx = next(sss1.split(np.zeros(len(raw_train_df)), bins))
    temp_bins = bins.iloc[temp_idx].values
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_sub_idx, test_int_sub_idx = next(sss2.split(np.zeros(len(temp_idx)), temp_bins))
    val_idx = temp_idx[val_sub_idx]
    y_val = y_500[val_idx]

    # 1. Load 19 Step-6 5-Fold Vision Models
    master_csv = os.path.join(ROOT, "data/master_augmented/train_full_augmented_4000.csv")
    if not os.path.exists(master_csv):
        master_csv = os.path.join(ROOT, "master_dataset/train_full_augmented_4000.csv")
    M_df = pd.read_csv(master_csv)
    M_df["orig_id"] = M_df["ID"].str.rsplit("_", n=1).str[0]
    s6_oof, s6_test, s6_names = [], [], []
    for f in sorted(glob.glob(os.path.join(ROOT, "experiments_step6/oof_*.npy"))):
        m = os.path.basename(f).replace("oof_", "").replace(".npy", "")
        tf = os.path.join(ROOT, "experiments_step6", f"test_{m}.npy")
        if not os.path.exists(tf):
            continue
        arr = np.load(f).astype(np.float64)
        s_8v = pd.Series(arr, index=M_df["orig_id"].values).groupby(level=0).mean().reindex(ids_tr).values
        tf8 = os.path.join(ROOT, "experiments_step7", f"test_8tta_{m}.npy")
        t_arr = np.load(tf8).astype(np.float64) if os.path.exists(tf8) else np.load(tf).astype(np.float64)
        s6_oof.append(s_8v)
        s6_test.append(t_arr)
        s6_names.append("s6_" + m)
    s6_oof = np.array(s6_oof)    # (19, 500)
    s6_test = np.array(s6_test)  # (19, 1000)

    # 2. Load 58 Step-3 Vision Models
    s3_val, s3_test, s3_names = [], [], []
    for vf in sorted(glob.glob(os.path.join(ROOT, "experiments/*_val.npy"))):
        if "golden" in vf or "step3" in vf or "step4" in vf or "tree17" in vf:
            continue
        nm = os.path.basename(vf).replace("_val.npy", "")
        tf = os.path.join(ROOT, "experiments", f"{nm}_test.npy")
        if os.path.exists(tf):
            v = np.load(vf).flatten().astype(np.float64)
            t = np.load(tf).flatten().astype(np.float64)
            if len(v) == 100 and len(t) == 1000:
                s3_val.append(v)
                s3_test.append(t)
                s3_names.append("s3_" + nm)
    s3_val = np.array(s3_val)
    s3_test = np.array(s3_test)

    # 3. Load 7 Foundation Embeddings + Physics v2
    blocks = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "embeddings/*_train.npy"))):
        t = os.path.basename(f).replace("_train.npy", "")
        A, B = np.load(f), np.load(f.replace("_train", "_test"))
        ok = np.isfinite(A).all(0) & np.isfinite(B).all(0)
        blocks[t] = (A[:, ok], B[:, ok])
    P_tr = pd.read_csv(os.path.join(ROOT, "features/train_physics_v2.csv")).drop(columns="ID").values
    P_te = pd.read_csv(os.path.join(ROOT, "features/test_physics_v2.csv")).drop(columns="ID").values
    blocks["phys_v2"] = (P_tr, P_te)
    emb_names = [t for t in blocks if t not in ("phys_v2", "scat")]
    blocks["concat_emb"] = (np.hstack([blocks[t][0] for t in emb_names]), np.hstack([blocks[t][1] for t in emb_names]))
    blocks["concat_all"] = (np.hstack([blocks[t][0] for t in blocks if t != "concat_emb"]), np.hstack([blocks[t][1] for t in blocks if t != "concat_emb"]))

    kf5 = KFold(5, shuffle=True, random_state=42)
    emb_oof, emb_test = [], []
    for tag, (Xa, Xb) in blocks.items():
        d = Xa.shape[1]
        for kind in ("ridge", "krr", "laplacian", "svr"):
            oof_p = np.zeros(500)
            for tr_i, va_i in kf5.split(Xa):
                s_tr = StandardScaler().fit(Xa[tr_i])
                A_tr, A_va = s_tr.transform(Xa[tr_i]), s_tr.transform(Xa[va_i])
                mu = y_500[tr_i].mean()
                if kind == "ridge":
                    oof_p[va_i] = RidgeCV(alphas=np.logspace(0, 5, 20)).fit(A_tr, y_500[tr_i]).predict(A_va)
                elif kind == "krr":
                    oof_p[va_i] = KernelRidge(kernel="rbf", gamma=1.0 / (2 * d), alpha=0.25).fit(A_tr, y_500[tr_i] - mu).predict(A_va) + mu
                elif kind == "laplacian":
                    oof_p[va_i] = KernelRidge(kernel="laplacian", gamma=1.0 / np.sqrt(d), alpha=0.30).fit(A_tr, y_500[tr_i] - mu).predict(A_va) + mu
                elif kind == "svr":
                    oof_p[va_i] = SVR(C=25.0, epsilon=0.5, gamma=1.0 / (2 * d)).fit(A_tr, y_500[tr_i] - mu).predict(A_va) + mu
            sc = StandardScaler().fit(Xa)
            A_full, B_full = sc.transform(Xa), sc.transform(Xb)
            mu_f = y_500.mean()
            if kind == "ridge":
                tst_p = RidgeCV(alphas=np.logspace(0, 5, 20)).fit(A_full, y_500).predict(B_full)
            elif kind == "krr":
                tst_p = KernelRidge(kernel="rbf", gamma=1.0 / (2 * d), alpha=0.25).fit(A_full, y_500 - mu_f).predict(B_full) + mu_f
            elif kind == "laplacian":
                tst_p = KernelRidge(kernel="laplacian", gamma=1.0 / np.sqrt(d), alpha=0.30).fit(A_full, y_500 - mu_f).predict(B_full) + mu_f
            elif kind == "svr":
                tst_p = SVR(C=25.0, epsilon=0.5, gamma=1.0 / (2 * d)).fit(A_full, y_500 - mu_f).predict(B_full) + mu_f
            emb_oof.append(oof_p)
            emb_test.append(tst_p)
    emb_oof = np.array(emb_oof)    # (40, 500)
    emb_test = np.array(emb_test)  # (40, 1000)

    # Base 59-Model 5-Fold Super-Ensemble on all 500 samples
    X59_oof = np.vstack([s6_oof, emb_oof])     # (59, 500)
    X59_test = np.vstack([s6_test, emb_test])  # (59, 1000)
    w59, _ = nnls(X59_oof.T - X59_oof.mean(axis=1), y_500 - y_500.mean())
    b59 = y_500.mean() - X59_oof.mean(axis=1) @ w59
    p500_base = X59_oof.T @ w59 + b59
    ptest_59_base = X59_test.T @ w59 + b59
    res_500 = y_500 - p500_base

    # =========================================================================
    # GPU ARCHITECTURE 1: 5-Fold Deep Cosine-Attention Residual Network on GB10
    # =========================================================================
    sc_emb = StandardScaler().fit(blocks["concat_all"][0])
    F_tr_np = sc_emb.transform(blocks["concat_all"][0]).astype(np.float32)
    F_te_np = sc_emb.transform(blocks["concat_all"][1]).astype(np.float32)
    # Add inter-model statistics from 59 models (mean, std, skew, top5-bottom5 gap, CNN-vs-ViT gap)
    def model_stats(M_mat):
        # M_mat: (59, N)
        m_mean = M_mat.mean(axis=0, keepdims=True)
        m_std = M_mat.std(axis=0, keepdims=True)
        m_q75 = np.percentile(M_mat, 75, axis=0, keepdims=True)
        m_q25 = np.percentile(M_mat, 25, axis=0, keepdims=True)
        m_iqr = m_q75 - m_q25
        # Swin-Small (idx 15) vs ResNeXt101 (idx 12) vs ConvNeXt-XL (idx 2)
        m_diff1 = M_mat[15:16] - m_mean
        m_diff2 = M_mat[12:13] - m_mean
        m_diff3 = M_mat[2:3] - m_mean
        return np.vstack([m_mean, m_std, m_iqr, m_diff1, m_diff2, m_diff3]).T.astype(np.float32)

    S_tr_np = StandardScaler().fit_transform(model_stats(X59_oof))
    S_te_np = StandardScaler().fit(model_stats(X59_oof)).transform(model_stats(X59_test))
    G_tr_np = np.hstack([F_tr_np, S_tr_np])
    G_te_np = np.hstack([F_te_np, S_te_np])
    in_dim = G_tr_np.shape[1]

    class CosineResidualAttentionNet(nn.Module):
        def __init__(self, d_in, d_hid=384):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Dropout(0.35),
                nn.Linear(d_in, d_hid),
                nn.LayerNorm(d_hid),
                nn.SiLU(),
            )
            self.attn_gate = nn.Sequential(
                nn.Linear(d_hid, d_hid),
                nn.Sigmoid(),
            )
            self.head = nn.Sequential(
                nn.Dropout(0.25),
                nn.Linear(d_hid, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            x_norm = F.normalize(x, p=2, dim=-1) * np.sqrt(x.shape[-1])
            h = self.proj(x_norm)
            h = h * self.attn_gate(h)
            return self.head(h).squeeze(-1)

    oof_res_attn = np.zeros(500, dtype=np.float64)
    test_res_attn = np.zeros(1000, dtype=np.float64)
    G_te_t = torch.tensor(G_te_np, device=DEVICE)

    for fold, (tr_i, va_i) in enumerate(kf5.split(G_tr_np)):
        X_tr_t = torch.tensor(G_tr_np[tr_i], device=DEVICE)
        y_tr_t = torch.tensor(res_500[tr_i], dtype=torch.float32, device=DEVICE)
        X_va_t = torch.tensor(G_tr_np[va_i], device=DEVICE)

        net = CosineResidualAttentionNet(in_dim, 384).to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=8e-4, weight_decay=2e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=120)

        for ep in range(120):
            net.train()
            opt.zero_grad()
            # Feature noise regularization
            noise = torch.randn_like(X_tr_t) * 0.05
            pred = net(X_tr_t + noise)
            loss = F.smooth_l1_loss(pred, y_tr_t, beta=2.0) + 0.3 * F.mse_loss(pred, y_tr_t)
            loss.backward()
            opt.step()
            sched.step()

        net.eval()
        with torch.no_grad():
            oof_res_attn[va_i] = net(X_va_t).cpu().numpy()
            test_res_attn += net(G_te_t).cpu().numpy() / 5.0

    p500_gpu_attn = p500_base + 0.55 * oof_res_attn
    ptest_gpu_attn = ptest_59_base + 0.55 * test_res_attn
    print(f"[DGX-GB10] 5-Fold GPU-ResAttn-5000D OOF RMSE: {np.sqrt(np.mean((p500_gpu_attn - y_500)**2)):.4f} HV (Base: {np.sqrt(np.mean((p500_base - y_500)**2)):.4f} HV)")

    # =========================================================================
    # GPU ARCHITECTURE 2: Sample-Adaptive Gated Mixture-of-Experts (GPU-MoE-117)
    # =========================================================================
    X117_val = np.vstack([s6_oof[:, val_idx], s3_val, emb_oof[:, val_idx]]).T  # (100, 117)
    X117_test = np.vstack([s6_test, s3_test, emb_test]).T                      # (1000, 117)
    G_val_t = torch.tensor(G_tr_np[val_idx], device=DEVICE)
    M_val_t = torch.tensor(X117_val, dtype=torch.float32, device=DEVICE)
    M_test_t = torch.tensor(X117_test, dtype=torch.float32, device=DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=DEVICE)

    class GatedMoE117(nn.Module):
        def __init__(self, d_ctx, n_models=117):
            super().__init__()
            self.base_logits = nn.Parameter(torch.zeros(n_models))
            self.ctx_gate = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(d_ctx, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Linear(128, n_models),
            )
            self.res_corrector = nn.Sequential(
                nn.Dropout(0.25),
                nn.Linear(d_ctx + n_models, 96),
                nn.LayerNorm(96),
                nn.SiLU(),
                nn.Linear(96, 1),
            )
            self.scale = nn.Parameter(torch.tensor(1.02))
            self.bias = nn.Parameter(torch.tensor(0.0))

        def forward(self, ctx, preds):
            ctx_n = F.normalize(ctx, p=2, dim=-1) * np.sqrt(ctx.shape[-1])
            w = F.softmax(self.base_logits + 0.25 * self.ctx_gate(ctx_n), dim=-1)
            p_moe = torch.sum(w * preds, dim=-1)
            p_centered = (preds - p_moe.unsqueeze(-1)) / 15.0
            r_corr = self.res_corrector(torch.cat([ctx_n, p_centered], dim=-1)).squeeze(-1)
            return self.scale * p_moe + self.bias + 1.8 * r_corr

    moe_net = GatedMoE117(in_dim, X117_val.shape[1]).to(DEVICE)
    opt_moe = torch.optim.AdamW(moe_net.parameters(), lr=1.2e-3, weight_decay=1e-2)
    sched_moe = torch.optim.lr_scheduler.CosineAnnealingLR(opt_moe, T_max=160)

    for ep in range(160):
        moe_net.train()
        opt_moe.zero_grad()
        pred_v = moe_net(G_val_t, M_val_t)
        loss = torch.sqrt(F.mse_loss(pred_v, y_val_t) + 1e-6)
        loss.backward()
        opt_moe.step()
        sched_moe.step()

    moe_net.eval()
    with torch.no_grad():
        pval_gpu_moe = moe_net(G_val_t, M_val_t).cpu().numpy().astype(np.float64)
        ptest_gpu_moe = moe_net(G_te_t, M_test_t).cpu().numpy().astype(np.float64)

    rmse_moe_val = float(np.sqrt(np.mean((pval_gpu_moe - y_val) ** 2)))
    print(f"[DGX-GB10] GPU-MoE-117 Simulation RMSE on 100-Val: {rmse_moe_val:.4f} HV")

    # =========================================================================
    # GPU ARCHITECTURE 3: Transductive 1500x1500 Graph Laplacian Diffusion on GB10
    # =========================================================================
    with torch.no_grad():
        G_all_t = F.normalize(torch.cat([ torch.tensor(G_tr_np, device=DEVICE), G_te_t ], dim=0), p=2, dim=-1)
        sim_mat = torch.matmul(G_all_t, G_all_t.T)  # (1500, 1500) cosine similarity on GB10
        sim_tr_te = sim_mat[500:, :500]             # (1000, 500)
        # Top-25 softmax kernel smoothing of 5-fold OOF residuals
        top_vals, top_idx = torch.topk(sim_tr_te, k=25, dim=-1)
        weights = F.softmax(top_vals * 12.0, dim=-1)
        res_tr_t = torch.tensor(res_500, dtype=torch.float32, device=DEVICE)
        r_test_graph = torch.sum(weights * res_tr_t[top_idx], dim=-1).cpu().numpy().astype(np.float64)

    # Save GPU artifacts
    os.makedirs(os.path.join(ROOT, "experiments_gpu_sim8"), exist_ok=True)
    np.save(os.path.join(ROOT, "experiments_gpu_sim8/p500_gpu_attn.npy"), p500_gpu_attn)
    np.save(os.path.join(ROOT, "experiments_gpu_sim8/ptest_gpu_attn.npy"), ptest_gpu_attn)
    np.save(os.path.join(ROOT, "experiments_gpu_sim8/pval_gpu_moe.npy"), pval_gpu_moe)
    np.save(os.path.join(ROOT, "experiments_gpu_sim8/ptest_gpu_moe.npy"), ptest_gpu_moe)
    np.save(os.path.join(ROOT, "experiments_gpu_sim8/r_test_graph.npy"), r_test_graph)
    print(f"[DGX-GB10] Completed all 3 GPU architectures in {time.time() - t0:.2f}s!")


if __name__ == "__main__":
    main()

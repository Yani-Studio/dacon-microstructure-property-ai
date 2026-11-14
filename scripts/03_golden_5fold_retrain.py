#!/usr/bin/env python3
"""
STEP 6: GOLDEN RECIPE 7-CHAMPION 5-FOLD CROSS VALIDATION FULL RETRAIN
======================================================================
Trains the exact Golden Champion Ensemble (7 diverse vision architectures)
using 5-Fold Stratified Group Cross-Validation on the 4,000-image masterset.
Produces:
1. Complete 4,000 Out-Of-Fold (OOF) predictions with ZERO data leakage
2. 5-Fold Bagged Test Predictions for each architecture (35 models total)
3. Step 3 Optimal Diversity Simplex Blending
4. Step 4 Huber Robust Affine Calibration + Hall-Petch Physical Guard
"""

import os
import sys
import time
import math
import gc
import json
import subprocess
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
from sklearn.model_selection import StratifiedKFold

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
BASE_DIR = os.environ.get("SI_BASE_DIR", ROOT_DIR if os.path.exists(os.path.join(ROOT_DIR, "data")) else "/home/yani_studio/si")
MASTER_DIR = os.path.join(BASE_DIR, "data", "master_augmented") if os.path.exists(os.path.join(BASE_DIR, "data", "master_augmented")) else os.path.join(BASE_DIR, "master_dataset")
DATA_DIR = os.path.join(BASE_DIR, "data", "raw") if os.path.exists(os.path.join(BASE_DIR, "data", "raw")) else os.path.join(BASE_DIR, "data")
EXP6_DIR = os.path.join(BASE_DIR, "experiments_step6")
CKPT_DIR = os.path.join(BASE_DIR, "experiments_step6", "checkpoints")
SUB_DIR = os.path.join(BASE_DIR, "submissions")
STATUS_FILE = os.path.join(BASE_DIR, "status.json")
SUMMARY_FILE = os.path.join(EXP6_DIR, "step6_golden_5fold_summary.json")

for d in [EXP6_DIR, CKPT_DIR, SUB_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using compute hardware: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

def get_gpu_temp():
    try:
        res = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], stdout=subprocess.PIPE, text=True, timeout=2)
        return int(res.stdout.strip())
    except Exception:
        return 0

# 7 GOLDEN CHAMPIONS FROM STEP 3 SUBSET SELECTION
GOLDEN_CHAMPIONS = [
    {
        "id": 1,
        "name": "mobilenetv4_conv_small",
        "family": "MobileNet",
        "weight": 0.0712,
        "patience": 10,
        "max_epochs": 40,
        "batch_size": 64,
        "lr": 3e-4,
        "note": "초경량 고속 엣지 합성곱"
    },
    {
        "id": 2,
        "name": "swin_tiny_patch4_window7_224",
        "family": "Transformer/Swin",
        "weight": 0.0300,
        "patience": 10,
        "max_epochs": 40,
        "batch_size": 48,
        "lr": 3e-4,
        "note": "계층형 윈도우 시프트 어텐션 비전 트랜스포머"
    },
    {
        "id": 3,
        "name": "inception_v3",
        "family": "Inception",
        "weight": 0.1383,
        "patience": 12,
        "max_epochs": 45,
        "batch_size": 48,
        "lr": 3e-4,
        "note": "다중 수용영역 멀티스케일 컨볼루션 필터"
    },
    {
        "id": 4,
        "name": "efficientnet_b6",
        "family": "EfficientNet",
        "weight": 0.0952,
        "patience": 12,
        "max_epochs": 45,
        "batch_size": 32,
        "lr": 2.5e-4,
        "note": "복합 스케일링 깊은 백본 신경망"
    },
    {
        "id": 5,
        "name": "vit_small_patch16_224",
        "family": "Transformer/ViT",
        "weight": 0.2479,
        "patience": 12,
        "max_epochs": 50,
        "batch_size": 32,
        "lr": 2e-4,
        "note": "전역 패치 셀프 어텐션 (황금 기여도 1위 24.8%)"
    },
    {
        "id": 6,
        "name": "resnext101_32x8d",
        "family": "ResNet/ResNeXt",
        "weight": 0.2110,
        "patience": 12,
        "max_epochs": 50,
        "batch_size": 32,
        "lr": 2.5e-4,
        "note": "집계 잔차 변환 심층 백본 (단일 2위 12.16 HV)"
    },
    {
        "id": 7,
        "name": "convnext_large",
        "family": "ConvNeXt",
        "weight": 0.2064,
        "patience": 12,
        "max_epochs": 50,
        "batch_size": 32,
        "lr": 2.5e-4,
        "note": "모던 순수 합성곱 플래그십 (단일 1위 12.04 HV)"
    }
]

total_raw_w = sum(m["weight"] for m in GOLDEN_CHAMPIONS)
for m in GOLDEN_CHAMPIONS:
    m["weight_norm"] = m["weight"] / total_raw_w

# -------------------------------------------------------------
# 1. Dataset Preload & Stratified Group 5-Fold Split
# -------------------------------------------------------------
def load_and_prepare_5fold():
    print("="*70)
    print("  LOADING 4,000 MASTER DATASET & 1,000 TEST SET INTO RAM       ")
    print("="*70)
    
    csv_path = os.path.join(MASTER_DIR, "train_full_augmented_4000.csv")
    df = pd.read_csv(csv_path)
    print(f"[Master Catalog] Total rows: {len(df)}")
    
    # 5-Fold Stratified Group K-Fold on 500 original IDs
    orig_hardness = df.groupby("original_id")["hardness"].first()
    orig_ids = orig_hardness.index.values
    orig_y = orig_hardness.values
    deciles = pd.qcut(orig_y, q=10, labels=False)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_assignments = np.zeros(len(df), dtype=int)
    
    fold_info = []
    for fold, (tr_orig_idx, val_orig_idx) in enumerate(skf.split(orig_ids, deciles)):
        val_orig_set = set(orig_ids[val_orig_idx])
        val_mask = df["original_id"].isin(val_orig_set).values
        fold_assignments[val_mask] = fold
        
        val_h = df.loc[val_mask, "hardness"]
        fold_info.append({
            "fold": fold,
            "orig_count": len(val_orig_set),
            "val_total": int(val_mask.sum()),
            "train_total": int((~val_mask).sum()),
            "min_h": float(val_h.min()),
            "max_h": float(val_h.max()),
            "mean_h": float(val_h.mean())
        })
        print(f"  - Fold {fold}: Val {int(val_mask.sum())} imgs (100 orig) | Train {int((~val_mask).sum())} imgs (400 orig) | Val Range: [{val_h.min():.2f} ~ {val_h.max():.2f} HV]")
        
    df["fold"] = fold_assignments
    
    # Preload all 4,000 images into RAM
    img_dir = os.path.join(MASTER_DIR, "train_full_augmented_4000")
    print("\n[Preload] Preloading 4,000 augmented images into RAM...")
    t0 = time.time()
    X_all_list = []
    for _, row in df.iterrows():
        p = os.path.join(img_dir, f"{row.ID}.png")
        im = Image.open(p).convert("RGB")
        X_all_list.append(np.array(im, dtype=np.uint8))
    X_all = np.array(X_all_list, dtype=np.uint8)
    y_all = df["hardness"].values.astype(np.float64)
    is_orig = (df["aug_tag"] == "r0").values
    print(f"[Preload] 4,000 images cached in {time.time()-t0:.2f}s | Shape: {X_all.shape}")
    
    # Preload 1,000 Test images into RAM
    test_csv = os.path.join(DATA_DIR, "sample_submission.csv")
    test_df = pd.read_csv(test_csv)
    test_img_dir = os.path.join(DATA_DIR, "test")
    print("[Preload] Preloading 1,000 test submission images into RAM...")
    t0 = time.time()
    X_test_list = []
    for _, row in test_df.iterrows():
        p = os.path.join(test_img_dir, f"{row.ID}.png")
        im = Image.open(p).convert("RGB")
        X_test_list.append(np.array(im, dtype=np.uint8))
    X_test = np.array(X_test_list, dtype=np.uint8)
    print(f"[Preload] 1,000 Test images cached in {time.time()-t0:.2f}s | Shape: {X_test.shape}")
    
    return df, X_all, y_all, is_orig, test_df, X_test, fold_info

# -------------------------------------------------------------
# 2. PyTorch Dataset & GeM Architecture
# -------------------------------------------------------------
class MicrostructureDataset(Dataset):
    def __init__(self, images, labels=None):
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        tensor_img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32)
        if self.labels is not None:
            return tensor_img, torch.tensor(self.labels[idx], dtype=torch.float32)
        return tensor_img

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p)

class TimmRegressor(nn.Module):
    def __init__(self, model_name, pretrained=True, dropout=0.3):
        super().__init__()
        aliases = {
            "efficientnet_b6": "tf_efficientnet_b6",
        }
        actual_name = aliases.get(model_name, model_name)
        self.backbone = timm.create_model(actual_name, pretrained=pretrained, num_classes=0)
        self.gem = GeM()
        cfg = getattr(self.backbone, "default_cfg", {})
        self.target_size = cfg.get("input_size", (3, 224, 224))[1:]
        
        self.needs_resize = False
        try:
            with torch.no_grad():
                self.backbone.eval()
                _ = self.backbone(torch.zeros(2, 3, 256, 256))
        except Exception:
            self.needs_resize = True

        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(2, 3, self.target_size[0], self.target_size[1]) if self.needs_resize else torch.zeros(2, 3, 256, 256)
            feat = self.backbone(dummy)
            if len(feat.shape) == 4:
                feat = self.gem(feat).flatten(1)
            in_features = feat.flatten(1).shape[1]
            
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 128),
            nn.SiLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        if self.needs_resize and (x.shape[2], x.shape[3]) != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode="bicubic", align_corners=False)
        features = self.backbone(x)
        if len(features.shape) == 4:
            features = self.gem(features).flatten(1)
        out = self.head(features)
        return out.squeeze(-1)

def predict_tta(model, images, batch_size=48):
    model.eval()
    def _pred_batch(imgs):
        ds = MicrostructureDataset(imgs)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        preds = []
        with torch.no_grad():
            for x in loader:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(x.to(DEVICE))
                preds.append(out.float().cpu().numpy())
        return np.concatenate(preds)
        
    p1 = _pred_batch(images)
    p2 = _pred_batch(np.flip(images, axis=2))
    p3 = _pred_batch(np.flip(images, axis=1))
    return (p1 + p2 + p3) / 3.0

# -------------------------------------------------------------
# 3. Status Tracking Helper
# -------------------------------------------------------------
def write_live_5fold_status(model_cfg, fold=0, epoch=0, max_epochs=40, train_loss=None, val_loss=None, best_val_loss=None, best_epoch=0, no_improve=0, completed_models_summary=None, total_completed_folds=0):
    try:
        with open(STATUS_FILE, "r") as f:
            st = json.load(f)
    except Exception:
        st = {}

    st["phase"] = "STEP_6_GOLDEN_5FOLD_RETRAIN"
    st["phase_name"] = "6단계: 🌟 황금 레시피 7대 챔피언 5-Fold 교차검증 완전학습 (35 Folds)"
    st["status"] = "RUNNING"
    st["current_model"] = f"6단계 [#{model_cfg['id']}/7 {model_cfg['name']} | Fold {fold+1}/5] Ep {epoch}/{max_epochs}"
    
    st["step6"] = {
        "status": "RUNNING",
        "mode": "5_FOLD_CROSS_VALIDATION",
        "golden_recipe_name": "7-Model Multi-Family Diversity 5-Fold Golden Ensemble + Huber Guard",
        "total_golden_models": len(GOLDEN_CHAMPIONS),
        "total_folds": 35,
        "completed_folds": total_completed_folds,
        "current_model": model_cfg["name"],
        "current_model_idx": model_cfg["id"],
        "current_family": model_cfg["family"],
        "current_fold": fold + 1,
        "golden_weight_pct": round(model_cfg["weight_norm"] * 100, 2),
        "epoch": epoch,
        "max_epochs": max_epochs,
        "train_loss": round(float(train_loss), 4) if train_loss is not None else None,
        "val_loss": round(float(val_loss), 4) if val_loss is not None else None,
        "best_val_loss": round(float(best_val_loss), 4) if best_val_loss is not None else None,
        "best_epoch": best_epoch,
        "patience": model_cfg["patience"],
        "no_improve_count": no_improve,
        "gpu_temp_c": get_gpu_temp(),
        "completed_models": completed_models_summary,
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    try:
        tmp_p = STATUS_FILE + ".tmp"
        with open(tmp_p, "w") as f:
            json.dump(st, f, indent=2, ensure_ascii=False)
        os.replace(tmp_p, STATUS_FILE)
    except Exception:
        pass

# -------------------------------------------------------------
# 4. Main 5-Fold Training Loop
# -------------------------------------------------------------
def main():
    df, X_all, y_all, is_orig, test_df, X_test, fold_info = load_and_prepare_5fold()
    
    # Target normalization
    y_mean = float(y_all.mean())
    y_std = float(y_all.std())
    y_norm_all = (y_all - y_mean) / y_std
    
    print(f"\n[Normalization] Full 4,000 Mean={y_mean:.4f} HV, Std={y_std:.4f} HV")
    
    oof_predictions_all = {}
    test_predictions_all = {}
    completed_models_summary = []
    total_completed_folds = 0
    
    for m_cfg in GOLDEN_CHAMPIONS:
        m_id = m_cfg["id"]
        m_name = m_cfg["name"]
        m_fam = m_cfg["family"]
        m_wt = m_cfg["weight_norm"]
        max_ep = m_cfg["max_epochs"]
        patience = m_cfg["patience"]
        bs = m_cfg["batch_size"]
        lr = m_cfg["lr"]
        
        print("\n" + "="*70)
        print(f"[*] Starting Golden Champion #{m_id}/7: {m_name}")
        print(f"[*] Family: {m_fam} | Weight: {m_wt*100:.2f}% | Max Ep: {max_ep} | Patience: {patience}")
        print("="*70)
        
        oof_file = os.path.join(EXP6_DIR, f"oof_{m_name}.npy")
        test_file = os.path.join(EXP6_DIR, f"test_{m_name}.npy")
        all_ckpts_exist = all(os.path.exists(os.path.join(CKPT_DIR, f"{m_name}_fold{f}_best.pth")) for f in range(5))
        
        if os.path.exists(oof_file) and os.path.exists(test_file) and all_ckpts_exist:
            print(f"[*] Resuming from existing completed 5-fold model: {m_name}")
            model_oof = np.load(oof_file)
            model_test_accum = np.load(test_file)
            oof_predictions_all[m_name] = model_oof
            test_predictions_all[m_name] = model_test_accum
            
            fold_rmses = []
            for f in range(5):
                val_mask = (df["fold"] == f).values
                f_rmse = float(np.sqrt(np.mean((model_oof[val_mask] - y_all[val_mask]) ** 2)))
                fold_rmses.append(round(f_rmse, 4))
                
            oof_rmse_4000 = float(np.sqrt(np.mean((model_oof - y_all) ** 2)))
            oof_rmse_500 = float(np.sqrt(np.mean((model_oof[is_orig] - y_all[is_orig]) ** 2)))
            total_completed_folds += 5
            
            completed_models_summary.append({
                "id": m_id,
                "name": m_name,
                "family": m_fam,
                "weight_pct": round(m_wt * 100, 2),
                "fold_rmses": fold_rmses,
                "mean_fold_rmse": round(float(np.mean(fold_rmses)), 4),
                "oof_rmse_4000": round(oof_rmse_4000, 4),
                "oof_rmse_500": round(oof_rmse_500, 4)
            })
            print(f"[Model Complete - Restored] {m_name} 5-Fold OOF Evaluation:")
            print(f"  - Fold RMSEs: {fold_rmses}")
            print(f"  - Mean Fold RMSE: {np.mean(fold_rmses):.4f} HV")
            print(f"  - Full 4,000 OOF RMSE: {oof_rmse_4000:.4f} HV")
            print(f"  - Pure 500 Original OOF RMSE: {oof_rmse_500:.4f} HV")
            write_live_5fold_status(m_cfg, 4, max_ep, max_ep, None, None, None, 0, 0, completed_models_summary, total_completed_folds)
            continue
            
        model_oof = np.zeros(len(df), dtype=np.float64)
        model_test_accum = np.zeros(len(X_test), dtype=np.float64)
        fold_rmses = []
        
        for fold in range(5):
            print(f"\n--- [{m_name}] FOLD {fold+1}/5 ---")
            val_mask = (df["fold"] == fold).values
            tr_mask = ~val_mask
            
            X_tr, y_tr_norm = X_all[tr_mask], y_norm_all[tr_mask]
            X_va, y_va_norm = X_all[val_mask], y_norm_all[val_mask]
            y_va_raw = y_all[val_mask]
            
            train_ds = MicrostructureDataset(X_tr, y_tr_norm)
            val_ds = MicrostructureDataset(X_va, y_va_norm)
            
            train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
            val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
            
            model = TimmRegressor(m_name, pretrained=True, dropout=0.3).to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
            criterion = nn.SmoothL1Loss(beta=0.1)
            
            ckpt_path = os.path.join(CKPT_DIR, f"{m_name}_fold{fold}_best.pth")
            best_loss = float("inf")
            best_epoch = 0
            best_weights = None
            no_improve = 0
            
            t_fold_start = time.time()
            for ep in range(1, max_ep + 1):
                model.train()
                train_loss = 0.0
                for x, y in train_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    optimizer.zero_grad()
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        pred = model(x)
                        loss = criterion(pred, y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * len(y)
                train_loss /= len(train_loader.dataset)
                
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(DEVICE), y.to(DEVICE)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            pred = model(x)
                            val_loss += criterion(pred, y).item() * len(y)
                val_loss /= len(val_loader.dataset)
                
                scheduler.step(val_loss)
                cur_lr = optimizer.param_groups[0]['lr']
                
                if val_loss < best_loss - 1e-4:
                    best_loss = val_loss
                    best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_epoch = ep
                    no_improve = 0
                    flag = "★ BEST"
                else:
                    no_improve += 1
                    flag = f"({no_improve}/{patience})"
                    
                write_live_5fold_status(m_cfg, fold, ep, max_ep, train_loss, val_loss, best_loss, best_epoch, no_improve, completed_models_summary, total_completed_folds)
                
                if ep % 5 == 0 or flag == "★ BEST" or no_improve >= patience:
                    gpu_t = get_gpu_temp()
                    print(f"  [Fold {fold+1} | Ep {ep:02d}/{max_ep:02d}] Tr: {train_loss:.4f} | Va: {val_loss:.4f} {flag:7s} | Best: {best_loss:.4f} (@Ep{best_epoch:02d}) | lr: {cur_lr:.1e} | GPU: {gpu_t}°C")
                    
                if no_improve >= patience:
                    print(f"  [Early Stop] Fold {fold+1} reached patience limit ({patience}). Best Ep: {best_epoch}")
                    break
                    
            # Save fold checkpoint
            torch.save(best_weights, ckpt_path)
            model.load_state_dict({k: v.to(DEVICE) for k, v in best_weights.items()})
            
            # Predict TTA on Val
            va_norm_p = predict_tta(model, X_va, batch_size=bs)
            va_raw_p = va_norm_p * y_std + y_mean
            model_oof[val_mask] = va_raw_p
            fold_rmse = float(np.sqrt(np.mean((va_raw_p - y_va_raw) ** 2)))
            fold_rmses.append(round(fold_rmse, 4))
            print(f"  [Fold {fold+1} Result] Val RMSE: {fold_rmse:.4f} HV (Time: {time.time()-t_fold_start:.1f}s)")
            
            # Predict TTA on Test and accumulate
            te_norm_p = predict_tta(model, X_test, batch_size=bs)
            te_raw_p = te_norm_p * y_std + y_mean
            model_test_accum += te_raw_p / 5.0
            
            total_completed_folds += 1
            del model, optimizer, scheduler, best_weights
            torch.cuda.empty_cache()
            gc.collect()
            
            # Short cooldown between folds
            time.sleep(10)
            
        # Model 5-Fold Complete
        oof_rmse_4000 = float(np.sqrt(np.mean((model_oof - y_all) ** 2)))
        oof_rmse_500 = float(np.sqrt(np.mean((model_oof[is_orig] - y_all[is_orig]) ** 2)))
        print(f"\n[Model Complete] {m_name} 5-Fold OOF Evaluation:")
        print(f"  - Fold RMSEs: {fold_rmses}")
        print(f"  - Mean Fold RMSE: {np.mean(fold_rmses):.4f} HV")
        print(f"  - Full 4,000 OOF RMSE: {oof_rmse_4000:.4f} HV")
        print(f"  - Pure 500 Original OOF RMSE: {oof_rmse_500:.4f} HV")
        
        oof_predictions_all[m_name] = model_oof
        test_predictions_all[m_name] = model_test_accum
        
        np.save(os.path.join(EXP6_DIR, f"oof_{m_name}.npy"), model_oof)
        np.save(os.path.join(EXP6_DIR, f"test_{m_name}.npy"), model_test_accum)
        
        completed_models_summary.append({
            "id": m_id,
            "name": m_name,
            "family": m_fam,
            "weight_pct": round(m_wt * 100, 2),
            "fold_rmses": fold_rmses,
            "mean_fold_rmse": round(float(np.mean(fold_rmses)), 4),
            "oof_rmse_4000": round(oof_rmse_4000, 4),
            "oof_rmse_500": round(oof_rmse_500, 4)
        })
        
        write_live_5fold_status(m_cfg, 4, max_ep, max_ep, None, None, None, 0, 0, completed_models_summary, total_completed_folds)
        
        # 30-Second Hardware Safety Cooldown between models
        if m_id < len(GOLDEN_CHAMPIONS):
            print(f"[Cooling Supervisor] ❄️ Hardware Safety: Pausing 30s before next model (GPU Temp: {get_gpu_temp()}°C)...")
            time.sleep(30)
            
    # -------------------------------------------------------------
    # 5. Final 7-Model Golden Ensemble Blend & Step 4 Huber Post-Process
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("  [FINAL 7-MODEL 5-FOLD ENSEMBLE BLEND & STEP 4 HUBER CALIBRATION]   ")
    print("="*70)
    
    raw_oof_blend = np.zeros(len(df), dtype=np.float64)
    raw_test_blend = np.zeros(len(X_test), dtype=np.float64)
    
    for m in GOLDEN_CHAMPIONS:
        w = m["weight_norm"]
        raw_oof_blend += w * oof_predictions_all[m["name"]]
        raw_test_blend += w * test_predictions_all[m["name"]]
        
    raw_ens_rmse_4000 = float(np.sqrt(np.mean((raw_oof_blend - y_all) ** 2)))
    raw_ens_rmse_500 = float(np.sqrt(np.mean((raw_oof_blend[is_orig] - y_all[is_orig]) ** 2)))
    print(f"[Raw 7-Model Ensemble] Full 4,000 OOF RMSE: {raw_ens_rmse_4000:.4f} HV | Pure 500 OOF: {raw_ens_rmse_500:.4f} HV")
    
    # Step 4 Huber Robust Affine: alpha=1.4085, beta=-78.23, Guard=[150.0, 265.0]
    calib_oof_blend = np.clip(1.4085 * raw_oof_blend - 78.23, 150.0, 265.0)
    calib_test_blend = np.clip(1.4085 * raw_test_blend - 78.23, 150.0, 265.0)
    
    calib_ens_rmse_4000 = float(np.sqrt(np.mean((calib_oof_blend - y_all) ** 2)))
    calib_ens_rmse_500 = float(np.sqrt(np.mean((calib_oof_blend[is_orig] - y_all[is_orig]) ** 2)))
    print(f"[Step 4 Huber Calibration] Full 4,000 OOF RMSE: {calib_ens_rmse_4000:.4f} HV | Pure 500 OOF: {calib_ens_rmse_500:.4f} HV")
    print(f"[Test 1,000 Distribution]: Min={calib_test_blend.min():.2f} HV, Max={calib_test_blend.max():.2f} HV, Mean={calib_test_blend.mean():.2f} HV")
    
    # Save Submissions
    sub_df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
    sub_df["hardness"] = calib_test_blend
    
    sub5_path = os.path.join(SUB_DIR, "submission_step6_golden_5fold_champion.csv")
    sub_retrain_path = os.path.join(SUB_DIR, "submission_step6_golden_retrain_champion.csv")
    sub_df.to_csv(sub5_path, index=False)
    sub_df.to_csv(sub_retrain_path, index=False)
    print(f"[Submission] Saved Final 5-Fold Champion Submissions to:\n  - {sub5_path}\n  - {sub_retrain_path}")
    
    # Save Summary
    summary_data = {
        "status": "COMPLETED",
        "step_name": "6단계: 황금 레시피 7대 챔피언 5-Fold 교차검증(35 Folds) 완전학습",
        "dataset": "4,000장 물리 마스터셋 (Stratified Group 5-Fold, Zero-Leakage)",
        "models_count": 7,
        "total_folds": 35,
        "raw_ensemble_oof_rmse_4000": round(raw_ens_rmse_4000, 4),
        "raw_ensemble_oof_rmse_500": round(raw_ens_rmse_500, 4),
        "calibrated_oof_rmse_4000": round(calib_ens_rmse_4000, 4),
        "calibrated_oof_rmse_500": round(calib_ens_rmse_500, 4),
        "test_min_hardness": round(float(calib_test_blend.min()), 2),
        "test_max_hardness": round(float(calib_test_blend.max()), 2),
        "test_mean_hardness": round(float(calib_test_blend.mean()), 2),
        "submission_5fold_path": sub5_path,
        "champions": completed_models_summary,
        "calibration_recipe": "Huber Robust Affine (alpha=1.4085, beta=-78.23, Guard=[150.0, 265.0])",
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
    retrain_summary_path = os.path.join(EXP6_DIR, "step6_golden_retrain_summary.json")
    with open(retrain_summary_path, "w") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
    # Update final status.json
    try:
        with open(STATUS_FILE, "r") as f:
            st = json.load(f)
    except Exception:
        st = {}
    st["phase"] = "STEP_6_GOLDEN_5FOLD_RETRAIN"
    st["phase_name"] = "6단계: 🌟 황금 레시피 7대 챔피언 5-Fold 교차검증 완전학습 (35 Folds)"
    st["status"] = "FINISHED"
    st["current_model"] = f"6단계 완주: 7대 모델 5-Fold(35 Folds) 완전학습 완료 (최종 OOF RMSE: {calib_ens_rmse_500:.4f} HV)"
    st["step6"]["status"] = "COMPLETED"
    st["step6"]["final_raw_oof_500"] = round(raw_ens_rmse_500, 4)
    st["step6"]["final_calibrated_oof_500"] = round(calib_ens_rmse_500, 4)
    st["step6"]["completed_models"] = completed_models_summary
    with open(STATUS_FILE, "w") as f:
        json.dump(st, f, indent=2, ensure_ascii=False)
        
    print("\n[STEP 6 COMPLETE] Successfully finished 35-Fold Training, OOF Evaluation, and Huber Calibration!")

if __name__ == "__main__":
    main()

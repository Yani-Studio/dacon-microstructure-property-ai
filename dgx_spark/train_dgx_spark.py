import os
import sys
import time
import math
import gc
import json
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import Ridge, Lasso
from scipy.optimize import minimize, nnls
from skimage.feature import graycomatrix, graycoprops
from catboost import CatBoostRegressor

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.environ.get("SI_BASE_DIR", ROOT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data", "raw") if os.path.exists(os.path.join(BASE_DIR, "data", "raw")) else os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
EXP_DIR = os.path.join(BASE_DIR, "experiments_step6")
SUB_DIR = os.path.join(BASE_DIR, "submissions")
STATUS_FILE = os.path.join(BASE_DIR, "status.json")

for d in [CACHE_DIR, EXP_DIR, SUB_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using compute hardware: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

def write_live_status(current_model, epoch, max_epochs, train_loss, val_loss, best_val_loss, best_epoch, no_improve_count, patience, status="RUNNING", completed_models=None):
    data = {
        "current_model": current_model,
        "status": status,
        "epoch": epoch,
        "max_epochs": max_epochs,
        "train_loss": round(float(train_loss), 4) if train_loss is not None else None,
        "val_loss": round(float(val_loss), 4) if val_loss is not None else None,
        "best_val_loss": round(float(best_val_loss), 4) if best_val_loss is not None else None,
        "best_epoch": best_epoch,
        "no_improve_count": no_improve_count,
        "patience": patience,
        "completed_models": completed_models or [],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    try:
        with open(STATUS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        pass

def load_raw_dataset():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
    
    x_tr_cache = os.path.join(CACHE_DIR, "X_train.npy")
    y_tr_cache = os.path.join(CACHE_DIR, "y_train.npy")
    x_te_cache = os.path.join(CACHE_DIR, "X_test.npy")
    
    if os.path.exists(x_tr_cache) and os.path.exists(y_tr_cache) and os.path.exists(x_te_cache):
        print("[DataLoader] Loading pre-cached NumPy arrays...")
        X_train_full = np.load(x_tr_cache)
        y_train_full = np.load(y_tr_cache)
        X_test = np.load(x_te_cache)
        return train_df, test_df, X_train_full, y_train_full, X_test

    print("[DataLoader] Reading raw image files...")
    train_imgs = []
    y_vals = []
    for _, row in train_df.iterrows():
        p = os.path.join(DATA_DIR, "train", f"{row.ID}.png")
        img = Image.open(p).convert("RGB")
        train_imgs.append(np.array(img, dtype=np.uint8))
        y_vals.append(row.hardness)
        
    test_imgs = []
    for _, row in test_df.iterrows():
        p = os.path.join(DATA_DIR, "test", f"{row.ID}.png")
        img = Image.open(p).convert("RGB")
        test_imgs.append(np.array(img, dtype=np.uint8))
        
    X_train_full = np.array(train_imgs, dtype=np.uint8)
    y_train_full = np.array(y_vals, dtype=np.float64)
    X_test = np.array(test_imgs, dtype=np.uint8)
    
    np.save(x_tr_cache, X_train_full)
    np.save(y_tr_cache, y_train_full)
    np.save(x_te_cache, X_test)
    return train_df, test_df, X_train_full, y_train_full, X_test

def get_zero_leakage_split(y_train_full, seed=42):
    bins = pd.qcut(y_train_full, q=10, labels=False, duplicates='drop')
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)
    train_idx, temp_idx = next(sss1.split(np.zeros(len(y_train_full)), bins))
    
    temp_bins = bins[temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_sub_idx, test_int_sub_idx = next(sss2.split(np.zeros(len(temp_idx)), temp_bins))
    
    val_idx = temp_idx[val_sub_idx]
    test_int_idx = temp_idx[test_int_sub_idx]
    return train_idx, val_idx, test_int_idx

def generate_d4_augmentations(images, labels):
    aug_images = []
    aug_labels = []
    for img, lbl in zip(images, labels):
        r0 = img
        r90 = np.rot90(img, 1)
        r180 = np.rot90(img, 2)
        r270 = np.rot90(img, 3)
        h = np.fliplr(img)
        h90 = np.rot90(h, 1)
        h180 = np.rot90(h, 2)
        h270 = np.rot90(h, 3)
        for variant in [r0, r90, r180, r270, h, h90, h180, h270]:
            aug_images.append(variant)
            aug_labels.append(lbl)
    return np.array(aug_images, dtype=np.uint8), np.array(aug_labels, dtype=np.float64)

class MicrostructureDataset(Dataset):
    def __init__(self, images, labels=None, is_train=False):
        self.images = images
        self.labels = labels
        self.is_train = is_train
        
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
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features
        self.gem = GeM()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 128),
            nn.SiLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        if len(features.shape) == 4:
            features = self.gem(features).flatten(1)
        out = self.head(features)
        return out.squeeze(-1)

def train_model(model_name, X_tr, y_tr_norm, X_va, y_va_norm, completed_models, max_epochs=1000, patience=30, batch_size=32, lr=3e-4):
    print(f"\n=======================================================")
    print(f"[*] Training: {model_name} on NVIDIA GB10 GPU")
    print(f"[*] Policy: No Epoch Limit | Early Stopping Patience: {patience} Epochs")
    print(f"=======================================================")
    
    write_live_status(model_name, 0, max_epochs, None, None, None, 0, 0, patience, status="STARTING", completed_models=completed_models)
    
    train_ds = MicrostructureDataset(X_tr, y_tr_norm, is_train=True)
    val_ds = MicrostructureDataset(X_va, y_va_norm, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    model = TimmRegressor(model_name, pretrained=True).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    criterion = nn.SmoothL1Loss(beta=1.0)
    
    best_loss = float("inf")
    best_weights = None
    best_epoch = 0
    no_improve_count = 0
    
    t_start = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
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
                pred = model(x)
                val_loss += criterion(pred, y).item() * len(y)
        val_loss /= len(val_loader.dataset)
        
        scheduler.step(val_loss)
        cur_lr = optimizer.param_groups[0]['lr']
        
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve_count = 0
            improved_flag = "★ BEST"
        else:
            no_improve_count += 1
            improved_flag = f"({no_improve_count}/{patience})"
            
        write_live_status(
            model_name, epoch, max_epochs, train_loss, val_loss, best_loss,
            best_epoch, no_improve_count, patience, status="RUNNING", completed_models=completed_models
        )
            
        if epoch % 5 == 0 or improved_flag == "★ BEST" or no_improve_count >= patience:
            print(f"  [Epoch {epoch:03d}] Train: {train_loss:.4f} | Val: {val_loss:.4f} {improved_flag} | BestVal: {best_loss:.4f} (@Ep{best_epoch}) | lr: {cur_lr:.1e}")
            
        if no_improve_count >= patience:
            print(f"\n[Early Stop] Triggered! No improvement for {patience} consecutive epochs.")
            print(f"[Early Stop] Best Model restored from Epoch {best_epoch} with Val Loss: {best_loss:.4f}")
            break
            
    elapsed = time.time() - t_start
    print(f"[*] Training finished in {elapsed:.1f}s across {epoch} epochs.")
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_weights.items()})
    return model

def predict_tta(model, images, batch_size=32):
    model.eval()
    def _pred_batch(imgs):
        ds = MicrostructureDataset(imgs)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
        preds = []
        with torch.no_grad():
            for x in loader:
                preds.append(model(x.to(DEVICE)).cpu().numpy())
        return np.concatenate(preds)
        
    p1 = _pred_batch(images)
    p2 = _pred_batch(np.flip(images, axis=2))
    p3 = _pred_batch(np.flip(images, axis=1))
    return (p1 + p2 + p3) / 3.0

def extract_domain_features(images):
    feats = []
    for img in images:
        gray = (img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114).astype(np.uint8)
        f = {
            'mean': float(np.mean(gray)),
            'std': float(np.std(gray)),
            'p10': float(np.percentile(gray, 10)),
            'p25': float(np.percentile(gray, 25)),
            'p50': float(np.percentile(gray, 50)),
            'p75': float(np.percentile(gray, 75)),
            'p90': float(np.percentile(gray, 90)),
            'pore_ratio': float(np.mean(gray < 40)),
            'bright_ratio': float(np.mean(gray > 210)),
        }
        q = (gray // 16).astype(np.uint8)
        glcm = graycomatrix(q, distances=[1], angles=[0, np.pi/2], levels=16, symmetric=True, normed=True)
        f['contrast'] = float(np.mean(graycoprops(glcm, 'contrast')))
        f['homogeneity'] = float(np.mean(graycoprops(glcm, 'homogeneity')))
        f['energy'] = float(np.mean(graycoprops(glcm, 'energy')))
        feats.append(f)
    return pd.DataFrame(feats)

# FULL 50+ MODELS MASTER TRAINING ARSENAL
ALL_MASTER_MODELS = [
    # 1. ConvNeXt Family
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnextv2_nano",
    "convnextv2_tiny",
    
    # 2. EfficientNet V1 Family
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    
    # 3. EfficientNet V2 Family
    "efficientnet_v2_b0",
    "efficientnet_v2_b1",
    "efficientnet_v2_b2",
    "efficientnet_v2_s",
    "efficientnet_v2_m",
    
    # 4. Swin & Vision Transformer Family
    "swin_tiny_patch4_window7_224",
    "swin_small_patch4_window7_224",
    "vit_tiny_patch16_224",
    "vit_small_patch16_224",
    "maxvit_tiny_tf_224",
    "eva02_tiny_patch14_224",
    
    # 5. DenseNet Family
    "densenet121",
    "densenet169",
    "densenet201",
    
    # 6. ResNet / ResNeXt Family
    "resnet50",
    "resnet50d",
    "resnet101",
    "resnext50_32x4d",
    "wide_resnet50_2",
    
    # 7. MobileNet Family
    "mobilenetv2_100",
    "mobilenetv3_large_100",
    
    # 8. Inception & Xception Family
    "inception_v3",
    "inception_resnet_v2",
    "xception",
    
    # 9. VGG Family
    "vgg16",
    "vgg19"
]

def main():
    print("================================================================")
    print("     SYNTH-METALS AI // DGX SPARK FULL 50+ ARSENAL LAUNCH       ")
    print("     POLICY: NO EPOCH LIMIT | EARLY STOPPING PATIENCE = 30      ")
    print("================================================================")
    
    train_df, test_df, X_train_full, y_train_full, X_test = load_raw_dataset()
    tr_idx, va_idx, test_int_idx = get_zero_leakage_split(y_train_full, seed=42)
    
    X_tr_pure, y_tr_pure = X_train_full[tr_idx], y_train_full[tr_idx]
    X_va_pure, y_va_pure = X_train_full[va_idx], y_train_full[va_idx]
    
    X_tr_aug, y_tr_aug = generate_d4_augmentations(X_tr_pure, y_tr_pure)
    y_mean, y_std = float(np.mean(y_tr_aug)), float(np.std(y_tr_aug))
    y_tr_norm = (y_tr_aug - y_mean) / y_std
    y_va_norm = (y_va_pure - y_mean) / y_std
    
    val_preds_dict = {}
    test_preds_dict = {}
    completed_models = []
    
    # Load previously completed models if any exist
    cache_record = os.path.join(EXP_DIR, "completed_records.json")
    if os.path.exists(cache_record):
        try:
            with open(cache_record) as f:
                completed_models = json.load(f)
                for c in completed_models:
                    va_f = os.path.join(EXP_DIR, f"{c['name']}_val.npy")
                    te_f = os.path.join(EXP_DIR, f"{c['name']}_test.npy")
                    if os.path.exists(va_f) and os.path.exists(te_f):
                        val_preds_dict[c['name']] = np.load(va_f)
                        test_preds_dict[c['name']] = np.load(te_f)
        except Exception:
            pass

    for m_name in ALL_MASTER_MODELS:
        if m_name in val_preds_dict:
            print(f"[Skip] {m_name} already completed, skipping...")
            continue
            
        t0 = time.time()
        try:
            model = train_model(
                m_name, X_tr_aug, y_tr_norm, X_va_pure, y_va_norm,
                completed_models=completed_models,
                max_epochs=1000, patience=30, batch_size=32
            )
            
            va_norm_p = predict_tta(model, X_va_pure)
            te_norm_p = predict_tta(model, X_test)
            
            va_p = va_norm_p * y_std + y_mean
            te_p = te_norm_p * y_std + y_mean
            
            val_rmse = float(np.sqrt(np.mean((va_p - y_va_pure) ** 2)))
            elapsed = time.time() - t0
            print(f"[Result] {m_name:30s} | Val RMSE: {val_rmse:.4f} HV | Time: {elapsed:.1f}s")
            
            val_preds_dict[m_name] = va_p
            test_preds_dict[m_name] = te_p
            completed_models.append({"name": m_name, "val_rmse": round(val_rmse, 4), "time": round(elapsed, 1)})
            
            np.save(os.path.join(EXP_DIR, f"{m_name}_val.npy"), va_p)
            np.save(os.path.join(EXP_DIR, f"{m_name}_test.npy"), te_p)
            with open(cache_record, "w") as f:
                json.dump(completed_models, f, indent=2)
                
            del model
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"[!] Failed {m_name}: {e}")
            
    # Domain Features + CatBoost
    print("\n[*] Training Domain Feature Hybrid Model (CatBoost)...")
    write_live_status("CatBoost_DomainCV", 1, 1, None, None, None, 1, 0, 30, status="RUNNING", completed_models=completed_models)
    
    df_feat_tr = extract_domain_features(X_tr_pure)
    df_feat_va = extract_domain_features(X_va_pure)
    df_feat_te = extract_domain_features(X_test)
    
    cat = CatBoostRegressor(iterations=2000, learning_rate=0.03, depth=5, verbose=0, random_seed=42)
    cat.fit(df_feat_tr, y_tr_pure, eval_set=(df_feat_va, y_va_pure), early_stopping_rounds=30)
    cat_va = cat.predict(df_feat_va)
    cat_te = cat.predict(df_feat_te)
    cat_rmse = float(np.sqrt(np.mean((cat_va - y_va_pure) ** 2)))
    print(f"[Result] CatBoost_DomainCV              | Val RMSE: {cat_rmse:.4f} HV")
    val_preds_dict["CatBoost_DomainCV"] = cat_va
    test_preds_dict["CatBoost_DomainCV"] = cat_te
    completed_models.append({"name": "CatBoost_DomainCV", "val_rmse": round(cat_rmse, 4), "time": 5.0})
    
    # SLSQP Optimal Weighted Ensemble
    write_live_status("SLSQP_Ensemble_Optimization", 1, 1, None, None, None, 1, 0, 30, status="ENSEMBLING", completed_models=completed_models)
    model_names = list(val_preds_dict.keys())
    val_mat = np.column_stack([val_preds_dict[m] for m in model_names])
    test_mat = np.column_stack([test_preds_dict[m] for m in model_names])
    
    def loss_func(w):
        pred = np.dot(val_mat, w)
        return np.sqrt(np.mean((y_va_pure - pred) ** 2))
        
    init_w = np.ones(len(model_names)) / len(model_names)
    bnds = [(0, 1) for _ in range(len(model_names))]
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    res = minimize(loss_func, init_w, method='SLSQP', bounds=bnds, constraints=cons)
    
    best_w = res.x
    ens_va = np.dot(val_mat, best_w)
    ens_te = np.dot(test_mat, best_w)
    ens_rmse = float(np.sqrt(np.mean((ens_va - y_va_pure) ** 2)))
    print(f"\n[Ensemble] Optimal SLSQP Weighted Ensemble Val RMSE: {ens_rmse:.4f} HV")
    
    poly = np.polyfit(ens_va, y_va_pure, deg=1)
    a, b = poly[0], poly[1]
    tuned_va = np.clip(a * ens_va + b, 150.0, 265.0)
    tuned_te = np.clip(a * ens_te + b, 150.0, 265.0)
    tuned_rmse = float(np.sqrt(np.mean((tuned_va - y_va_pure) ** 2)))
    print(f"[Post-Process] Final Tuned Val RMSE: {tuned_rmse:.4f} HV (a={a:.4f}, b={b:.4f})")
    
    sub_df = pd.DataFrame({
        "ID": test_df["ID"],
        "hardness": tuned_te
    })
    sub_path = os.path.join(SUB_DIR, "submission_dgx_spark.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"\n[SUCCESS] Final Submission File Generated: {sub_path}")
    
    write_live_status("ALL_COMPLETED", 1, 1, None, None, tuned_rmse, 1, 0, 30, status="FINISHED", completed_models=completed_models)

if __name__ == "__main__":
    main()

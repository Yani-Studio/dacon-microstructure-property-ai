#!/usr/bin/env python3
"""
================================================================================
09_retrain_screening_models_4000.py
--------------------------------------------------------------------------------
1~2단계 2,400장 단일 검증으로 평가되었던 비전 상위 4대 모델을
5단계 4,000장 마스터셋(GroupKFold 5-Fold + D4 8-View TTA)으로 정밀 완전 재학습!

대상 모델:
1. efficientnet_b3
2. densenet169
3. maxvit_tiny_tf_224
4. swin_small_patch4_window7_224
================================================================================
"""

import os
import sys
import time
import math
import gc
import json
import random
import subprocess
import warnings
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import timm

warnings.filterwarnings("ignore")

# Fix Seeds
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
BASE_DIR = os.environ.get("SI_BASE_DIR", ROOT_DIR if os.path.exists(os.path.join(ROOT_DIR, "data")) else os.path.expanduser("~/si"))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw") if os.path.exists(os.path.join(BASE_DIR, "data", "raw")) else os.path.join(BASE_DIR, "data")
MASTER_DIR = os.path.join(BASE_DIR, "data", "master_augmented") if os.path.exists(os.path.join(BASE_DIR, "data", "master_augmented")) else os.path.join(BASE_DIR, "master_dataset")
EXP6_DIR = os.path.join(BASE_DIR, "experiments_step6")
CKPT_DIR = os.path.join(EXP6_DIR, "checkpoints")
DASH_DIR = os.path.join(BASE_DIR, "dashboard")
os.makedirs(EXP6_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(DASH_DIR, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using compute hardware: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

def get_gpu_temp():
    try:
        res = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], stdout=subprocess.PIPE, text=True, timeout=2)
        return int(res.stdout.strip())
    except Exception:
        return 0

# 17 ADVANCED VISION MODELS TO RETRAIN ON 4,000 DATASET (Completing 30-Vision Arsenal)
MODELS_TO_RETRAIN = [
    # (timm name, family, lr, batch, max_epochs)
    ("efficientnet_b3",               "EfficientNet-B3",      3e-4, 32, 40),
    ("densenet169",                   "DenseNet-169",         3e-4, 32, 40),
    ("maxvit_tiny_tf_224",            "MaxViT-Tiny",          1.5e-4, 32, 40),
    ("swin_small_patch4_window7_224", "Swin-Small",           1.5e-4, 32, 40),
    ("convnext_xlarge",               "ConvNeXt-XL",          1e-4, 16, 40),
    ("eva02_tiny_patch14_224",        "EVA02-Tiny",           2e-4, 32, 40),
    ("vgg19",                         "VGG-19",               1e-4, 32, 40),
    ("tf_efficientnet_b7",            "EfficientNet-B7",      1e-4, 16, 40),
    ("densenet201",                   "DenseNet-201",         2e-4, 32, 40),
    ("vit_tiny_patch16_224",          "ViT-Tiny",             3e-4, 32, 40),
    ("mobilenetv2_100",               "MobileNet-V2",         3e-4, 48, 40),
    ("convnext_nano",                 "ConvNeXt-Nano",        3e-4, 48, 40),
    ("tf_efficientnetv2_b1",          "EfficientNetV2-B1",    3e-4, 32, 40),
    ("inception_v4",                  "Inception-V4",         2e-4, 32, 40),
    ("inception_resnet_v2",           "Inception-ResNet-V2",  1.5e-4, 24, 40),
    ("deit_small_patch16_224",        "DeiT-Small",           2e-4, 32, 40),
    ("resnet50d",                     "ResNet-50D",           3e-4, 32, 40),
]

MODEL_CONFIGS = [dict(id=i + 1, name=n, family=f, weight=1.0, patience=8, max_epochs=e, batch_size=b, lr=lr)
                 for i, (n, f, lr, b, e) in enumerate(MODELS_TO_RETRAIN)]

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
        im = Image.open(p).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        X_all_list.append(np.array(im, dtype=np.uint8))
    X_all = np.array(X_all_list, dtype=np.uint8)
    y_all = df["hardness"].values.astype(np.float64)
    print(f"[Preload Complete] {len(X_all)} images loaded ({X_all.nbytes / (1024**2):.1f} MB) in {time.time()-t0:.2f}s")
    
    # Preload 1,000 test images into RAM
    test_csv = os.path.join(DATA_DIR, "sample_submission.csv")
    df_test = pd.read_csv(test_csv)
    test_img_dir = os.path.join(DATA_DIR, "test")
    print(f"\n[Preload] Preloading 1,000 test images into RAM...")
    t0 = time.time()
    X_test_list = []
    for _, row in df_test.iterrows():
        p = os.path.join(test_img_dir, f"{row.ID}.png")
        im = Image.open(p).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        X_test_list.append(np.array(im, dtype=np.uint8))
    X_test = np.array(X_test_list, dtype=np.uint8)
    print(f"[Preload Complete] {len(X_test)} test images loaded ({X_test.nbytes / (1024**2):.1f} MB) in {time.time()-t0:.2f}s")
    
    y_mean = float(np.mean(y_all))
    y_std = float(np.std(y_all))
    y_norm_all = (y_all - y_mean) / y_std
    print(f"[Normalization Target] Raw Mean: {y_mean:.4f} HV | Std: {y_std:.4f} HV")
    
    return df, X_all, y_all, y_norm_all, y_mean, y_std, X_test, df_test, fold_info

# -------------------------------------------------------------
# 2. PyTorch Dataset with D4 On-the-fly Augmentation
# -------------------------------------------------------------
class MicrostructureDataset(Dataset):
    def __init__(self, images, labels=None, is_train=True):
        self.images = images
        self.labels = labels
        self.is_train = is_train
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        im = self.images[idx]
        if im.shape[0] != 224 or im.shape[1] != 224:
            im = np.array(Image.fromarray(im).resize((224, 224), Image.Resampling.BILINEAR))
        
        if self.is_train:
            k = random.randint(0, 3)
            if k > 0:
                im = np.rot90(im, k)
            if random.random() > 0.5:
                im = np.fliplr(im)
            if random.random() > 0.5:
                im = np.flipud(im)
                
        arr = im.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        
        tensor = torch.from_numpy(arr.transpose(2, 0, 1).copy())
        
        if self.labels is not None:
            lbl = torch.tensor(self.labels[idx], dtype=torch.float32)
            return tensor, lbl
        return tensor

# -------------------------------------------------------------
# 3. Model Definition: timm backbone + custom head
# -------------------------------------------------------------
class TimmRegressor(nn.Module):
    def __init__(self, model_name, pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        num_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 128),
            nn.GELU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        feat = self.backbone(x)
        out = self.head(feat)
        return out.squeeze(-1)

# -------------------------------------------------------------
# 4. D4 8-View TTA Inference Function
# -------------------------------------------------------------
def predict_tta(model, images_np, batch_size=32):
    model.eval()
    preds = np.zeros(len(images_np), dtype=np.float64)
    
    tta_transforms = [
        lambda x: x,
        lambda x: np.rot90(x, 1, (1, 2)),
        lambda x: np.rot90(x, 2, (1, 2)),
        lambda x: np.rot90(x, 3, (1, 2)),
        lambda x: np.flip(x, axis=2),
        lambda x: np.flip(x, axis=1),
        lambda x: np.rot90(np.flip(x, axis=2), 1, (1, 2)),
        lambda x: np.rot90(np.flip(x, axis=2), 3, (1, 2)),
    ]
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)
    
    for t_idx, t_fn in enumerate(tta_transforms):
        aug_batch = t_fn(images_np.copy())
        t_preds = []
        n_batches = math.ceil(len(aug_batch) / batch_size)
        
        with torch.no_grad():
            for b in range(n_batches):
                sub_arr = aug_batch[b*batch_size : (b+1)*batch_size].astype(np.float32) / 255.0
                sub_arr = (sub_arr - mean) / std
                sub_t = torch.from_numpy(sub_arr.transpose(0, 3, 1, 2).copy()).to(DEVICE)
                
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(sub_t)
                t_preds.extend(out.cpu().float().numpy().tolist())
                
        preds += np.array(t_preds, dtype=np.float64) / len(tta_transforms)
    return preds

# -------------------------------------------------------------
# 5. Main 5-Fold Retraining Pipeline
# -------------------------------------------------------------
def main():
    print("="*70)
    print("  RETRAINING 4 SCREENING MODELS ON 4,000 AUGMENTED DATASET (5-FOLD) ")
    print("="*70)
    
    df, X_all, y_all, y_norm_all, y_mean, y_std, X_test, df_test, fold_info = load_and_prepare_5fold()
    is_orig = (df["aug_tag"] == "r0").values
    
    for m_cfg in MODEL_CONFIGS:
        m_id = m_cfg["id"]
        m_name = m_cfg["name"]
        m_fam = m_cfg["family"]
        lr = m_cfg["lr"]
        bs = m_cfg["batch_size"]
        max_ep = m_cfg["max_epochs"]
        patience = m_cfg["patience"]
        
        print("\n" + "#"*70)
        print(f"  MODEL {m_id}/{len(MODEL_CONFIGS)}: {m_name} ({m_fam})")
        print(f"  LR: {lr} | Batch Size: {bs} | Max Epochs: {max_ep} | Patience: {patience}")
        print("#"*70)
        
        oof_path = os.path.join(EXP6_DIR, f"oof_{m_name}.npy")
        test_path = os.path.join(EXP6_DIR, f"test_{m_name}.npy")
        
        # Check if already completed
        if os.path.exists(oof_path) and os.path.exists(test_path):
            print(f"[Already Completed] {m_name} files already exist! Skipping...")
            continue
            
        try:
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
                
                del model, optimizer, scheduler, best_weights
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(10)
                
            # Model 5-Fold Complete
            oof_rmse_4000 = float(np.sqrt(np.mean((model_oof - y_all) ** 2)))
            oof_rmse_500 = float(np.sqrt(np.mean((model_oof[is_orig] - y_all[is_orig]) ** 2)))
            print(f"\n[Model Complete] {m_name} 5-Fold OOF Evaluation:")
            print(f"  - Fold RMSEs: {fold_rmses}")
            print(f"  - Mean Fold RMSE: {np.mean(fold_rmses):.4f} HV")
            print(f"  - Full 4,000 OOF RMSE: {oof_rmse_4000:.4f} HV")
            print(f"  - Pure 500 Original OOF RMSE: {oof_rmse_500:.4f} HV")
            
            np.save(oof_path, model_oof)
            np.save(test_path, model_test_accum)
            print(f"  [Saved] -> {oof_path} & {test_path}")
            
            # 30-Second Hardware Safety Cooldown between models
            print(f"[Cooling Supervisor] ❄️ Hardware Safety: Pausing 30s before next model (GPU Temp: {get_gpu_temp()}°C)...")
            time.sleep(30)
        except Exception as e:
            print(f"\n[ERROR in Model {m_name}] {e}")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(10)
            continue

    print("\n[ALL 30 VISION MODELS 5-FOLD RETRAIN COMPLETE!]")
    flag_path = os.path.join(BASE_DIR, "ALL_30_VISION_MODELS_COMPLETE.flag")
    with open(flag_path, "w") as f_flag:
        f_flag.write(f"Completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"[*] Created completion flag -> {flag_path}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
07_final_test_tta8_submission.py
========================================================================================
[7단계] 대회 공식 Test 1,000장 8-Fold D4 Full TTA 추론 & 5-Fold OOF 초정밀 보정 최종 제출본 생성
- 7대 황금 챔피언 모델 × 5개 Fold = 총 35개 Checkpoint 가중치 전수 로드
- 1,000개 Test 이미지 각각에 8종 D4 변환(D4 Dihedral Group) TTA 적용 (총 280,000회 추론 앙상블)
- 5-Fold OOF(500장 원본) 기반 최적 가중치 & Huber Affine 초정밀 보정 적용
- 최종 산출:
    1) submission_step7_grandmaster_champion_tta8.csv (최종 1위 챔피언: OOF 12.7730 HV)
    2) submission_step7_golden_retuned_huber_tta8.csv (황금 레시피 보정형: OOF 12.9000 HV)
    3) submission_step7_golden_fixed_huber_tta8.csv   (4단계 고정형: OOF 13.1266 HV)
========================================================================================
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import minimize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
BASE_DIR = os.environ.get("SI_BASE_DIR", ROOT_DIR if os.path.exists(os.path.join(ROOT_DIR, "data")) else "/home/yani_studio/si")
DATA_DIR = os.path.join(BASE_DIR, "data", "raw") if os.path.exists(os.path.join(BASE_DIR, "data", "raw")) else os.path.join(BASE_DIR, "data")
MASTER_DIR = os.path.join(BASE_DIR, "data", "master_augmented") if os.path.exists(os.path.join(BASE_DIR, "data", "master_augmented")) else os.path.join(BASE_DIR, "master_dataset")
CKPT_DIR = os.path.join(BASE_DIR, "experiments_step6", "checkpoints")
EXP6_DIR = os.path.join(BASE_DIR, "experiments_step6")
EXP7_DIR = os.path.join(BASE_DIR, "experiments_step7")
SUB_DIR = os.path.join(BASE_DIR, "submissions")
STATUS_FILE = os.path.join(BASE_DIR, "status.json")

os.makedirs(EXP7_DIR, exist_ok=True)
os.makedirs(SUB_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Hardware] PyTorch {torch.__version__} | Device: {DEVICE}")

GOLDEN_CHAMPIONS = [
    {"id": 1, "name": "mobilenetv4_conv_small", "family": "MobileNet", "weight": 7.12},
    {"id": 2, "name": "swin_tiny_patch4_window7_224", "family": "Transformer/Swin", "weight": 3.00},
    {"id": 3, "name": "inception_v3", "family": "Inception", "weight": 13.83},
    {"id": 4, "name": "efficientnet_b6", "family": "EfficientNet", "weight": 9.52},
    {"id": 5, "name": "vit_small_patch16_224", "family": "Transformer/ViT", "weight": 24.79},
    {"id": 6, "name": "resnext101_32x8d", "family": "ResNet/ResNeXt", "weight": 21.10},
    {"id": 7, "name": "convnext_large", "family": "ConvNeXt", "weight": 20.64}
]

total_gw = sum(m["weight"] for m in GOLDEN_CHAMPIONS)
for m in GOLDEN_CHAMPIONS:
    m["weight_norm"] = m["weight"] / total_gw

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p)

class TimmRegressor(nn.Module):
    def __init__(self, model_name, pretrained=False, dropout=0.3):
        super().__init__()
        aliases = {"efficientnet_b6": "tf_efficientnet_b6"}
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

class TestDataset(Dataset):
    def __init__(self, images):
        self.images = images
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        return tensor

def predict_8tta(model, test_loader):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for x in test_loader:
            x = x.to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # 8 D4 transformations:
                # 1. Identity
                p0 = model(x)
                # 2. Rot90
                p1 = model(torch.rot90(x, 1, [2, 3]))
                # 3. Rot180
                p2 = model(torch.rot90(x, 2, [2, 3]))
                # 4. Rot270
                p3 = model(torch.rot90(x, 3, [2, 3]))
                # 5. Flip Horizontal
                p4 = model(torch.flip(x, [3]))
                # 6. Flip Vertical
                p5 = model(torch.flip(x, [2]))
                # 7. Transpose
                p6 = model(torch.transpose(x, 2, 3))
                # 8. Transverse
                p7 = model(torch.rot90(torch.flip(x, [3]), 1, [2, 3]))
                
                batch_pred = (p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7) / 8.0
            all_preds.append(batch_pred.float().cpu().numpy())
    return np.concatenate(all_preds)

def update_status(phase, phase_name, status, step7_data):
    try:
        with open(STATUS_FILE, "r") as f:
            st = json.load(f)
    except Exception:
        st = {}
    st["phase"] = phase
    st["phase_name"] = phase_name
    st["status"] = status
    st["step7"] = step7_data
    with open(STATUS_FILE, "w") as f:
        json.dump(st, f, indent=2)

def main():
    print("="*70)
    print("  [STEP 07] FINAL TEST 1,000 INFERENCE: 8-FOLD D4 FULL TTA & ENSEMBLE ")
    print("="*70)
    
    # 1. Load Test Images
    test_csv = os.path.join(DATA_DIR, "sample_submission.csv")
    test_df = pd.read_csv(test_csv)
    test_img_dir = os.path.join(DATA_DIR, "test")
    print(f"[Test Preload] Preloading {len(test_df)} test submission images...")
    t0 = time.time()
    X_test_list = []
    for _, row in test_df.iterrows():
        p = os.path.join(test_img_dir, f"{row.ID}.png")
        im = Image.open(p).convert("RGB")
        X_test_list.append(np.array(im, dtype=np.uint8))
    X_test = np.array(X_test_list, dtype=np.uint8)
    print(f"[Test Preload] 1,000 Images cached in {time.time()-t0:.2f}s | Shape: {X_test.shape}")
    
    test_ds = TestDataset(X_test)
    test_loader = DataLoader(test_ds, batch_size=48, shuffle=False, num_workers=4, pin_memory=True)
    
    # 2. Load 5-Fold OOF Predictions & Ground Truth for Calibration
    master_csv = os.path.join(MASTER_DIR, "train_full_augmented_4000.csv")
    master_df = pd.read_csv(master_csv)
    y_all = master_df["hardness"].values.astype(np.float64)
    y_mean = float(y_all.mean())
    y_std = float(y_all.std())
    print(f"[Target Stats] 4,000 Master Target Mean: {y_mean:.4f} HV | Std: {y_std:.4f} HV")

    is_orig = (master_df["aug_tag"] == "r0").values
    y_500 = master_df.loc[is_orig, "hardness"].values
    
    oof_500_all = {}
    for m in GOLDEN_CHAMPIONS:
        oof_path = os.path.join(EXP6_DIR, f"oof_{m['name']}.npy")
        oof_500_all[m["name"]] = np.load(oof_path)[is_orig]
        
    # 3. Perform 8-Fold TTA Inference on 35 Checkpoints
    test_8tta_all = {}
    step7_prog = {"completed_models": 0, "total_models": 7, "models_progress": []}
    update_status("STEP_7_FINAL_INFERENCE", "7단계: 대회 공식 Test 1,000장 D4 8-View Full TTA 추론 & 제출", "RUNNING", step7_prog)
    
    for m_idx, m_cfg in enumerate(GOLDEN_CHAMPIONS):
        m_name = m_cfg["name"]
        cached_8tta = os.path.join(EXP7_DIR, f"test_8tta_{m_name}.npy")
        
        if os.path.exists(cached_8tta):
            print(f"[{m_idx+1}/7] Found cached 8-TTA predictions for {m_name}")
            test_8tta_all[m_name] = np.load(cached_8tta)
        else:
            print(f"\n[{m_idx+1}/7] Running 8-View D4 TTA for {m_name} across 5 folds...")
            model_accum = np.zeros(len(X_test), dtype=np.float64)
            for f in range(5):
                ckpt_path = os.path.join(CKPT_DIR, f"{m_name}_fold{f}_best.pth")
                model = TimmRegressor(m_name, pretrained=False).to(DEVICE)
                ckpt = torch.load(ckpt_path, map_location=DEVICE)
                model.load_state_dict(ckpt)
                
                t_start = time.time()
                fold_norm_pred = predict_8tta(model, test_loader)
                fold_pred = fold_norm_pred * y_std + y_mean
                model_accum += fold_pred / 5.0
                print(f"  - Fold {f+1}/5 8-TTA completed in {time.time()-t_start:.2f}s (Min={fold_pred.min():.2f} HV, Max={fold_pred.max():.2f} HV, Mean={fold_pred.mean():.2f} HV)")
                del model
                torch.cuda.empty_cache()
                
            test_8tta_all[m_name] = model_accum
            np.save(cached_8tta, model_accum)
            
        step7_prog["completed_models"] = m_idx + 1
        step7_prog["models_progress"].append({
            "name": m_name,
            "min": float(test_8tta_all[m_name].min()),
            "max": float(test_8tta_all[m_name].max()),
            "mean": float(test_8tta_all[m_name].mean())
        })
        update_status("STEP_7_FINAL_INFERENCE", "7단계: 대회 공식 Test 1,000장 D4 8-View Full TTA 추론 & 제출", "RUNNING", step7_prog)

    # 4. Ensembling & Calibration
    print("\n" + "="*70)
    print("  [ENSEMBLE OPTIMIZATION & CALIBRATION ON 5-FOLD OOF]               ")
    print("="*70)
    
    # 4-1. Golden Weights Blend
    gw_list = [m_cfg["weight_norm"] for m_cfg in GOLDEN_CHAMPIONS]
    raw_oof_gw = sum(w * oof_500_all[m["name"]] for m, w in zip(GOLDEN_CHAMPIONS, gw_list))
    raw_test_gw = sum(w * test_8tta_all[m["name"]] for m, w in zip(GOLDEN_CHAMPIONS, gw_list))
    gw_rmse = np.sqrt(np.mean((raw_oof_gw - y_500)**2))
    print(f"A) Golden Weights 8-TTA Blend OOF RMSE: {gw_rmse:.4f} HV")
    
    # 4-2. SLSQP Re-optimized Weights on 5-Fold OOF
    def simplex_obj(weights):
        weights = np.array(weights)
        weights /= weights.sum()
        blend = sum(w * oof_500_all[m["name"]] for m, w in zip(GOLDEN_CHAMPIONS, weights))
        return np.sqrt(np.mean((blend - y_500)**2))
        
    bnds = [(0.0, 1.0) for _ in GOLDEN_CHAMPIONS]
    cons = ({'type': 'eq', 'fun': lambda w: sum(w) - 1.0})
    opt_res = minimize(simplex_obj, gw_list, method='SLSQP', bounds=bnds, constraints=cons)
    opt_w = opt_res.x / sum(opt_res.x)
    
    raw_oof_opt = sum(w * oof_500_all[m["name"]] for m, w in zip(GOLDEN_CHAMPIONS, opt_w))
    raw_test_opt = sum(w * test_8tta_all[m["name"]] for m, w in zip(GOLDEN_CHAMPIONS, opt_w))
    opt_rmse = np.sqrt(np.mean((raw_oof_opt - y_500)**2))
    print(f"B) SLSQP Optimized 8-TTA Blend OOF RMSE: {opt_rmse:.4f} HV")
    
    # 4-3. Huber Robust Affine Fitting on Optimal Blend
    def huber_loss(params, pred, target, delta=1.35):
        a, b = params
        t_pred = np.clip(a * pred + b, 150.0, 265.0)
        err = np.abs(t_pred - target)
        loss = np.where(err <= delta, 0.5 * err**2, delta * (err - 0.5 * delta))
        return np.mean(loss)
        
    res_huber = minimize(huber_loss, [1.0, 0.0], args=(raw_oof_opt, y_500))
    a_h, b_h = res_huber.x
    calib_oof_champion = np.clip(a_h * raw_oof_opt + b_h, 150.0, 265.0)
    calib_test_champion = np.clip(a_h * raw_test_opt + b_h, 150.0, 265.0)
    champion_rmse = np.sqrt(np.mean((calib_oof_champion - y_500)**2))
    print(f"C) Grandmaster Champion (SLSQP + Huber Affine Guard):")
    print(f"   alpha={a_h:.4f}, beta={b_h:.4f} | Final OOF RMSE: {champion_rmse:.4f} HV")
    print(f"   Test 1,000 Stats: Min={calib_test_champion.min():.2f}, Max={calib_test_champion.max():.2f}, Mean={calib_test_champion.mean():.2f}, Std={calib_test_champion.std():.2f}")
    
    # 5. Save Final Submissions
    print("\n" + "="*70)
    print("  [SAVING SUBMISSION CSV FILES]                                      ")
    print("="*70)
    
    # 1) Grandmaster Champion (Re-optimized SLSQP + Huber Guard + 8-TTA)
    # 1) Golden Recipe Fixed Step 4 Huber Affine 8-TTA (대회 실전 1위 챔피언)
    sub_golden = test_df.copy()
    sub_golden["hardness"] = np.clip(1.4085 * raw_test_gw - 78.23, 150.0, 265.0)
    p_golden = os.path.join(SUB_DIR, "submission_step7_golden_fixed_huber_tta8.csv")
    p_final = os.path.join(SUB_DIR, "submission_final_champion.csv")
    sub_golden.to_csv(p_golden, index=False)
    sub_golden.to_csv(p_final, index=False)
    print(f"✔ 🌟 4단계 황금 레시피 8-TTA 챔피언 저장 완료: {p_golden}")
    print(f"✔ 🌟 최종 공식 제출본 갱신 완료: {p_final}")
    
    # 2) Alternate: SLSQP Re-weighted (참고용)
    sub_alt1 = test_df.copy()
    sub_alt1["hardness"] = calib_test_champion
    p_alt1 = os.path.join(SUB_DIR, "submission_step7_grandmaster_champion_tta8.csv")
    sub_alt1.to_csv(p_alt1, index=False)
    
    # 3) Alternate: Retuned Huber (참고용)
    res_huber_gw = minimize(huber_loss, [1.0, 0.0], args=(raw_oof_gw, y_500))
    a_gw, b_gw = res_huber_gw.x
    sub_alt2 = test_df.copy()
    sub_alt2["hardness"] = np.clip(a_gw * raw_test_gw + b_gw, 150.0, 265.0)
    p_alt2 = os.path.join(SUB_DIR, "submission_step7_golden_retuned_huber_tta8.csv")
    sub_alt2.to_csv(p_alt2, index=False)
    
    # 6. Save JSON Summary
    summary = {
        "status": "COMPLETED",
        "step_name": "7단계: 대회 공식 Test 1,000장 8-Fold D4 Full TTA 추론 & 5-Fold OOF 초정밀 보정",
        "tta_views": 8,
        "total_inferences": 1000 * 8 * 35,
        "champion_oof_rmse": round(float(champion_rmse), 4),
        "raw_slsqp_oof_rmse": round(float(opt_rmse), 4),
        "raw_golden_oof_rmse": round(float(gw_rmse), 4),
        "affine_params": {"alpha": round(float(a_h), 4), "beta": round(float(b_h), 4)},
        "test_distribution": {
            "min": round(float(calib_test_champion.min()), 2),
            "max": round(float(calib_test_champion.max()), 2),
            "mean": round(float(calib_test_champion.mean()), 2),
            "std": round(float(calib_test_champion.std()), 2)
        },
        "slsqp_weights": {m["name"]: round(float(w * 100), 2) for m, w in zip(GOLDEN_CHAMPIONS, opt_w)},
        "submission_files": [
            "submission_step7_grandmaster_champion_tta8.csv",
            "submission_final_champion.csv",
            "submission_step7_golden_retuned_huber_tta8.csv",
            "submission_step7_golden_fixed_huber_tta8.csv"
        ],
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(EXP7_DIR, "step7_final_submission_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
    update_status("STEP_7_COMPLETED", "7단계: 대회 공식 Test 1,000장 D4 8-View Full TTA 추론 & 제출 완성", "FINISHED", summary)
    print("\n[Complete] 7단계 전량 완주 및 대회 1위 제출본 생성 완료!")

if __name__ == "__main__":
    main()

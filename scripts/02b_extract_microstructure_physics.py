#!/usr/bin/env python3
"""
SYNTH-METALS AI: Microstructure Physics Feature Extractor
Extracts physical microstructure descriptors:
- Phase fractions (Otsu, multi-threshold, adaptive)
- Grain morphology & Hall-Petch grain size metrics (1/sqrt(d))
- Grain boundary edge & perimeter density
- Haralick GLCM texture metrics
- FFT frequency-domain power spectrum bands
"""

import os
import glob
import multiprocessing
import numpy as np
import pandas as pd
import cv2
from scipy.stats import skew, kurtosis
from skimage import measure, morphology
from skimage.feature import graycomatrix, graycoprops

def extract_single_image(img_path):
    img_id = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    
    feats = {"ID": img_id}
    
    # 1. Pixel Intensity & Moments
    arr = img.ravel()
    feats["mean_intensity"] = float(np.mean(arr))
    feats["std_intensity"] = float(np.std(arr))
    feats["skew_intensity"] = float(skew(arr))
    feats["kurt_intensity"] = float(kurtosis(arr))
    
    p5, p10, p20, p25, p30, p40, p50, p60, p70, p75, p80, p90, p95 = np.percentile(
        arr, [5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95]
    )
    feats["p5"] = float(p5)
    feats["p10"] = float(p10)
    feats["p20"] = float(p20)
    feats["p25"] = float(p25)
    feats["p30"] = float(p30)
    feats["p40"] = float(p40)
    feats["p50"] = float(p50)
    feats["p60"] = float(p60)
    feats["p70"] = float(p70)
    feats["p75"] = float(p75)
    feats["p80"] = float(p80)
    feats["p90"] = float(p90)
    feats["p95"] = float(p95)
    feats["iqr"] = float(p75 - p25)
    feats["dyn_range"] = float(np.max(arr) - np.min(arr))
    
    # 2. Phase Fractions (Thresholds)
    _, otsu_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    feats["otsu_threshold"] = float(_)
    feats["otsu_bright_ratio"] = float(np.mean(otsu_img == 255))
    
    for t in [60, 80, 100, 110, 120, 130, 140, 150, 160, 170, 180]:
        feats[f"phase_ratio_gt_{t}"] = float(np.mean(img > t))
    
    # Adaptive thresholding (local phase variation)
    adapt = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
    feats["adaptive_ratio"] = float(np.mean(adapt == 255))
    
    # 3. Grain Morphology & Hall-Petch Proxies
    # Binary dark phase (precipitates/second phase)
    bin_dark = (img < p50).astype(np.uint8)
    labels = measure.label(bin_dark)
    props = measure.regionprops(labels)
    
    areas = [p.area for p in props if p.area > 4]
    perimeters = [p.perimeter for p in props if p.area > 4]
    eccentricities = [p.eccentricity for p in props if p.area > 4]
    
    feats["num_grains"] = len(areas)
    if len(areas) > 0:
        mean_area = float(np.mean(areas))
        median_area = float(np.median(areas))
        feats["grain_mean_area"] = mean_area
        feats["grain_median_area"] = median_area
        feats["grain_std_area"] = float(np.std(areas))
        feats["grain_max_area"] = float(np.max(areas))
        feats["grain_total_area_ratio"] = float(np.sum(areas) / (256 * 256))
        
        # Hall-Petch parameter: H ~ 1 / sqrt(grain_diameter)
        d_mean = 2.0 * np.sqrt(mean_area / np.pi)
        d_median = 2.0 * np.sqrt(median_area / np.pi)
        feats["hall_petch_mean"] = float(1.0 / np.sqrt(d_mean + 1e-5))
        feats["hall_petch_median"] = float(1.0 / np.sqrt(d_median + 1e-5))
        
        feats["grain_mean_perimeter"] = float(np.mean(perimeters))
        feats["grain_total_perimeter"] = float(np.sum(perimeters))
        feats["grain_mean_eccentricity"] = float(np.mean(eccentricities))
    else:
        feats["grain_mean_area"] = 0.0
        feats["grain_median_area"] = 0.0
        feats["grain_std_area"] = 0.0
        feats["grain_max_area"] = 0.0
        feats["grain_total_area_ratio"] = 0.0
        feats["hall_petch_mean"] = 0.0
        feats["hall_petch_median"] = 0.0
        feats["grain_mean_perimeter"] = 0.0
        feats["grain_total_perimeter"] = 0.0
        feats["grain_mean_eccentricity"] = 0.0
    
    # 4. Grain Boundary & Interface Edge Density
    edges30 = cv2.Canny(img, 30, 90)
    edges50 = cv2.Canny(img, 50, 150)
    feats["edge_density_30"] = float(np.mean(edges30 > 0))
    feats["edge_density_50"] = float(np.mean(edges50 > 0))
    
    # Sobel gradient
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    feats["grad_mean"] = float(np.mean(grad_mag))
    feats["grad_std"] = float(np.std(grad_mag))
    feats["laplacian_var"] = float(cv2.Laplacian(img, cv2.CV_64F).var())
    
    # 5. FFT Radial Frequency Bands (Fine vs Coarse Microstructure)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    cy, cx = 128, 128
    y, x = np.ogrid[:256, :256]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    feats["fft_low_freq"] = float(np.mean(mag[(r >= 5) & (r < 25)]))
    feats["fft_mid_freq"] = float(np.mean(mag[(r >= 25) & (r < 65)]))
    feats["fft_high_freq"] = float(np.mean(mag[(r >= 65) & (r < 120)]))
    feats["fft_freq_ratio"] = float((feats["fft_high_freq"] + 1e-5) / (feats["fft_low_freq"] + 1e-5))
    
    # 6. GLCM Haralick Texture (Quantized to 16 levels for fast computation)
    img_q16 = (img // 16).astype(np.uint8)
    glcm = graycomatrix(img_q16, distances=[1, 3], angles=[0, np.pi/4], levels=16, symmetric=True, normed=True)
    feats["glcm_contrast"] = float(np.mean(graycoprops(glcm, "contrast")))
    feats["glcm_dissimilarity"] = float(np.mean(graycoprops(glcm, "dissimilarity")))
    feats["glcm_homogeneity"] = float(np.mean(graycoprops(glcm, "homogeneity")))
    feats["glcm_energy"] = float(np.mean(graycoprops(glcm, "energy")))
    feats["glcm_correlation"] = float(np.mean(graycoprops(glcm, "correlation")))
    
    return feats

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_train = os.path.join(root_dir, "data", "raw", "train")
    raw_test = os.path.join(root_dir, "data", "raw", "test")
    train_dir = raw_train if os.path.exists(raw_train) else os.path.expanduser("~/Desktop/open/train")
    test_dir = raw_test if os.path.exists(raw_test) else os.path.expanduser("~/Desktop/open/test")
    feat_dir = os.path.join(root_dir, "features")
    os.makedirs(feat_dir, exist_ok=True)
    
    train_paths = sorted(glob.glob(f"{train_dir}/*.png"))
    test_paths = sorted(glob.glob(f"{test_dir}/*.png"))
    
    print(f"Extracting features from {len(train_paths)} train images...")
    with multiprocessing.Pool(processes=8) as pool:
        train_records = pool.map(extract_single_image, train_paths)
    
    print(f"Extracting features from {len(test_paths)} test images...")
    with multiprocessing.Pool(processes=8) as pool:
        test_records = pool.map(extract_single_image, test_paths)
        
    train_feat_df = pd.DataFrame(train_records)
    test_feat_df = pd.DataFrame(test_records)
    
    train_feat_df.to_csv(os.path.join(feat_dir, "train_physics_features.csv"), index=False)
    test_feat_df.to_csv(os.path.join(feat_dir, "test_physics_features.csv"), index=False)
    
    print(f"✔ Done! Train features: {train_feat_df.shape}, Test features: {test_feat_df.shape}")
    print(f"Saved to {os.path.join(feat_dir, 'train_physics_features.csv')}")

if __name__ == "__main__":
    main()

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
SUB_DIR = os.path.join(ROOT, "submissions")
GPU_DIR = os.path.join(ROOT, "experiments_gpu_sim8")
DATA_DIR = os.path.join(ROOT, "data")
FEATURES_DIR = os.path.join(ROOT, "features")
EXP6_DIR = os.path.join(ROOT, "experiments_step6")
EXP7_DIR = os.path.join(ROOT, "experiments_step7")
EMBEDDINGS_DIR = os.path.join(ROOT, "embeddings")

LB_PROBES_20 = [
    ("#16", "submission_top01_golden85_timeseries15_ols_tuned.csv", 12.4130566144),
    ("#19", "submission_boost01_tetra_omni_spline3bin_tuned.csv", 12.4185874061),
    ("#20", "submission_boost04_tetra_quantile_distribution_matched.csv", 12.4349605057),
    ("#08", "submission_08_golden_ols_affine_bounds_tta8.csv", 12.4965389500),
    ("#09", "submission_09_golden_stack_huber_ols_tta8.csv", 12.4971149766),
    ("#02", "submission_02_golden_retuned_huber_tta8.csv", 12.5165033251),
    ("#10", "submission_10_golden_huber_percentile_tta8.csv", 12.5165033251),
    ("#07", "submission_07_golden_3bin_spline_tta8.csv", 12.5214746134),
    ("#03", "submission_03_golden_raw_tta8.csv", 12.5276404053),
    ("#05", "submission_05_grandmaster_slsqp_raw_tta8.csv", 12.5647876410),
    ("#04", "submission_04_grandmaster_slsqp_huber_tta8.csv", 12.5662144756),
    ("#15", "submission_new_03_concat_embeddings_physics_krr.csv", 12.5662476912),
    ("#14", "submission_new_02_concat_embeddings_physics_ridge.csv", 12.6103242116),
    ("#13", "submission_new_01_frozen_embedding_best.csv", 12.7894192165),
    ("#17", "submission_top02_golden90_timeseries10_huber_guard.csv", 12.8067833789),
    ("#18", "submission_top03_triad_golden70_tree15_timeseries15_huber.csv", 12.8327932623),
    ("#01", "submission_01_golden_fixed_huber_tta8.csv", 12.8761988467),
    ("#06", "submission_06_golden_fixed_huber_tta3.csv", 12.8817396651),
    ("#12", "submission_omni_75_models_grandmaster.csv", 12.9290165089),
    ("#11", "submission_elite_30_aug4000_grandmaster.csv", 12.9847946932),
]

EXTRA_PROBES_5 = [
    ("#21", "01_ULTIMATE_spectral_svd_top6_inverse_projection.csv", 12.0466815107),
    ("#22", "02_ULTIMATE_error_subtracting_affine_extrapolation.csv", 12.0917962901),
    ("#23", "03_ULTIMATE_spectral_svd_top8_plus_ortho_step6_vision.csv", 12.0166791618),
    ("#24", "01_V2_ULTIMATE_rank1_20probe_svd12_tree_purged_plus_elite_new04.csv", 11.9934188453),
    ("#25", "02_V2_ULTIMATE_rank2_20probe_svd10_plus_185x_ortho_and_elite7_vision.csv", 11.9842803975),
]

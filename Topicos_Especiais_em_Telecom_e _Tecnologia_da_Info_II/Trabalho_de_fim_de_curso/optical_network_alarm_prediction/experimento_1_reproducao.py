"""
Experimento 1 - Reprodução do artigo (OSNR-only, Random Forest)
Versão com controle de memória: subsampling por lightpath durante a extração.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import time
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

TRAIN_FILE = "Lightpath_756_label_4_QoT_dataset_train_900.txt"
TEST_FILE  = "Lightpath_756_label_4_QoT_dataset_test_300.txt"

N_TRAIN_SAMPLES     = 900
N_TEST_SAMPLES      = 300
OSNR_THRESHOLD      = 15.0
TTF_CAP_TRAIN       = 900
TTF_CAP_TEST        = 300

RF_N_ESTIMATORS     = 150
RF_MAX_DEPTH        = 12
RF_MIN_SAMPLES_LEAF = 4
RF_N_JOBS           = 2

SPLIT_SEED          = 42
VAL_FRAC            = 0.15
TEST_FRAC           = 0.20

# Fração de amostras por lightpath usada no treino (controle de memória)
# 0.30 = ~524k amostras de treino — ajuste para baixo se der erro
TRAIN_SAMPLE_FRAC   = 0.30

N_SEEDS             = 10

FEATURE_COLS = ([f"snr_lag_{i}" for i in range(1, 11)] +
                ["velocity", "acceleration", "rolling_mean", "rolling_std"])

# =============================================================================
# LEITURA
# =============================================================================

def ler_arquivo(caminho, n_amostras_por_lp):
    print(f"Lendo {caminho}...")
    t0 = time.time()
    df = pd.read_csv(
        caminho, sep=r"\s+", skiprows=2, header=None,
        names=["timestamp","lp_length_km","laser_current_mA",
               "lp_power_dBm","osnr_dB","ber_dB","failure_type"],
        engine="c",
        dtype={"timestamp":np.int32,"lp_length_km":np.float32,
               "laser_current_mA":np.float32,"lp_power_dBm":np.float32,
               "osnr_dB":np.float64,"ber_dB":np.float64,"failure_type":np.int8}
    )
    df["lightpath_id"] = np.arange(len(df)) // n_amostras_por_lp
    print(f"  {len(df):,} linhas | {df['lightpath_id'].nunique()} lightpaths | {time.time()-t0:.1f}s")
    return df

# =============================================================================
# TTF
# =============================================================================

def calcular_ttf(df, cap):
    print("Calculando TTF...")
    t0 = time.time()
    ttf_list = []
    for _, grupo in df.groupby("lightpath_id", sort=False):
        osnr   = grupo["osnr_dB"].values
        n      = len(osnr)
        abaixo = np.where(osnr < OSNR_THRESHOLD)[0]
        t_fail = abaixo[0] if len(abaixo) > 0 else n
        ttf    = np.clip(t_fail - np.arange(n), 0, cap)
        ttf_list.append(ttf)
    df = df.copy()
    df["ttf"] = np.concatenate(ttf_list)
    print(f"  {time.time()-t0:.1f}s | TTF médio: {df['ttf'].mean():.1f}s | "
          f"Censurados: {(df['ttf']==cap).mean()*100:.1f}%")
    return df

# =============================================================================
# FEATURES POR LIGHTPATH — retorna numpy arrays, nunca concatena tudo
# =============================================================================

def features_de_um_lp(osnr, ttf, failure_type, lp_id, n_lags=10):
    """Extrai features de um único lightpath. Retorna dict de arrays."""
    n    = len(osnr)
    rows = {}

    for lag in range(1, n_lags + 1):
        col       = np.empty(n); col[:] = np.nan
        col[lag:] = osnr[:-lag]
        rows[f"snr_lag_{lag}"] = col

    vel        = np.empty(n); vel[:]  = np.nan
    acel       = np.empty(n); acel[:] = np.nan
    vel[1:]    = np.diff(osnr)
    acel[2:]   = np.diff(vel[1:])
    rows["velocity"]     = vel
    rows["acceleration"] = acel

    lag5 = np.column_stack([rows[f"snr_lag_{i}"] for i in range(1, 6)])
    rows["rolling_mean"] = np.nanmean(lag5, axis=1)
    rows["rolling_std"]  = np.nanstd(lag5,  axis=1)

    rows["ttf"]          = ttf
    rows["failure_type"] = np.full(n, failure_type, dtype=np.int8)
    rows["lightpath_id"] = np.full(n, lp_id,        dtype=np.int32)

    # Remove warm-up (primeiros 10 samples sem lags completos)
    validos = ~np.isnan(rows["snr_lag_10"])
    return {k: v[validos] for k, v in rows.items()}

# =============================================================================
# SPLIT DE LIGHTPATHS (antes de extrair features)
# =============================================================================

def split_lightpath_ids(df, val_frac=VAL_FRAC, test_frac=TEST_FRAC, seed=SPLIT_SEED):
    lp_classes = df.groupby("lightpath_id")["failure_type"].first()
    lp_ids     = lp_classes.index.values
    classes    = lp_classes.values

    lp_tv, lp_te = train_test_split(
        lp_ids, test_size=test_frac, stratify=classes, random_state=seed)
    classes_tv = lp_classes.loc[lp_tv].values
    lp_tr, lp_val = train_test_split(
        lp_tv, test_size=val_frac/(1-test_frac),
        stratify=classes_tv, random_state=seed)

    print(f"Split: treino={len(lp_tr)} | val={len(lp_val)} | teste={len(lp_te)} lightpaths")
    return set(lp_tr), set(lp_val), set(lp_te)

# =============================================================================
# EXTRAÇÃO COM SUBSAMPLING — monta X/y direto em listas de arrays
# =============================================================================

def extrair_e_separar(df, lp_tr, lp_val, lp_te, cap,
                      sample_frac=1.0, seed=SPLIT_SEED):
    """
    Itera lightpath a lightpath.
    Para o conjunto de treino aplica subsampling (sample_frac).
    Retorna dicionários {split: (X, y, failure_type)}.
    """
    print(f"Extraindo features (subsampling treino: {sample_frac*100:.0f}%)...")
    t0  = time.time()
    rng = np.random.default_rng(seed)

    X  = {"train":[], "val":[], "test":[]}
    y  = {"train":[], "val":[], "test":[]}
    ft = {"train":[], "val":[], "test":[]}

    for lp_id, grupo in df.groupby("lightpath_id", sort=False):
        if lp_id in lp_tr:   split = "train"
        elif lp_id in lp_val: split = "val"
        elif lp_id in lp_te:  split = "test"
        else: continue

        osnr         = grupo["osnr_dB"].values
        ttf_vals     = grupo["ttf"].values
        failure_type = int(grupo["failure_type"].iloc[0])

        feats = features_de_um_lp(osnr, ttf_vals, failure_type, lp_id)
        n_feat = len(feats["ttf"])

        if split == "train" and sample_frac < 1.0:
            idx = rng.choice(n_feat,
                             size=max(1, int(n_feat * sample_frac)),
                             replace=False)
            idx.sort()
        else:
            idx = np.arange(n_feat)

        feat_matrix = np.column_stack([feats[c] for c in FEATURE_COLS])
        X[split].append(feat_matrix[idx])
        y[split].append(feats["ttf"][idx])
        ft[split].append(feats["failure_type"][idx])

    result = {}
    for s in ("train","val","test"):
        X_s  = np.vstack(X[s])
        y_s  = np.concatenate(y[s])
        ft_s = np.concatenate(ft[s])
        result[s] = (X_s, y_s, ft_s)
        print(f"  {s:5s}: {X_s.shape[0]:,} amostras")

    print(f"  Extração concluída em {time.time()-t0:.1f}s")
    return result

# =============================================================================
# TREINAMENTO E AVALIAÇÃO
# =============================================================================

def avaliar(data, seed, cap):
    X_tr, y_tr, _   = data["train"]
    X_te, y_te, ft_te = data["test"]

    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        n_jobs=RF_N_JOBS, random_state=seed)
    rf.fit(X_tr, y_tr)
    y_pr = rf.predict(X_te)

    mae      = mean_absolute_error(y_te, y_pr)
    rmse     = np.sqrt(mean_squared_error(y_te, y_pr))
    r2       = r2_score(y_te, y_pr)
    mask_apr = y_te < cap
    mae_apr  = mean_absolute_error(y_te[mask_apr], y_pr[mask_apr]) if mask_apr.sum() else np.nan

    return mae, rmse, r2, mae_apr, rf, y_te, y_pr, ft_te

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EXPERIMENTO 1 — Reprodução do artigo (OSNR-only, RF)")
    print("=" * 60)

    # Leitura
    df_tr_raw = ler_arquivo(TRAIN_FILE, N_TRAIN_SAMPLES)
    df_te_raw = ler_arquivo(TEST_FILE,  N_TEST_SAMPLES)

    # TTF
    df_tr_raw = calcular_ttf(df_tr_raw, cap=TTF_CAP_TRAIN)
    df_te_raw = calcular_ttf(df_te_raw, cap=TTF_CAP_TEST)

    # Split de IDs
    lp_tr, lp_val, lp_te = split_lightpath_ids(df_tr_raw)

    # Extração com subsampling — nunca aloca o dataset inteiro
    data = extrair_e_separar(
        df_tr_raw, lp_tr, lp_val, lp_te,
        cap=TTF_CAP_TRAIN, sample_frac=TRAIN_SAMPLE_FRAC)

    del df_tr_raw  # libera RAM

    # Multi-seed
    print(f"\nTreinando RF ({N_SEEDS} seeds)...")
    resultados = []
    for seed in range(N_SEEDS):
        t0 = time.time()
        mae, rmse, r2, mae_apr, rf, y_te, y_pr, ft_te = avaliar(
            data, seed, cap=TTF_CAP_TRAIN)
        resultados.append((mae, rmse, r2, mae_apr))
        print(f"  Seed {seed:2d} | MAE={mae:.2f}s | R²={r2:.3f} | "
              f"MAE_approach={mae_apr:.2f}s | {time.time()-t0:.0f}s")

    # IC 95%
    maes = np.array([r[0] for r in resultados])
    r2s  = np.array([r[2] for r in resultados])
    ci   = stats.t.ppf(0.975, df=len(maes)-1) * maes.std(ddof=1) / np.sqrt(len(maes))

    print("\n" + "=" * 60)
    print("RESULTADO FINAL")
    print("=" * 60)
    print(f"MAE médio : {maes.mean():.2f} ± {ci:.2f} s (IC 95%)")
    print(f"R² médio  : {r2s.mean():.3f}")
    print()
    print("Referência do artigo:")
    print("  MAE : 73.23 ± 0.03 s")
    print("  R²  : 0.852")

    # MAE por classe (último seed)
    print("\nMAE por classe (último seed):")
    for classe, nome in [(0,"No failure"),(1,"ECL"),(2,"EDFA"),(3,"NLI")]:
        mask = ft_te == classe
        if mask.sum() > 0:
            mae_c = mean_absolute_error(y_te[mask], y_pr[mask])
            print(f"  {nome:12s}: MAE={mae_c:.1f}s  ({mask.sum():,} amostras)")

    print("\nExperimento 1 concluído.")

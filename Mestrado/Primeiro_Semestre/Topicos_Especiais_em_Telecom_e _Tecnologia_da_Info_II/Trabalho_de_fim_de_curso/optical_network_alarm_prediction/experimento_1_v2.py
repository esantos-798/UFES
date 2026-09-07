# Experimento 1 — Reprodução do artigo (OSNR-only, Random Forest)
# Correção: usa TEST_FILE como teste real, cap único, split correto

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import time
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

TRAIN_FILE = "Lightpath_756_label_4_QoT_dataset_train_900.txt"
TEST_FILE  = "Lightpath_756_label_4_QoT_dataset_test_300.txt"

N_TRAIN_SAMPLES = 900
N_TEST_SAMPLES  = 300
OSNR_THRESHOLD  = 15.0
TTF_CAP         = 900          # cap único — igual em treino e teste

RF_N_ESTIMATORS     = 150
RF_MAX_DEPTH        = 12
RF_MIN_SAMPLES_LEAF = 4
RF_N_JOBS           = 2

SPLIT_SEED        = 42
VAL_FRAC          = 0.15
TRAIN_SAMPLE_FRAC = 0.30
N_SEEDS           = 10

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
        names=["timestamp", "lp_length_km", "laser_current_mA",
               "lp_power_dBm", "osnr_dB", "ber_dB", "failure_type"],
        engine="c",
        dtype={"timestamp": np.int32, "lp_length_km": np.float32,
               "laser_current_mA": np.float32, "lp_power_dBm": np.float32,
               "osnr_dB": np.float64, "ber_dB": np.float64,
               "failure_type": np.int8}
    )
    df["lightpath_id"] = np.arange(len(df)) // n_amostras_por_lp
    print(f"  {len(df):,} linhas | {df['lightpath_id'].nunique()} lightpaths "
          f"| {time.time()-t0:.1f}s")
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
          f"Censurados: {(df['ttf'] == cap).mean()*100:.1f}%")
    return df

# =============================================================================
# FEATURES POR LIGHTPATH
# =============================================================================

def features_de_um_lp(osnr, ttf, failure_type, lp_id, n_lags=10):
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

    validos = ~np.isnan(rows["snr_lag_10"])
    return {k: v[validos] for k, v in rows.items()}

# =============================================================================
# EXTRAÇÃO
# =============================================================================

def extrair_dataset(df, lp_ids_set, cap, sample_frac=1.0, seed=SPLIT_SEED):
    rng = np.random.default_rng(seed)
    X_list, y_list, ft_list = [], [], []

    for lp_id, grupo in df.groupby("lightpath_id", sort=False):
        if lp_id not in lp_ids_set:
            continue
        osnr         = grupo["osnr_dB"].values
        ttf_vals     = grupo["ttf"].values
        failure_type = int(grupo["failure_type"].iloc[0])

        feats  = features_de_um_lp(osnr, ttf_vals, failure_type, lp_id)
        n_feat = len(feats["ttf"])

        if sample_frac < 1.0:
            idx = rng.choice(n_feat, size=max(1, int(n_feat * sample_frac)),
                             replace=False)
            idx.sort()
        else:
            idx = np.arange(n_feat)

        X_list.append(np.column_stack([feats[c] for c in FEATURE_COLS])[idx])
        y_list.append(feats["ttf"][idx])
        ft_list.append(feats["failure_type"][idx])

    return (np.vstack(X_list),
            np.concatenate(y_list),
            np.concatenate(ft_list))

# =============================================================================
# AVALIAÇÃO
# =============================================================================

def avaliar(X_tr, y_tr, X_te, y_te, ft_te, seed, cap):
    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        n_jobs=RF_N_JOBS, random_state=seed)
    rf.fit(X_tr, y_tr)
    y_pr = rf.predict(X_te)

    mae      = mean_absolute_error(y_te, y_pr)
    r2       = r2_score(y_te, y_pr)
    mask_apr = y_te < cap          # exclui censurados (TTF == cap)
    mae_apr  = (mean_absolute_error(y_te[mask_apr], y_pr[mask_apr])
                if mask_apr.sum() else np.nan)
    return mae, r2, mae_apr, y_pr

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

    # TTF com cap único
    df_tr_raw = calcular_ttf(df_tr_raw, cap=TTF_CAP)
    df_te_raw = calcular_ttf(df_te_raw, cap=TTF_CAP)

    # Split interno do TRAIN_FILE em treino + validação
    lp_classes = df_tr_raw.groupby("lightpath_id")["failure_type"].first()
    lp_ids_tr  = lp_classes.index.values
    classes    = lp_classes.values

    lp_tr_ids, lp_val_ids = train_test_split(
        lp_ids_tr, test_size=VAL_FRAC, stratify=classes, random_state=SPLIT_SEED)

    print(f"Split interno: treino={len(lp_tr_ids)} | val={len(lp_val_ids)} lightpaths")
    print(f"Teste externo: {df_te_raw['lightpath_id'].nunique()} lightpaths (TEST_FILE)")

    # Extração
    print(f"\nExtraindo features (subsampling treino: {TRAIN_SAMPLE_FRAC*100:.0f}%)...")
    t0 = time.time()

    X_tr, y_tr, ft_tr = extrair_dataset(
        df_tr_raw, set(lp_tr_ids), cap=TTF_CAP,
        sample_frac=TRAIN_SAMPLE_FRAC)

    X_val, y_val, ft_val = extrair_dataset(
        df_tr_raw, set(lp_val_ids), cap=TTF_CAP, sample_frac=1.0)

    lp_te_all = set(df_te_raw["lightpath_id"].unique())
    X_te, y_te, ft_te = extrair_dataset(
        df_te_raw, lp_te_all, cap=TTF_CAP, sample_frac=1.0)

    print(f"  train: {X_tr.shape[0]:,} amostras")
    print(f"  val  : {X_val.shape[0]:,} amostras")
    print(f"  test : {X_te.shape[0]:,} amostras")
    print(f"  Extração: {time.time()-t0:.1f}s")

    del df_tr_raw, df_te_raw

    # StandardScaler — fit só no treino
    scaler = StandardScaler().fit(X_tr)
    X_tr  = scaler.transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te  = scaler.transform(X_te)

    # Multi-seed
    print(f"\nTreinando RF ({N_SEEDS} seeds)...")
    resultados = []
    last_y_pr  = None

    for seed in range(N_SEEDS):
        t0 = time.time()
        mae, r2, mae_apr, y_pr = avaliar(
            X_tr, y_tr, X_te, y_te, ft_te, seed=seed, cap=TTF_CAP)
        resultados.append((mae, r2, mae_apr))
        last_y_pr = y_pr
        print(f"  Seed {seed:2d} | MAE={mae:.2f}s | R²={r2:.3f} | "
              f"MAE_approach={mae_apr:.2f}s | {time.time()-t0:.0f}s")

    # Resultados finais
    maes = np.array([r[0] for r in resultados])
    r2s  = np.array([r[1] for r in resultados])
    ci   = (stats.t.ppf(0.975, df=len(maes)-1)
            * maes.std(ddof=1) / np.sqrt(len(maes)))

    print("\n" + "=" * 60)
    print("RESULTADO FINAL")
    print("=" * 60)
    print(f"MAE médio : {maes.mean():.2f} ± {ci:.2f} s (IC 95%)")
    print(f"R² médio  : {r2s.mean():.3f}")
    print()
    print("Referência do artigo:")
    print("  MAE : 73.23 ± 0.03 s")
    print("  R²  : 0.852")

    print("\nMAE por classe (último seed):")
    for classe, nome in [(0, "No failure"), (1, "ECL"), (2, "EDFA"), (3, "NLI")]:
        mask = ft_te == classe
        if mask.sum() > 0:
            mae_c = mean_absolute_error(y_te[mask], last_y_pr[mask])
            print(f"  {nome:12s}: MAE={mae_c:.1f}s  ({mask.sum():,} amostras)")

    print("\nExperimento 1 concluído.")
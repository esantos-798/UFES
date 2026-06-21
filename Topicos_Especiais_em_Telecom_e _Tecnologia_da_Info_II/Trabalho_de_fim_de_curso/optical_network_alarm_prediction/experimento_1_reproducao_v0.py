"""
Experimento 1 - Reprodução do artigo:
"Proactive soft-failure prediction in optical transport networks
via physics-inspired features and Infrastructure-as-Code orchestration"

Objetivo: reproduzir o resultado principal do artigo usando apenas OSNR
como feature de entrada, com Random Forest regressor.

Referência:
- MAE reportado no artigo: 73.2 ± 0.03 s (real benchmark, n=10 seeds)
- R² reportado: 0.852
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 1. CONFIGURAÇÕES
# =============================================================================

# Ajuste o caminho para a pasta onde estão os arquivos
TRAIN_FILE = "Lightpath_756_label_4_QoT_dataset_train_900.txt"
TEST_FILE  = "Lightpath_756_label_4_QoT_dataset_test_300.txt"

N_TRAIN_SAMPLES = 900   # amostras por lightpath no treino
N_TEST_SAMPLES  = 300   # amostras por lightpath no teste
OSNR_THRESHOLD  = 15.0  # dB — hard-failure threshold (artigo)
TTF_CAP_TRAIN   = 900   # cap de TTF para trajetórias sem falha (treino)
TTF_CAP_TEST    = 300   # cap de TTF para trajetórias sem falha (teste)

# Hiperparâmetros do Random Forest (conforme artigo)
RF_N_ESTIMATORS = 150
RF_MAX_DEPTH    = 12
RF_MIN_SAMPLES_LEAF = 4

# Split por trajetória (conforme artigo: 65% treino, 15% val, 20% teste)
# Usaremos seed=42 para reprodutibilidade
SPLIT_SEED = 42
VAL_FRAC   = 0.15
TEST_FRAC  = 0.20

# Número de seeds para avaliação (artigo usa 10)
N_SEEDS = 10

# =============================================================================
# 2. LEITURA DOS DADOS
# =============================================================================

def ler_arquivo(caminho, n_amostras_por_lp):
    """
    Lê o arquivo do dataset Mendeley e atribui lightpath_id sequencial.
    O timestamp reinicia de 1 a n_amostras_por_lp a cada lightpath.
    """
    print(f"Lendo {caminho}...")
    t0 = time.time()

    df = pd.read_csv(
        caminho,
        sep=r"\s+",          # separador: espaço(s)
        skiprows=2,           # pula failure_description e cabeçalho
        header=None,
        names=["timestamp", "lp_length_km", "laser_current_mA",
               "lp_power_dBm", "osnr_dB", "ber_dB", "failure_type"],
        engine="c",
        dtype={
            "timestamp":        np.int32,
            "lp_length_km":     np.float32,
            "laser_current_mA": np.float32,
            "lp_power_dBm":     np.float32,
            "osnr_dB":          np.float64,
            "ber_dB":           np.float64,
            "failure_type":     np.int8,
        }
    )

    # Atribui lightpath_id: cada bloco de n_amostras_por_lp linhas = 1 lightpath
    df["lightpath_id"] = np.arange(len(df)) // n_amostras_por_lp

    print(f"  {len(df):,} linhas lidas em {time.time()-t0:.1f}s")
    print(f"  {df['lightpath_id'].nunique()} lightpaths")
    print(f"  Distribuição de classes:")
    print(df.groupby("failure_type")["lightpath_id"].nunique().rename(
        {0:"no_failure", 1:"ECL", 2:"EDFA", 3:"NLI"}))
    return df


# =============================================================================
# 3. CÁLCULO DO TTF (Time-to-Failure)
# =============================================================================

def calcular_ttf(df, threshold=OSNR_THRESHOLD, cap=TTF_CAP_TRAIN):
    """
    Para cada lightpath, calcula o TTF em cada timestep:
        TTF_t = min(t_fail - t, cap)
    onde t_fail é o primeiro timestep com OSNR < threshold.
    Trajetórias que nunca cruzam o threshold são censuradas em cap.
    """
    print("Calculando TTF por lightpath...")
    t0 = time.time()

    ttf_list = []

    for lp_id, grupo in df.groupby("lightpath_id", sort=False):
        osnr = grupo["osnr_dB"].values
        n    = len(osnr)

        # Encontra o primeiro timestep abaixo do threshold
        abaixo = np.where(osnr < threshold)[0]
        if len(abaixo) > 0:
            t_fail = abaixo[0]  # índice (0-based)
        else:
            t_fail = n          # nunca falha → censurado

        # TTF para cada amostra
        t_indices = np.arange(n)
        ttf = np.minimum(t_fail - t_indices, cap)
        ttf = np.maximum(ttf, 0)  # garante não-negativo

        ttf_list.append(ttf)

    df["ttf"] = np.concatenate(ttf_list)
    print(f"  TTF calculado em {time.time()-t0:.1f}s")
    print(f"  TTF médio: {df['ttf'].mean():.1f}s | "
          f"Censurados (ttf=={cap}): {(df['ttf']==cap).mean()*100:.1f}%")
    return df


# =============================================================================
# 4. ENGENHARIA DE FEATURES (Physics-inspired, conforme artigo)
# =============================================================================

def extrair_features(df, n_lags=10):
    """
    Extrai features por lightpath:
    - 10 lags de OSNR: SNR_{t-1} ... SNR_{t-10}
    - Velocidade: v_t = SNR_t - SNR_{t-1}
    - Aceleração: a_t = v_t - v_{t-1}
    - Rolling mean (janela 5)
    - Rolling std  (janela 5)
    Total: 15 features (conforme artigo)
    """
    print("Extraindo features physics-inspired...")
    t0 = time.time()

    feature_frames = []

    for lp_id, grupo in df.groupby("lightpath_id", sort=False):
        osnr = grupo["osnr_dB"].values
        n    = len(osnr)

        feats = {}

        # Lags
        for lag in range(1, n_lags + 1):
            col = np.empty(n)
            col[:] = np.nan
            col[lag:] = osnr[:-lag]
            feats[f"snr_lag_{lag}"] = col

        # Velocidade e aceleração
        vel  = np.empty(n); vel[:]  = np.nan
        acel = np.empty(n); acel[:] = np.nan
        vel[1:]  = np.diff(osnr)
        acel[2:] = np.diff(vel[1:])

        feats["velocity"]     = vel
        feats["acceleration"] = acel

        # Rolling mean e std (janela 5, sobre os lags já calculados)
        # Usando os 5 lags mais recentes: lag_1 ... lag_5
        lag_matrix = np.column_stack([feats[f"snr_lag_{i}"] for i in range(1, 6)])
        feats["rolling_mean"] = np.nanmean(lag_matrix, axis=1)
        feats["rolling_std"]  = np.nanstd(lag_matrix,  axis=1)

        feat_df = pd.DataFrame(feats, index=grupo.index)
        feat_df["lightpath_id"] = lp_id
        feat_df["ttf"]          = grupo["ttf"].values
        feat_df["failure_type"] = grupo["failure_type"].values
        feature_frames.append(feat_df)

    result = pd.concat(feature_frames)
    print(f"  Features extraídas em {time.time()-t0:.1f}s")

    # Remove linhas com NaN (primeiras amostras de cada lightpath sem lags suficientes)
    antes = len(result)
    result = result.dropna()
    print(f"  Removidas {antes - len(result):,} linhas com NaN (warm-up de lags)")
    print(f"  Dataset final: {len(result):,} amostras")

    return result


# =============================================================================
# 5. SPLIT POR TRAJETÓRIA
# =============================================================================

def split_por_trajetoria(df_features, val_frac=VAL_FRAC, test_frac=TEST_FRAC,
                          seed=SPLIT_SEED):
    """
    Divide os lightpaths em treino/val/teste por trajetória completa,
    estratificado por classe (conforme artigo).
    Nenhuma amostra do mesmo lightpath aparece em splits diferentes.
    """
    from sklearn.model_selection import train_test_split

    lp_classes = (df_features.groupby("lightpath_id")["failure_type"]
                              .first().reset_index())
    lp_ids     = lp_classes["lightpath_id"].values
    classes    = lp_classes["failure_type"].values

    # Primeiro separa teste
    lp_trainval, lp_test = train_test_split(
        lp_ids, test_size=test_frac,
        stratify=classes, random_state=seed
    )
    classes_trainval = lp_classes.set_index("lightpath_id").loc[lp_trainval, "failure_type"].values

    # Depois separa val do restante
    val_frac_ajustado = val_frac / (1 - test_frac)
    lp_train, lp_val = train_test_split(
        lp_trainval, test_size=val_frac_ajustado,
        stratify=classes_trainval, random_state=seed
    )

    mask_train = df_features["lightpath_id"].isin(lp_train)
    mask_val   = df_features["lightpath_id"].isin(lp_val)
    mask_test  = df_features["lightpath_id"].isin(lp_test)

    print(f"\nSplit por trajetória:")
    print(f"  Treino : {lp_train.shape[0]} lightpaths, {mask_train.sum():,} amostras")
    print(f"  Val    : {lp_val.shape[0]}  lightpaths, {mask_val.sum():,} amostras")
    print(f"  Teste  : {lp_test.shape[0]} lightpaths, {mask_test.sum():,} amostras")

    return (df_features[mask_train], df_features[mask_val],
            df_features[mask_test])


# =============================================================================
# 6. TREINAMENTO E AVALIAÇÃO
# =============================================================================

FEATURE_COLS = ([f"snr_lag_{i}" for i in range(1, 11)] +
                ["velocity", "acceleration", "rolling_mean", "rolling_std"])

def avaliar_modelo(df_train, df_val, df_test, seed=0):
    """Treina RF e retorna métricas no conjunto de teste."""
    X_train = df_train[FEATURE_COLS].values
    y_train = df_train["ttf"].values
    X_test  = df_test[FEATURE_COLS].values
    y_test  = df_test["ttf"].values

    rf = RandomForestRegressor(
        n_estimators     = RF_N_ESTIMATORS,
        max_depth        = RF_MAX_DEPTH,
        min_samples_leaf = RF_MIN_SAMPLES_LEAF,
        n_jobs           = -1,
        random_state     = seed
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    # MAE apenas nas amostras "approaching failure" (TTF < cap, não censuradas)
    mask_approaching = y_test < TTF_CAP_TRAIN
    mae_approaching  = mean_absolute_error(
        y_test[mask_approaching], y_pred[mask_approaching]
    ) if mask_approaching.sum() > 0 else np.nan

    return mae, rmse, r2, mae_approaching, rf


# =============================================================================
# 7. EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("EXPERIMENTO 1 — Reprodução do artigo (OSNR-only, RF)")
    print("=" * 60)

    # --- Leitura ---
    df_train_raw = ler_arquivo(TRAIN_FILE, N_TRAIN_SAMPLES)
    df_test_raw  = ler_arquivo(TEST_FILE,  N_TEST_SAMPLES)

    # --- TTF ---
    df_train_raw = calcular_ttf(df_train_raw, cap=TTF_CAP_TRAIN)
    df_test_raw  = calcular_ttf(df_test_raw,  cap=TTF_CAP_TEST)

    # --- Features ---
    df_train_feat = extrair_features(df_train_raw)
    df_test_feat  = extrair_features(df_test_raw)

    # --- Split por trajetória (sobre o conjunto de treino do dataset) ---
    df_tr, df_val, df_te = split_por_trajetoria(df_train_feat)

    # --- Avaliação multi-seed ---
    print(f"\nTreinando Random Forest com {N_SEEDS} seeds...")
    resultados = []
    for seed in range(N_SEEDS):
        mae, rmse, r2, mae_appr, _ = avaliar_modelo(df_tr, df_val, df_te, seed=seed)
        resultados.append((mae, rmse, r2, mae_appr))
        print(f"  Seed {seed:2d} | MAE={mae:.2f}s | RMSE={rmse:.2f}s | "
              f"R²={r2:.3f} | MAE_approaching={mae_appr:.2f}s")

    maes = np.array([r[0] for r in resultados])
    r2s  = np.array([r[2] for r in resultados])

    # IC 95% via t-distribution
    from scipy import stats
    n   = len(maes)
    ci  = stats.t.ppf(0.975, df=n-1) * maes.std(ddof=1) / np.sqrt(n)

    print("\n" + "=" * 60)
    print("RESULTADO FINAL")
    print("=" * 60)
    print(f"MAE médio : {maes.mean():.2f} ± {ci:.2f} s (IC 95%)")
    print(f"R² médio  : {r2s.mean():.3f}")
    print()
    print("Referência do artigo:")
    print("  MAE : 73.23 ± 0.03 s")
    print("  R²  : 0.852")
    print()

    # --- MAE por classe (usando último modelo treinado) ---
    print("MAE por classe de falha (último seed):")
    _, _, _, _, rf_final = avaliar_modelo(df_tr, df_val, df_te, seed=N_SEEDS-1)
    X_te = df_te[FEATURE_COLS].values
    y_te = df_te["ttf"].values
    y_pr = rf_final.predict(X_te)

    for classe, nome in [(0,"No failure"),(1,"ECL"),(2,"EDFA"),(3,"NLI")]:
        mask = df_te["failure_type"].values == classe
        if mask.sum() > 0:
            mae_c = mean_absolute_error(y_te[mask], y_pr[mask])
            print(f"  {nome:12s}: MAE = {mae_c:.1f}s  ({mask.sum():,} amostras)")

    print("\nExperimento 1 concluído.")

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import time
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURAÇÕES OTIMIZADAS
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
RF_N_JOBS           = -1  # Usa todas as CPUs para andar mais rápido

SPLIT_SEED          = 42
VAL_FRAC            = 0.15
TRAIN_SAMPLE_FRAC   = 0.30
N_SEEDS             = 10

FEATURE_COLS = ([f"snr_lag_{i}" for i in range(1, 11)] +
                ["velocity", "acceleration", "rolling_mean", "rolling_std"])

# =============================================================================
# LEITURA COM FILTRAGEM METODOLÓGICA
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
    
    # [ABORDAGEM A] Focar nas classes onde há assinaturas reais de degradação
    df = df[df["failure_type"].isin([2, 3])].reset_index(drop=True)
    
    print(f"  {len(df):,} linhas filtradas | {df['lightpath_id'].nunique()} lightpaths ativos | {time.time()-t0:.1f}s")
    return df

# =============================================================================
# CÁLCULO DE TTF RETIFICADO (CORREÇÃO DE CANAIS ESTÁVEIS)
# =============================================================================
def calcular_ttf(df, cap):
    print("Calculando TTF...")
    t0 = time.time()
    ttf_list = []
    
    for _, grupo in df.groupby("lightpath_id", sort=False):
        osnr   = grupo["osnr_dB"].values
        n      = len(osnr)
        abaixo = np.where(osnr < OSNR_THRESHOLD)[0]
        
        if len(abaixo) > 0:
            t_fail = abaixo[0]
            # Contagem regressiva real até o ponto exato da falha física
            ttf = np.clip(t_fail - np.arange(n), 0, cap)
        else:
            # Canal estável se mantém estável (Target Fixo)
            ttf = np.full(n, cap, dtype=np.float32)
            
        ttf_list.append(ttf)
        
    df = df.copy()
    df["ttf"] = np.concatenate(ttf_list)
    print(f"  {time.time()-t0:.1f}s | TTF médio: {df['ttf'].mean():.1f}s | Censurados: {(df['ttf']==cap).mean()*100:.1f}%")
    return df

def features_de_um_lp(osnr, ttf, failure_type, lp_id, n_lags=10):
    n    = len(osnr)
    rows = {}

    for lag in range(1, n_lags + 1):
        col       = np.empty(n); col[:] = np.nan
        col[lag:] = osnr[:-lag]
        rows[f"snr_lag_{lag}"] = col

    vel      = np.empty(n); vel[:]  = np.nan
    acel     = np.empty(n); acel[:] = np.nan
    vel[1:]  = np.diff(osnr)
    acel[2:] = np.diff(vel[1:])
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

def split_lightpath_ids(df, val_frac=VAL_FRAC, seed=SPLIT_SEED):
    lp_classes = df.groupby("lightpath_id")["failure_type"].first()
    lp_ids     = lp_classes.index.values
    classes    = lp_classes.values

    lp_tr, lp_val = train_test_split(
        lp_ids, test_size=val_frac, stratify=classes, random_state=seed
    )
    print(f"Split Interno: treino={len(lp_tr)} | val={len(lp_val)} lightpaths")
    return set(lp_tr), set(lp_val)

def extrair_e_separar(df, lp_tr, lp_val, lp_te, cap, sample_frac=1.0, seed=SPLIT_SEED):
    t0  = time.time()
    rng = np.random.default_rng(seed)

    X  = {"train":[], "val":[], "test":[]}
    y  = {"train":[], "val":[], "test":[]}
    ft = {"train":[], "val":[], "test":[]}

    for lp_id, grupo in df.groupby("lightpath_id", sort=False):
        if lp_id in lp_tr:     split = "train"
        elif lp_id in lp_val: split = "val"
        elif lp_id in lp_te:  split = "test"
        else: continue

        osnr         = grupo["osnr_dB"].values
        ttf_vals     = grupo["ttf"].values
        failure_type = int(grupo["failure_type"].iloc[0])

        feats = features_de_um_lp(osnr, ttf_vals, failure_type, lp_id)
        n_feat = len(feats["ttf"])

        if split == "train" and sample_frac < 1.0:
            idx = rng.choice(n_feat, size=max(1, int(n_feat * sample_frac)), replace=False)
            idx.sort()
        else:
            idx = np.arange(n_feat)

        feat_matrix = np.column_stack([feats[c] for c in FEATURE_COLS])
        X[split].append(feat_matrix[idx])
        y[split].append(feats["ttf"][idx])
        ft[split].append(feats["failure_type"][idx])

    result = {}
    for s in ("train","val","test"):
        if len(X[s]) == 0:
            result[s] = (np.empty((0, len(FEATURE_COLS))), np.empty((0,)), np.empty((0,), dtype=np.int8))
        else:
            result[s] = (np.vstack(X[s]), np.concatenate(y[s]), np.concatenate(ft[s]))
        print(f"  {s:5s}: {result[s][0].shape[0]:,} amostras")

    return result

def avaliar(data, seed, cap, peso_rampa=10.0):
    X_tr, y_tr, _   = data["train"]
    X_te, y_te, ft_te = data["test"]

    # Criando o vetor de pesos para o treino:
    # Se o TTF for menor que o CAP (rampa ativa), ganha peso_rampa. Se não, peso padrão 1.0.
    sample_weights = np.where(y_tr < TTF_CAP_TRAIN, peso_rampa, 1.0)

    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        n_jobs=RF_N_JOBS, random_state=seed)
    
    # Passando os pesos diretamente no FIT
    rf.fit(X_tr, y_tr, sample_weight=sample_weights)
    y_pr = rf.predict(X_te)

    mae      = mean_absolute_error(y_te, y_pr)
    rmse     = np.sqrt(mean_squared_error(y_te, y_pr))
    r2       = r2_score(y_te, y_pr)
    
    mask_apr = y_te < cap
    mae_apr  = mean_absolute_error(y_te[mask_apr], y_pr[mask_apr]) if mask_apr.sum() else np.nan

    return mae, rmse, r2, mae_apr, rf, y_te, y_pr, ft_te

# =============================================================================
# EXECUÇÃO DO PIPELINE E GERAÇÃO DE GRÁFICOS
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("EXPERIMENTO 1 — REPRODUÇÃO REAL (Com Geração de Gráficos para o TFC)")
    print("=" * 60)

    df_tr_raw = ler_arquivo(TRAIN_FILE, N_TRAIN_SAMPLES)
    df_te_raw = ler_arquivo(TEST_FILE,  N_TEST_SAMPLES)

    # Backup do dataframe de teste bruto antes de filtrar para podermos indexar os plots por tempo original
    df_te_plots = df_te_raw.copy()

    df_tr_raw = calcular_ttf(df_tr_raw, cap=TTF_CAP_TRAIN)
    df_te_raw = calcular_ttf(df_te_raw, cap=TTF_CAP_TEST)

    lp_tr, lp_val = split_lightpath_ids(df_tr_raw, val_frac=0.15)
    lp_te = set(df_te_raw["lightpath_id"].unique())

    print("\nExtraindo características estruturadas...")
    data_train_val = extrair_e_separar(df_tr_raw, lp_tr, lp_val, set(), cap=TTF_CAP_TRAIN, sample_frac=TRAIN_SAMPLE_FRAC)
    data_test      = extrair_e_separar(df_te_raw, set(), set(), lp_te, cap=TTF_CAP_TEST, sample_frac=1.0)

    data = {
        "train": data_train_val["train"],
        "val":   data_train_val["val"],
        "test":  data_test["test"]
    }
    del df_tr_raw, data_train_val

    print("\nAplicando StandardScaler...")
    scaler = StandardScaler().fit(data["train"][0])
    for s in ("train", "val", "test"):
        X, y, ft = data[s]
        data[s] = (scaler.transform(X), y, ft)

    print(f"\nTreinando Random Forest com Sample Weighting ({N_SEEDS} seeds)...")
    resultados = []
    last_outputs = None
    last_model = None

    for seed in range(N_SEEDS):
        t0 = time.time()
        mae, rmse, r2, mae_apr, rf, y_te, y_pr, ft_te = avaliar(data, seed, cap=TTF_CAP_TEST, peso_rampa=15.0)
        resultados.append((mae, rmse, r2, mae_apr))
        last_outputs = (y_te, y_pr, ft_te)
        last_model = rf  # Guardamos o último modelo treinado para gerar os gráficos
        print(f"  Seed {seed:2d} | MAE Global={mae:.2f}s | MAE Approach={mae_apr:.2f}s | R²={r2:.3f} | {time.time()-t0:.1f}s")

    maes_apr = np.array([r[3] for r in resultados])
    print("\n" + "=" * 60)
    print(f"MAE Approach Médio : {maes_apr.mean():.2f} ± {stats.t.ppf(0.975, df=len(maes_apr)-1) * maes_apr.std(ddof=1) / np.sqrt(len(maes_apr)):.2f} s (IC 95%)")
    print("=" * 60)

    # =============================================================================
    # BLOCO DE GERAÇÃO DE GRÁFICOS (Especial para o Relatório)
    # =============================================================================
    print("\nGerando gráficos de avaliação para o TFC...")
    
    # Vamos re-extrair as features de teste mantendo o vínculo com os IDs físicos dos Lightpaths
    # para plotar a curva contínua temporal de canais específicos
    for classe_id, classe_nome in [(2, "EDFA"), (3, "NLI")]:
        df_classe = df_te_raw[df_te_raw["failure_type"] == classe_id]
        IDs_disponiveis = df_classe["lightpath_id"].unique()
        
        # Sorteia um ID dessa classe para plotar
        lp_escolhido = IDs_disponiveis[0] 
        grupo = df_classe[df_classe["lightpath_id"] == lp_escolhido]
        
        osnr = grupo["osnr_dB"].values
        ttf_real = grupo["ttf"].values
        
        # Extrai features apenas deste LP para o predict continuo
        feats = features_de_um_lp(osnr, ttf_real, classe_id, lp_escolhido)
        X_lp = np.column_stack([feats[c] for c in FEATURE_COLS])
        X_lp_scaled = scaler.transform(X_lp)
        
        # Predição contínua ao longo do tempo do canal
        ttf_predito = last_model.predict(X_lp_scaled)
        ttf_real_valido = feats["ttf"] # ajusta tamanho tirando o warm-up dos lags
        
        # Plotagem
        plt.figure(figsize=(9, 4.5), dpi=300)
        plt.plot(ttf_real_valido, label="TTF Real (Alvo)", color="#2ca02c", linewidth=2.5)
        plt.plot(ttf_predito, label="TTF Predito (Random Forest)", color="#d62728", linestyle="--", linewidth=2)
        
        plt.title(f"Predição de Tempo até a Falha (TTF) - Classe {classe_nome} (LP {lp_escolhido})", fontsize=12, fontweight="bold")
        plt.xlabel("Tempo de Operação / Amostras Sequenciais (Passos)", fontsize=10)
        plt.ylabel("Tempo Restante até a Falha (Segundos)", fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=10, loc="upper right")
        plt.tight_layout()
        
        # Salva a imagem direto na pasta do seu projeto
        nome_arquivo = f"resultado_tfc_ttf_{classe_nome.lower()}.png"
        plt.savefig(nome_arquivo)
        plt.close()
        print(f"  Grafico salvo com sucesso: {nome_arquivo}")

    print("\nPipeline e geração de imagens concluídos com sucesso.")
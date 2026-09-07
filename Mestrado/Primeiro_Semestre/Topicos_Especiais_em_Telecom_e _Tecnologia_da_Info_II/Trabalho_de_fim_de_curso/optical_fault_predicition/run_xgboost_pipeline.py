import json
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
import shap
import matplotlib.pyplot as plt
import os

# ==============================================================================
# CONFIGURAÇÕES DE ISOLAMENTO
# ==============================================================================
RUNS_DIR = Path("results/runs")
ALLOWED_MODELS = ["lstm", "transformer", "lstnet"]
SHAP_OUTPUT_DIR = Path("results/summary/figures/shap")

def find_best_threshold(y_true, y_probs):
    """Encontra o melhor limiar de decisão para maximizar o F1-Score."""
    best_thresh = 0.5
    best_f1 = 0.0
    # Testa thresholds de 0.01 a 0.99
    thresholds = np.linspace(0.01, 0.99, 100)
    for thresh in thresholds:
        preds = (y_probs >= thresh).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh

def generate_shap_plots(xgb_model, X_df, run_name):
    """Gera e salva os plots do SHAP para interpretar o XGBoost."""
    try:
        SHAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Inicializa o TreeExplainer otimizado para árvores (XGBoost)
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer(X_df)
        
        # 1. Summary Plot (Beeswarm) - Distribuição do impacto de cada feature
        plt.figure(figsize=(10, 6))
        shap.plots.beeswarm(shap_values, max_display=15, show=False)
        plt.title(f"SHAP Feature Importance (Beeswarm) - {run_name}", fontsize=11, pad=15)
        plt.tight_layout()
        plt.savefig(SHAP_OUTPUT_DIR / f"{run_name}_shap_beeswarm.png", dpi=300)
        plt.close()
        
        # 2. Bar Plot - Importância Global Absoluta
        plt.figure(figsize=(10, 6))
        shap.plots.bar(shap_values, max_display=15, show=False)
        plt.title(f"SHAP Global Importance (Bar) - {run_name}", fontsize=11, pad=15)
        plt.tight_layout()
        plt.savefig(SHAP_OUTPUT_DIR / f"{run_name}_shap_bar.png", dpi=300)
        plt.close()
        
        print(f"📊 [SHAP] Gráficos gerados com sucesso em: {SHAP_OUTPUT_DIR}")
    except Exception as e:
        print(f"⚠️ [SHAP] Erro ao gerar explicabilidade para {run_name}: {e}")

def run_isolated_xgboost(run_path):
    run_name_lower = run_path.name.lower()
    if not any(model_key in run_name_lower for model_key in ALLOWED_MODELS):
        return

    print(f"\n🤖 [Ajuste de Limiar + SHAP] Processando: {run_path.name}")
    
    errors_file = run_path / "errors.npy"
    labels_file = run_path / "labels.npy"
    
    if not errors_file.exists() or not labels_file.exists():
        return

    run_errors = np.load(errors_file)
    run_labels = np.load(labels_file)
    
    if run_errors.ndim == 1:
        X_local = run_errors.reshape(-1, 1)
    else:
        X_local = run_errors.reshape(run_errors.shape[0], -1)
        
    y_local = run_labels.astype(int)

    # Split 50/50 sequencial fixo para treino/teste do XGBoost
    split_idx = int(len(X_local) * 0.5)
    X_train, X_test = X_local[:split_idx], X_local[split_idx:]
    y_train, y_test = y_local[:split_idx], y_local[split_idx:]
    
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        indices = np.arange(len(X_local))
        np.random.seed(42)
        np.random.shuffle(indices)
        X_local, y_local = X_local[indices], y_local[indices]
        X_train, X_test = X_local[:split_idx], X_local[split_idx:]
        y_train, y_test = y_local[:split_idx], y_local[split_idx:]

    # Converte os dados de teste para DataFrame para manter os nomes das features no SHAP
    num_features = X_test.shape[1]
    if num_features == 1:
        feature_names = ["Erro_Predicao"]
    else:
        # Se for um vetor temporal (ex: passos de tempo do erro), nomeia sequencialmente
        feature_names = [f"Erro_T{i}" for i in range(num_features)]
        
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    # Treina o modelo
    xgb_model = XGBClassifier(
        n_estimators=50, 
        max_depth=3, 
        learning_rate=0.1, 
        random_state=42, 
        eval_metric="logloss"
    )
    xgb_model.fit(X_train_df, y_train)
    
    # Obtém as probabilidades contínuas (em vez de classes discretas 0 ou 1)
    probs_train = xgb_model.predict_proba(X_train_df)[:, 1]
    probs_test = xgb_model.predict_proba(X_test_df)[:, 1]

    # Encontra o limiar ótimo baseado estritamente no conjunto de treino (evita data leakage)
    best_threshold = find_best_threshold(y_train, probs_train)
    
    # Aplica o limiar otimizado no conjunto de teste externo
    xgb_preds = (probs_test >= best_threshold).astype(int)

    # Cálculo das métricas robustas
    f1_res = float(f1_score(y_test, xgb_preds, zero_division=0))
    auc_res = float(roc_auc_score(y_test, probs_test) if len(np.unique(y_test)) > 1 else 0.5)
    prec_res = float(precision_score(y_test, xgb_preds, zero_division=0))
    rec_res = float(recall_score(y_test, xgb_preds, zero_division=0))

    # Executa a análise SHAP usando o conjunto de teste para explicar o comportamento geral externo
    generate_shap_plots(xgb_model, X_test_df, run_path.name)

    # Atualização cirúrgica e segura do JSON
    metrics_path = run_path / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            try: current_metrics = json.load(f)
            except: current_metrics = {}
    else:
        current_metrics = {}

    # Chaves dedicadas para NÃO misturar no script principal
    current_metrics["XGB_Optimized_Thresh"] = float(best_threshold)
    current_metrics["F1_XGB"] = f1_res
    current_metrics["AUC_XGB"] = auc_res
    current_metrics["Precision_XGB"] = prec_res
    current_metrics["Recall_XGB"] = rec_res

    with open(metrics_path, "w") as f:
        json.dump(current_metrics, f, indent=4)
        
    print(f"🎯 Limiar Ótimo: {best_threshold:.4f} | Novo F1_XGB: {f1_res:.4f} | AUC: {auc_res:.4f}")

if __name__ == "__main__":
    for run_folder in RUNS_DIR.iterdir():
        if run_folder.is_dir():
            run_isolated_xgboost(run_folder)
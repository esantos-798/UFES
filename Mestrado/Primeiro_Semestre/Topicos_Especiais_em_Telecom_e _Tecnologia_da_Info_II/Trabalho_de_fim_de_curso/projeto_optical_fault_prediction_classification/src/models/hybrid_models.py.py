import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import shap
from pathlib import Path
from sklearn.metrics import f1_score

class BaseHybridXGBoost:
    """Classe base abstrata para evitar repetição de código comum (XGBoost + SHAP)."""
    def __init__(self, core_model, model_label):
        self.core_model = core_model  # Rede DL Base (LSTM, Transformer ou LSTNet)
        self.model_label = model_label
        self.xgb_model = XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1, 
            random_state=42, eval_metric="logloss"
        )
        self.best_threshold = 0.5
        self.feature_names = None
        self.shap_output_dir = Path("results/summary/figures/shap")

    def _find_best_threshold(self, y_true, y_probs):
        best_thresh = 0.5
        best_f1 = 0.0
        thresholds = np.linspace(0.01, 0.99, 100)
        for thresh in thresholds:
            preds = (y_probs >= thresh).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        return best_thresh

    def _extract_errors(self, X, y_true):
        """Calcula o resíduo/erro da predição do modelo de Deep Learning."""
        # Suporta tanto modelos que retornam (classes,) quanto (predições,) de regressão
        dl_preds = self.core_model.predict(X)
        if isinstance(dl_preds, tuple): 
            dl_preds = dl_preds[0] # Tratamento para saídas multitask se necessário
            
        errors = y_true - dl_preds.squeeze()
        if errors.ndim == 1:
            errors = errors.reshape(-1, 1)
        else:
            errors = errors.reshape(errors.shape[0], -1)
        return errors

    def fit(self, X_train, y_train, **kwargs):
        print(f"🧠 [Híbrido {self.model_label}] Passo 1: Treinando a Rede Deep Learning...")
        # Treina o modelo DL nativo usando os kwargs originais do seu grid (epochs, batch_size...)
        self.core_model.fit(X_train, y_train, **kwargs)
        
        print(f"🌲 [Híbrido {self.model_label}] Passo 2: Treinando o XGBoost nos resíduos...")
        train_errors = self._extract_errors(X_train, y_train)
        
        if self.feature_names is None:
            self.feature_names = [f"Residuo_T{i}" for i in range(train_errors.shape[1])]
            
        X_train_xgb = pd.DataFrame(train_errors, columns=self.feature_names)
        self.xgb_model.fit(X_train_xgb, y_train)
        
        # Otimiza o limiar de decisão
        probs_train = self.xgb_model.predict_proba(X_train_xgb)[:, 1]
        self.best_threshold = self._find_best_threshold(y_train, probs_train)

    def predict_proba(self, X_test, y_test_true=None):
        """Calcula as probabilidades do XGBoost e gera o gráfico SHAP nativo."""
        if y_test_true is None:
            raise ValueError("Para modelos híbridos baseados em erro, 'y_test_true' precisa ser passado na inferência.")
            
        test_errors = self._extract_errors(X_test, y_test_true)
        X_test_xgb = pd.DataFrame(test_errors, columns=self.feature_names)
        
        probs_test = self.xgb_model.predict_proba(X_test_xgb)[:, 1]
        
        # Geração do SHAP Nativo em Tempo de Execução
        try:
            self.shap_output_dir.mkdir(parents=True, exist_ok=True)
            explainer = shap.TreeExplainer(self.xgb_model)
            shap_values = explainer(X_test_xgb)
            
            plt.figure(figsize=(10, 6))
            shap.plots.beeswarm(shap_values, max_display=15, show=False)
            plt.title(f"SHAP Explainer Nativo - {self.model_label} + XGBoost", fontsize=11, pad=15)
            plt.tight_layout()
            plt.savefig(self.shap_output_dir / f"{self.model_label}_native_shap.png", dpi=300)
            plt.close()
            print(f"📊 [SHAP] Gráfico salvo com sucesso para {self.model_label}")
        except Exception as e:
            print(f"⚠️ Erro ao gerar SHAP para {self.model_label}: {e}")
            
        return probs_test

    def predict(self, X_test, y_test_true=None):
        probs = self.predict_proba(X_test, y_test_true)
        return (probs >= self.best_threshold).astype(int)

# ==============================================================================
# MODELOS INDEPENDENTES FINAIS PARA O GRID
# ==============================================================================

class LSTM_XGBoost(BaseHybridXGBoost):
    def __init__(self, core_lstm_model):
        super().__init__(core_model=core_lstm_model, model_label="LSTM")

class Transformer_XGBoost(BaseHybridXGBoost):
    def __init__(self, core_transformer_model):
        super().__init__(core_model=core_transformer_model, model_label="Transformer")

class LSTNet_XGBoost(BaseHybridXGBoost):
    def __init__(self, core_lstnet_model):
        super().__init__(core_model=core_lstnet_model, model_label="LSTNet")
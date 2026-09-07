import torch
import torch.nn as nn
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np

class HybridXGBoostWrapper(nn.Module):
    def __init__(self, core_model, model_name):
        super().__init__()
        self.core_model = core_model  # O modelo base original (LSTM, Transformer, etc.)
        self.model_name = model_name
        self.xgb_model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
        self.is_hybrid = True  # Flag para sabermos que ele tem comportamento híbrido customizado

    def forward(self, x):
        # Repassa o forward padrão para a rede neural do PyTorch
        return self.core_model(x)

    def predict(self, x):
        # Garante compatibilidade caso chamem predict diretamente
        if hasattr(self.core_model, 'predict'):
            return self.core_model.predict(x)
        return self.forward(x)

    def fit_xgboost_and_shap(self, train_loader, val_loader, test_loader, device, output_dir):
        """
        Esta função é chamada dinamicamente no runner para treinar o XGBoost isolado
        com os resíduos em memória DESTE experimento específico, gerando o SHAP único.
        """
        print(f"🌲 Treinando XGBoost isolado para o experimento híbrido: {self.model_name}")
        output_path = Path(output_dir) / "figures" / "shap"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Extração de resíduos (Erros de previsão) e rótulos de falha para a classificação
        def get_residuals(loader):
            self.core_model.eval()
            all_errors, all_labels = [], []
            with torch.no_grad():
                for batch in loader:
                    # Suporte flexível para tuplas de 2 (X, y) ou 3 elementos (X, y, failure)
                    if len(batch) >= 3:
                        batch_x, batch_y, failure_label = batch[0], batch[1], batch[2]
                    else:
                        batch_x, batch_y = batch[0], batch[1]
                        failure_label = batch_y  # Fallback caso y já seja a falha
                    
                    batch_x = batch_x.to(device)
                    out = self.core_model(batch_x)
                    
                    # Adaptação para modelos multitask que retornam tuplas (ex: forecast, failure)
                    if isinstance(out, (tuple, list)): 
                        out = out[0]
                    elif isinstance(out, dict):
                        out = out.get("forecast", out)
                    
                    # Ajusta dimensão temporal se o modelo retornar [batch, seq, feature] e y for [batch, feature]
                    if out.dim() == 3 and batch_y.dim() == 2:
                        out = out[:, -1, :]

                    out_np = out.cpu().numpy()
                    y_np = batch_y.numpy() if isinstance(batch_y, torch.Tensor) else batch_y
                    
                    # Resíduo do Forecast (Erro de Regressão)
                    err = y_np - out_np
                    
                    # Garante formato 2D [batch, features]
                    all_errors.append(err.reshape(err.shape[0], -1))
                    
                    # Salva os rótulos de falha binários para o XGBoost classificar
                    fail_np = failure_label.numpy() if isinstance(failure_label, torch.Tensor) else failure_label
                    all_labels.append(fail_np.ravel())

            return np.vstack(all_errors), np.concatenate(all_labels)

        X_train_err, y_train = get_residuals(train_loader)
        X_test_err, y_test = get_residuals(test_loader)
        
        # Garante que rótulos de treino do XGBoost sejam inteiros/binários
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
        
        # 2. Treino do XGBoost nativo do experimento
        features = [f"Residuo_T{i}" for i in range(X_train_err.shape[1])]
        X_train_df = pd.DataFrame(X_train_err, columns=features)
        X_test_df = pd.DataFrame(X_test_err, columns=features)
        
        self.xgb_model.fit(X_train_df, y_train)
        
        # 3. Geração do Gráfico SHAP independente na pasta correta deste Run
        try:
            explainer = shap.TreeExplainer(self.xgb_model)
            shap_values = explainer(X_test_df)
            plt.figure(figsize=(10, 6))
            shap.plots.beeswarm(shap_values, max_display=15, show=False)
            plt.title(f"SHAP Explainer - {self.model_name.upper()}", fontsize=11, pad=15)
            plt.tight_layout()
            plt.savefig(output_path / "shap_analysis.png", dpi=300)
            plt.close()
            print(f"📊 [SHAP] Gráfico salvo em: {output_path / 'shap_analysis.png'}")
        except Exception as e:
            print(f"⚠️ Erro ao gerar SHAP para {self.model_name}: {e}")

        # 4. Cálculo das métricas de classificação ajustadas com o XGBoost
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
        preds = self.xgb_model.predict(X_test_df)
        
        probs = np.zeros(len(X_test_df))
        if hasattr(self.xgb_model, "predict_proba"):
            p_out = self.xgb_model.predict_proba(X_test_df)
            probs = p_out[:, 1] if p_out.shape[1] > 1 else p_out[:, 0]

        return {
            "F1_XGB": float(f1_score(y_test, preds, zero_division=0)),
            "Precision_XGB": float(precision_score(y_test, preds, zero_division=0)),
            "Recall_XGB": float(recall_score(y_test, preds, zero_division=0)),
            "AUC_XGB": float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else 0.0
        }
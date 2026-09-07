import numpy as np
import torch
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


class HybridClassifierPipeline:
    def __init__(self, model, device, xgb_params=None):
        self.model = model
        self.device = device
        
        # Parâmetros otimizados para dados imbalançados/degradação suave
        self.xgb_params = xgb_params if xgb_params else {
            'n_estimators': 150,
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'logloss',
            'random_state': 42
        }
        self.clf = xgb.XGBClassifier(**self.xgb_params)

    def _extract_latent_features(self, loader):
        """
        Extrai representações aprendidas pela rede neural ou o erro de reconstrução/previsão
        como features para o XGBoost.
        """
        self.model.eval()
        features = []
        labels = []
        
        with torch.no_grad():
            for batch in loader:
                batch_x = batch[0].to(self.device)
                
                if len(batch) > 2:
                    batch_y = batch[2].to(self.device)
                else:
                    batch_y = torch.zeros(batch_x.size(0)).to(self.device)
                
                try:
                    # 1. Se o modelo tiver camadas convolucionais (ex: LSTNet)
                    if hasattr(self.model, 'conv') and not hasattr(self.model, 'network'):
                        if isinstance(self.model.conv, torch.nn.Conv2d):
                            x_in = batch_x.unsqueeze(1)
                            c_out = self.model.conv(x_in)
                        else:
                            c_out = self.model.conv(batch_x.permute(0, 2, 1)).permute(0, 2, 1)
                        feat = c_out.reshape(c_out.size(0), -1)
                    
                    # 2. Se o modelo for Recorrente (LSTM, GRU, BiLSTM), extrai a última camada oculta (hidden state)
                    elif hasattr(self.model, 'lstm') or hasattr(self.model, 'gru') or hasattr(self.model, 'bilstm'):
                        rnn_layer = getattr(self.model, 'lstm', getattr(self.model, 'gru', getattr(self.model, 'bilstm', None)))
                        out, _ = rnn_layer(batch_x)
                        feat = out[:, -1, :]  # Pega o último passo temporal
                        
                    # 3. Se o modelo for um predor/forecast, usa o erro absoluto como feature rica
                    else:
                        output = self.model(batch_x)
                        if isinstance(output, (tuple, list)):
                            output = output[0]
                        elif isinstance(output, dict):
                            output = output.get("forecast", output)
                        
                        # Ajusta dimensão temporal se necessário
                        if output.dim() == 3 and batch_x.dim() == 3:
                            output = output[:, -1, :]
                            target_x = batch_x[:, -1, :]
                        else:
                            target_x = batch_x

                        # Usa o vetor de erro residual como feature discriminativa
                        feat = torch.abs(output - target_x)
                        
                except Exception:
                    # Fallback de segurança usando entrada achatada
                    feat = batch_x.reshape(batch_x.size(0), -1)
                
                features.append(feat.cpu().numpy())
                labels.append(batch_y.cpu().numpy())
                
        return np.vstack(features), np.concatenate(labels)

    def fit_and_evaluate(self, train_loader, val_loader, test_loader):
        """
        Extrai as características latentes, ajusta a razão de peso (scale_pos_weight),
        encontra o melhor limiar no conjunto de Validação e avalia no Teste.
        """
        print("\n--- [XGBoost] Extraindo embeddings temporais profundos ---")
        X_train, y_train = self._extract_latent_features(train_loader)
        X_val, y_val = self._extract_latent_features(val_loader)
        X_test, y_test = self._extract_latent_features(test_loader)

        # Ajuste dinamico de desbalanceamento de classes (scale_pos_weight)
        num_neg = np.sum(y_train == 0)
        num_pos = np.sum(y_train == 1)
        if num_pos > 0:
            scale_pos_weight = num_neg / num_pos
            self.clf.set_params(scale_pos_weight=scale_pos_weight)

        print(f"[XGBoost] Treinando classificador com formato de entrada: {X_train.shape}")
        
        # Treino com conjunto de validação
        self.clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Obtenção de probabilidades contínuas
        val_probs = self.clf.predict_proba(X_val)[:, 1]
        test_probs = self.clf.predict_proba(X_test)[:, 1]

        # --- Otimização de Limiar (Threshold Tuning) na Validação ---
        best_thresh = 0.5
        best_f1_val = -1.0

        for thresh in np.linspace(0.01, 0.99, 99):
            val_preds = (val_probs >= thresh).astype(int)
            score = f1_score(y_val, val_preds, zero_division=0)
            if score > best_f1_val:
                best_f1_val = score
                best_thresh = thresh

        # Aplicação do melhor limiar nos dados de TESTE
        test_preds = (test_probs >= best_thresh).astype(int)

        # Cálculo das métricas finais
        metrics = {
            "F1_XGB": float(f1_score(y_test, test_preds, zero_division=0)),
            "Precision_XGB": float(precision_score(y_test, test_preds, zero_division=0)),
            "Recall_XGB": float(recall_score(y_test, test_preds, zero_division=0)),
            "AUC_XGB": float(roc_auc_score(y_test, test_probs)) if len(np.unique(y_test)) > 1 else 0.0
        }

        # Diagnostics de Variância
        var_train = np.var(X_train, axis=0).mean()
        n_unique_probs = len(np.unique(test_probs))
        
        print(f"X_train Variância Média das Features: {var_train:.6f}")
        print(f"X_test Probabilidades Únicas Geradas: {n_unique_probs}")
        print(f"Melhor Limiar (Threshold) Otimizado : {best_thresh:.4f}")

        print("\n=============================================")
        print("    RESULTADOS DO CLASSIFICADOR HÍBRIDO XGBOOST   ")
        print("=============================================")
        print(f"F1-Score  : {metrics['F1_XGB']:.4f}")
        print(f"Precision : {metrics['Precision_XGB']:.4f}")
        print(f"Recall    : {metrics['Recall_XGB']:.4f}")
        print(f"AUC       : {metrics['AUC_XGB']:.4f}")
        print("=============================================")

        return metrics
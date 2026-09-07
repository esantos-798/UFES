import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


class ForecastEvaluator:

    def __init__(
        self,
        model,
        test_loader,
        device,
        experiment
    ):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.experiment = experiment

    def evaluate(self):
        """
        Executa a avaliação do modelo no conjunto de teste, tratando saídas 
        multi-tarefa e garantindo o alinhamento das dimensões.
        """
        self.model.eval()
        predictions = []
        targets = []

        # Ativa o modo de inferência seguro do PyTorch
        with torch.no_grad():
            for batch_idx, (X, y, failure) in enumerate(self.test_loader):
                X = X.to(self.device)
                outputs = self.model(X)
                
                # 1. Desempacota se for tupla, lista ou dicionário (Suporte Multi-Tarefa)
                if isinstance(outputs, (tuple, list)):
                    forecast_pred = outputs[0]
                elif isinstance(outputs, dict):
                    forecast_pred = outputs.get("forecast")
                else:
                    forecast_pred = outputs

                # 2. Varredura recursiva de segurança
                while isinstance(forecast_pred, (tuple, list)):
                    forecast_pred = forecast_pred[0]

                # 3. Ajusta a dimensão temporal se o modelo retornar [batch, 30, 12]
                if forecast_pred.dim() == 3 and y.dim() == 2:
                    forecast_pred = forecast_pred[:, -1, :]
                
                # 4. Move para CPU, converte para NumPy e armazena os dois lados
                predictions.append(forecast_pred.cpu().numpy())
                targets.append(y.numpy() if isinstance(y, torch.Tensor) else y)

        # [VALIDAÇÃO DE SEGURANÇA] Se o loader falhou por problemas de cache/escopo
        if len(predictions) == 0 or len(targets) == 0:
            raise RuntimeError(
                f"[ERRO CRÍTICO] O test_loader continha 0 batches durante a avaliação do "
                f"modelo. Verifique se o dataset de teste foi resetado ou limpo no escopo anterior."
            )

        # Agrupa os batches em arrays contínuos do NumPy
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)

        # Calcula as métricas científicas completas de regressão/previsão
        return self.calculate_metrics(predictions, targets)

    def calculate_metrics(self, predictions, targets):
        """
        Calcula o conjunto completo de métricas de regressão/forecasting (MSE, RMSE, MAE, R², Pearson).
        """
        # Achata arrays para calcular de forma global/robusta
        p_flat = predictions.ravel()
        t_flat = targets.ravel()

        # 1. Métricas da sklearn / numpy
        mse = mean_squared_error(t_flat, p_flat)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(t_flat, p_flat)
        r2 = r2_score(t_flat, p_flat)

        # 2. Correlação de Pearson (r)
        # Trata o caso isolado onde a variância é 0 para evitar estouro/warning NaN
        if np.std(p_flat) == 0 or np.std(t_flat) == 0:
            pearson_corr = 0.0
        else:
            pearson_corr, _ = pearsonr(t_flat, p_flat)

        metrics = {
            "Forecast_MSE": float(mse),
            "Forecast_RMSE": float(rmse),
            "Forecast_MAE": float(mae),
            "Forecast_R2": float(r2),
            "Forecast_Pearson": float(pearson_corr),
            
            # Aliases em caixa baixa para garantir compatibilidade com parsers legados
            "forecast_mse": float(mse),
            "forecast_rmse": float(rmse),
            "forecast_mae": float(mae),
            "forecast_r2": float(r2),
            "forecast_pearson": float(pearson_corr)
        }

        return metrics
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

import matplotlib.pyplot as plt


class AnomalyDetector:

    def __init__(
        self,
        model,
        val_loader,
        test_loader,
        device,
        alpha=0.5
    ):
        self.model = model
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

        # Split de TESTE (usado só para a métrica final, threshold já fixo)
        self.errors = None
        self.labels = None
        self.predictions = None
        self.failure_scores = None

        # Split de VALIDAÇÃO (usado só para escolher o threshold)
        self.val_errors = None
        self.val_labels = None

        self.threshold = None
        self.alpha = alpha  # Agora recebido dinamicamente

    ############################################################
    # Normaliza saída dos modelos
    ############################################################

    def extract_forecast(self, output):
        """
        Compatível com:
        Tensor
        tuple: (forecast, failure)
        dict: {"forecast": ..., "failure": ...}
        """
        failure_score = None

        if isinstance(output, torch.Tensor):
            forecast = output
        elif isinstance(output, tuple):
            forecast = output[0]
            if len(output) > 1:
                failure_score = output[1]
        elif isinstance(output, dict):
            forecast = output["forecast"]
            if "failure" in output:
                failure_score = output["failure"]
        else:
            raise TypeError(f"Unsupported output type: {type(output)}")

        return forecast, failure_score

    ############################################################
    # Prediction (genérico — roda em qualquer loader)
    ############################################################

    def _run_inference(self, loader):
        """
        Gera erros/labels/failure_scores para um loader qualquer
        (usado tanto para validação quanto para teste).
        """
        self.model.eval()

        errors_list = []
        labels_list = []
        failure_scores_list = []

        with torch.no_grad():
            for X, y, failure in loader:
                X = X.to(self.device)
                outputs = self.model(X)

                forecast_pred, failure_score = self.extract_forecast(outputs)

                if forecast_pred.dim() == 3 and y.dim() == 2:
                    forecast_pred = forecast_pred[:, -1, :]

                mse = torch.mean((forecast_pred - y.to(self.device)) ** 2, dim=1)

                errors_list.extend(mse.cpu().numpy())
                labels_list.extend(failure.cpu().numpy())

                if failure_score is not None:
                    if failure_score.dim() > 1:
                        failure_score = failure_score.squeeze()
                    failure_scores_list.extend(failure_score.cpu().numpy())

        errors = np.array(errors_list)
        labels = np.array(labels_list)
        failure_scores = np.array(failure_scores_list) if len(failure_scores_list) else None

        return errors, labels, failure_scores

    def predict(self):
        """
        Roda a inferência nos dois splits: validação (para escolher o
        threshold) e teste (para a métrica final, reportada).
        """
        self.val_errors, self.val_labels, _ = self._run_inference(self.val_loader)
        self.errors, self.labels, self.failure_scores = self._run_inference(self.test_loader)

        return self.errors, self.labels

    ############################################################
    # Threshold — agora escolhido em VALIDAÇÃO, não em teste
    ############################################################

    def compute_threshold(self):
        best_threshold = None
        best_f1 = -1

        normal_errors = self.val_errors[self.val_labels == 0]

        # --- BUSCA REFINADA DO LIMIAR (sobre validação) ---
        for percentile in np.arange(70.0, 99.9, 0.1):
            threshold = np.percentile(normal_errors, percentile)

            predictions = (self.val_errors > threshold).astype(int)

            f1 = f1_score(
                self.val_labels,
                predictions,
                zero_division=0
            )

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        self.threshold = best_threshold

        print(f"\nBest threshold found (on validation): {best_threshold:.6f}")
        print(f"Best validation F1 : {best_f1:.4f}")

        return self.threshold

    ############################################################
    # Detection — aplicado no split de TESTE, com threshold já fixo
    ############################################################

    def detect(self):
        self.predictions = (self.errors > self.threshold).astype(int)
        return self.predictions

    ############################################################
    # Metrics (sempre sobre teste)
    ############################################################

    def compute_metrics(self):
        accuracy = accuracy_score(self.labels, self.predictions)
        precision = precision_score(self.labels, self.predictions, zero_division=0)
        recall = recall_score(self.labels, self.predictions, zero_division=0)
        f1 = f1_score(self.labels, self.predictions, zero_division=0)

        try:
            auc = roc_auc_score(self.labels, self.errors)
        except ValueError:
            auc = 0.0

        tn, fp, fn, tp = confusion_matrix(self.labels, self.predictions).ravel()

        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        mdr = fn / (fn + tp) if (fn + tp) > 0 else 0

        return {
            "Threshold": float(self.threshold),
            "Accuracy": float(accuracy),
            "Precision": float(precision),
            "Recall": float(recall),
            "F1": float(f1),
            "AUC": float(auc),
            "False Alarm Rate": float(far),
            "Miss Detection Rate": float(mdr),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        }

    ############################################################
    # Lead Time
    ############################################################

    def lead_time(self):
        lead_times = []
        failures = np.where(self.labels == 1)[0]

        for idx in failures:
            start = max(0, idx - 50)
            detected = np.where(self.predictions[start:idx] == 1)[0]

            if len(detected):
                lead = idx - (start + detected[-1])
                lead_times.append(lead)

        return lead_times

    ############################################################
    # Complete evaluation
    ############################################################

    def evaluate(self, plot=False):
        self.predict()          # popula val (threshold) e test (métrica) separadamente
        self.compute_threshold()  # escolhido em validação
        self.detect()            # aplicado em teste

        metrics = self.compute_metrics()
        lead = self.lead_time()

        metrics.update({
            "Average Lead Time": float(np.mean(lead)) if len(lead) else 0,
            "Maximum Lead Time": int(np.max(lead)) if len(lead) else 0,
            "Minimum Lead Time": int(np.min(lead)) if len(lead) else 0,
            "Detected Failures": int(np.sum(self.predictions)),
            "Real Failures": int(np.sum(self.labels))
        })

        if plot:
            self.plot_score_distribution()
        return metrics

    def plot_score_distribution(self):
        normal = self.errors[self.labels == 0]
        failure = self.errors[self.labels == 1]

        plt.figure(figsize=(10, 5))
        plt.hist(normal, bins=50, alpha=0.6, density=True, label="Normal")
        plt.hist(failure, bins=50, alpha=0.6, density=True, label="Failure")
        plt.axvline(self.threshold, color="red", linestyle="--", label="Threshold")

        plt.xlabel("Anomaly score")
        plt.ylabel("Density")
        plt.title(f"Distribution of anomaly scores (Alpha: {self.alpha})")
        plt.legend()
        plt.grid(True)
        plt.show()
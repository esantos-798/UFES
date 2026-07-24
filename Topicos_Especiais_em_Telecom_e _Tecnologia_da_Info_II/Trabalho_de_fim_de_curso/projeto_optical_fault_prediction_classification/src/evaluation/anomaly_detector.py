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

        # Metrics & Predictions - Split TEST
        self.errors = None
        self.labels = None
        self.predictions = None
        self.failure_scores = None

        # Metrics & Predictions - Split VALIDATION
        self.val_errors = None
        self.val_labels = None

        # Scaling stats (Evita Data Leakage entre Val e Test)
        self.min_mse = None
        self.max_mse = None

        self.threshold = None
        self.alpha = alpha

    def extract_forecast(self, output):
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

    def _run_inference(self, loader, is_validation=False):
        self.model.eval()

        mse_list = []
        labels_list = []
        failure_scores_list = []

        with torch.no_grad():
            for X, y, failure in loader:
                X = X.to(self.device)
                outputs = self.model(X)

                forecast_pred, failure_score = self.extract_forecast(outputs)

                if forecast_pred.dim() == 3 and y.dim() == 2:
                    forecast_pred = forecast_pred[:, -1, :]

                mse = torch.mean(
                    (forecast_pred - y.to(self.device)) ** 2,
                    dim=1
                )

                mse_list.extend(mse.cpu().numpy())
                labels_list.extend(failure.cpu().numpy())

                if failure_score is not None:
                    if failure_score.dim() > 1:
                        failure_score = failure_score.squeeze()

                    prob = torch.sigmoid(failure_score)
                    failure_scores_list.extend(prob.cpu().numpy())

        mse = np.array(mse_list)
        labels = np.array(labels_list)
        failure_scores = (
            np.array(failure_scores_list)
            if len(failure_scores_list)
            else None
        )

        # Atualiza parâmetros de escala no Validation e reutiliza no Test
        if is_validation or self.min_mse is None:
            self.min_mse = mse.min()
            self.max_mse = mse.max()

        mse_norm = (mse - self.min_mse) / (self.max_mse - self.min_mse + 1e-8)

        if failure_scores is not None:
            anomaly_score = (
                self.alpha * mse_norm +
                (1 - self.alpha) * failure_scores
            )
        else:
            anomaly_score = mse_norm

        return anomaly_score, labels, failure_scores

    def predict(self):
        # Valida primeiro para fixar escala min/max do MSE
        self.val_errors, self.val_labels, _ = self._run_inference(self.val_loader, is_validation=True)
        self.errors, self.labels, self.failure_scores = self._run_inference(self.test_loader, is_validation=False)

        return self.errors, self.labels

    def compute_threshold(self):
        thresholds = np.arange(0.01, 0.99, 0.005)
        preds = self.val_errors > thresholds[:, None]

        tp = (preds & (self.val_labels == 1)).sum(axis=1)
        fp = (preds & (self.val_labels == 0)).sum(axis=1)
        fn = ((~preds) & (self.val_labels == 1)).sum(axis=1)

        f1_scores = np.where((2 * tp + fp + fn) > 0, (2 * tp) / (2 * tp + fp + fn), 0.0)

        best_idx = np.argmax(f1_scores)
        self.threshold = float(thresholds[best_idx])
        best_f1 = f1_scores[best_idx]

        print(f"\nBest threshold found (on validation): {self.threshold:.6f}")
        print(f"Best validation F1 : {best_f1:.4f}")

        return self.threshold

    def detect(self):
        if self.threshold is None:
            print("[Aviso] self.threshold era None. Atribuindo valor padrão de fallback (0.05).")
            self.threshold = 0.05

        self.predictions = (self.errors > self.threshold).astype(int)
        return self.predictions

    def compute_metrics(self):
        accuracy = accuracy_score(self.labels, self.predictions)
        precision = precision_score(self.labels, self.predictions, zero_division=0)
        recall = recall_score(self.labels, self.predictions, zero_division=0)
        f1 = f1_score(self.labels, self.predictions, zero_division=0)

        try:
            auc = roc_auc_score(self.labels, self.errors)
        except ValueError:
            auc = 0.0

        # Mapeamento explícito de labels para evitar falhas em matrizes 1x1
        tn, fp, fn, tp = confusion_matrix(self.labels, self.predictions, labels=[0, 1]).ravel()

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

    def evaluate(self, plot=False):
        self.predict()
        self.compute_threshold()
        self.detect()

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
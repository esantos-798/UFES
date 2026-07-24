import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

class ExperimentLogger:

    def __init__(self, experiment):
        self.experiment = experiment
        self.exp_dir = Path(experiment.output_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    def save_config(self):
        config = {}
        for key, value in vars(self.experiment).items():
            try:
                json.dumps(value)
                config[key] = value
            except TypeError:
                config[key] = str(value)

        with open(self.exp_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

    def save_metrics(self, metrics):
        metrics_json = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics_json[key] = value.tolist()
            elif isinstance(value, np.generic):
                metrics_json[key] = value.item()
            else:
                metrics_json[key] = value

        with open(self.exp_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_json, f, indent=4)

    def save_history(self, history):
        clean_history = {}
        for key, value in history.items():
            if isinstance(value, (list, tuple)):
                clean_history[key] = list(map(float, value))
            else:
                clean_history[key] = float(value)

        with open(self.exp_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(clean_history, f, indent=4)

    def save_model(self, model):
        if model is not None:
            torch.save(model.state_dict(), self.exp_dir / "model.pt")

    def save_threshold(self, threshold):
        if threshold is not None:
            with open(self.exp_dir / "threshold.txt", "w") as f:
                f.write(str(float(threshold)))

    def save_predictions(self, predictions):
        if predictions is not None:
            np.save(self.exp_dir / "predictions.npy", predictions)

    def save_labels(self, labels):
        if labels is not None:
            np.save(self.exp_dir / "labels.npy", labels)

    def save_errors(self, errors):
        if errors is not None:
            np.save(self.exp_dir / "errors.npy", errors)

    def save_loss_plot(self, history):
        if "train_loss" in history and "val_loss" in history:
            train = history["train_loss"]
            val = history["val_loss"]
            plt.figure(figsize=(8, 5))
            plt.plot(train, label="Train")
            plt.plot(val, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(self.experiment.name)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(self.exp_dir / "loss.png", dpi=300)
            plt.close()

    def save_all(self, model, metrics, history, predictions=None, labels=None, errors=None, threshold=None):
        self.save_config()
        self.save_metrics(metrics)
        self.save_history(history)
        self.save_model(model)
        self.save_loss_plot(history)
        self.save_predictions(predictions)
        self.save_labels(labels)
        self.save_errors(errors)
        self.save_threshold(threshold)
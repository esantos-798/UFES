import json
from pathlib import Path

import numpy as np
import torch

import matplotlib.pyplot as plt


class ExperimentLogger:

    def __init__(self, experiment):

        self.experiment = experiment

        self.exp_dir = Path(
            experiment.output_dir
        )

        self.exp_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # ==========================================================
    # CONFIG
    # ==========================================================

    def save_config(self):

        config = {}

        for key, value in vars(self.experiment).items():

            try:

                json.dumps(value)

                config[key] = value

            except TypeError:

                config[key] = str(value)

        with open(

            self.exp_dir / "config.json",

            "w",

            encoding="utf-8"

        ) as f:

            json.dump(

                config,

                f,

                indent=4

            )

    # ==========================================================
    # METRICS
    # ==========================================================

    def save_metrics(self, metrics):

        metrics_json = {}

        for key, value in metrics.items():

            if isinstance(value, np.ndarray):

                metrics_json[key] = value.tolist()

            elif isinstance(value, np.generic):

                metrics_json[key] = value.item()

            else:

                metrics_json[key] = value

        with open(

            self.exp_dir / "metrics.json",

            "w",

            encoding="utf-8"

        ) as f:

            json.dump(

                metrics_json,

                f,

                indent=4

            )

    # ==========================================================
    # HISTORY
    # ==========================================================

    def save_history(self, history):

        clean_history = {}

        for key, value in history.items():

            if isinstance(value, (list, tuple)):

                clean_history[key] = list(
                    map(float, value)
                )

            else:

                clean_history[key] = float(value)


        path = self.exp_dir / "history.json"

        with open(path,"w") as f:

            json.dump(
                clean_history,
                f,
                indent=4
            )
    # ==========================================================
    # MODEL
    # ==========================================================

    def save_model(self, model):

        torch.save(

            model.state_dict(),

            self.exp_dir / "model.pt"

        )

    # ==========================================================
    # THRESHOLD
    # ==========================================================

    def save_threshold(self, threshold):

        with open(

            self.exp_dir / "threshold.txt",

            "w"

        ) as f:

            f.write(str(float(threshold)))

    # ==========================================================
    # NUMPY FILES
    # ==========================================================

    def save_predictions(self, predictions):

        np.save(

            self.exp_dir / "predictions.npy",

            predictions

        )

    def save_labels(self, labels):

        np.save(

            self.exp_dir / "labels.npy",

            labels

        )

    def save_errors(self, errors):

        np.save(

            self.exp_dir / "errors.npy",

            errors

        )

    # ==========================================================
    # LOSS PLOT
    # ==========================================================

    def save_loss_plot(self, history):

        train = history["train_loss"]

        val = history["val_loss"]

        plt.figure(figsize=(8,5))

        plt.plot(

            train,

            label="Train"

        )

        plt.plot(

            val,

            label="Validation"

        )

        plt.xlabel("Epoch")

        plt.ylabel("Loss")

        plt.title(

            self.experiment.name

        )

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(

            self.exp_dir / "loss.png",

            dpi=300

        )

        plt.close()

    # ==========================================================
    # COMPLETE SAVE
    # ==========================================================

    def save_everything(

        self,

        metrics,

        history,

        model,

        predictions=None,

        labels=None,

        errors=None,

        threshold=None

    ):

        self.save_config()

        self.save_metrics(metrics)

        self.save_history(history)

        self.save_model(model)

        self.save_loss_plot(history)

        if predictions is not None:

            self.save_predictions(

                predictions

            )

        if labels is not None:

            self.save_labels(

                labels

            )

        if errors is not None:

            self.save_errors(

                errors

            )

        if threshold is not None:

            self.save_threshold(

                threshold

            )


    def save_all(
        self,
        metrics,
        history,
        model=None,
        predictions=None,
        labels=None,
        errors=None,
        threshold=None
    ):

        return self.save_everything(
            metrics=metrics,
            history=history,
            model=model,
            predictions=predictions,
            labels=labels,
            errors=errors,
            threshold=threshold
        )
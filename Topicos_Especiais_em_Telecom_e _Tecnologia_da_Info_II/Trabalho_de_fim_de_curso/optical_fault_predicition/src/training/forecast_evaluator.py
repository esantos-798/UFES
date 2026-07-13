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

        self.model.eval()

        predictions = []

        targets = []

        with torch.no_grad():

            for X, y, _ in self.test_loader:

                X = X.to(self.device)

                pred = self.model(X)

                predictions.append(
                    pred.cpu().numpy()
                )

                targets.append(
                    y.numpy()
                )

        predictions = np.concatenate(
            predictions,
            axis=0
        )

        targets = np.concatenate(
            targets,
            axis=0
        )

        metrics = {

            "MSE":
                mean_squared_error(
                    targets,
                    predictions
                ),

            "RMSE":
                np.sqrt(
                    mean_squared_error(
                        targets,
                        predictions
                    )
                ),

            "MAE":
                mean_absolute_error(
                    targets,
                    predictions
                ),

            "R2":
                r2_score(
                    targets,
                    predictions
                ),

            "Pearson":
                pearsonr(
                    targets.flatten(),
                    predictions.flatten()
                )[0]
        }

        self.experiment.save_metrics(
            metrics
        )

        return metrics
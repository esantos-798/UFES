import torch

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from scipy.stats import pearsonr


class Evaluator:

    def __init__(

        self,

        model,

        test_loader,

        device,

        task="classification"

    ):

        self.model = model

        self.test_loader = test_loader

        self.device = device

        self.task = task

    def evaluate(self):

        if self.task == "classification":

            return self.evaluate_classification()

        elif self.task == "forecast":

            return self.evaluate_forecast()

        else:

            raise ValueError(
                f"Unknown task: {self.task}"
            )
        
    def evaluate_classification(self):

        self.model.eval()

        predictions = []

        probabilities = []

        targets = []

        with torch.no_grad():

            for X, y in self.test_loader:

                X = X.to(self.device)

                output = self.model(X)

                prob = torch.sigmoid(output)

                pred = (prob > 0.5).float()

                predictions.extend(
                    pred.cpu().numpy().flatten()
                )

                probabilities.extend(
                    prob.cpu().numpy().flatten()
                )

                targets.extend(
                    y.numpy().flatten()
                )

        predictions = np.array(predictions)

        probabilities = np.array(probabilities)

        targets = np.array(targets)

        return {

            "Accuracy": accuracy_score(
                targets,
                predictions
            ),

            "Precision": precision_score(
                targets,
                predictions,
                zero_division=0
            ),

            "Recall": recall_score(
                targets,
                predictions
            ),

            "F1": f1_score(
                targets,
                predictions
            ),

            "AUC": roc_auc_score(
                targets,
                probabilities
            ),

            "Confusion Matrix": confusion_matrix(
                targets,
                predictions
            )

        }
    
    def evaluate_forecast(self):

        self.model.eval()

        predictions = []

        targets = []

        with torch.no_grad():

            for X, y in self.test_loader:

                X = X.to(self.device)

                output = self.model(X)

                predictions.append(

                    output.cpu().numpy()

                )

                targets.append(

                    y.numpy()

                )

        predictions = np.vstack(
            predictions
        )

        targets = np.vstack(
            targets
        )

        mse = mean_squared_error(
            targets,
            predictions
        )

        rmse = np.sqrt(mse)

        mae = mean_absolute_error(
            targets,
            predictions
        )

        r2 = r2_score(
            targets,
            predictions
        )

        pearson = pearsonr(

            targets.flatten(),

            predictions.flatten()

        )[0]

        return {

            "MSE": mse,

            "RMSE": rmse,

            "MAE": mae,

            "R2": r2,

            "Pearson": pearson

        }
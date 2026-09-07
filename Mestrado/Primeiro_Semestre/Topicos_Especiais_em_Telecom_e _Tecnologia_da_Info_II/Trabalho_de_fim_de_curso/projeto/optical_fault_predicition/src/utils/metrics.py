import numpy as np

from scipy.stats import pearsonr

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


def forecast_metrics(y_true, y_pred):

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    pearson = pearsonr(

        y_true.flatten(),

        y_pred.flatten()

    )[0]

    return {

        "MSE": mse,

        "RMSE": rmse,

        "MAE": mae,

        "R2": r2,

        "Pearson": pearson

    }
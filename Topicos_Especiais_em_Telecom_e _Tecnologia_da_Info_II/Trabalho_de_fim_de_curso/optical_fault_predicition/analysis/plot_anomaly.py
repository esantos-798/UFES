import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ======================================================
# Arquivos gerados pelo AnomalyDetector
# ======================================================

experiment_dir = Path(
    "experiments/lstm_forecast_hard_failure"
)


errors = np.load(
    experiment_dir / "errors.npy"
)


labels = np.load(
    experiment_dir / "labels.npy"
)



# ======================================================
# Recupera threshold usado
# ======================================================

import json


with open(
    experiment_dir / "metrics.json",
    "r"
) as f:

    metrics = json.load(f)


threshold = metrics["Threshold"]



print("Samples:", len(errors))

print("Threshold:", threshold)

print(
    "Failures:",
    np.sum(labels)
)



# ======================================================
# Plot erro de previsão
# ======================================================

plt.figure(
    figsize=(14,5)
)


plt.plot(
    errors,
    label="Prediction Error"
)


plt.axhline(
    threshold,
    linestyle="--",
    label="Threshold"
)



# marca falhas

failure_index = np.where(
    labels == 1
)[0]


plt.scatter(

    failure_index,

    errors[failure_index],

    marker="x",

    label="Hard Failure"

)



plt.xlabel(
    "Time Window"
)


plt.ylabel(
    "MSE Error"
)


plt.title(
    "LSTM Forecast Error - Hard Failure Detection"
)


plt.legend()


plt.grid(
    True
)


plt.tight_layout()



plt.savefig(

    experiment_dir /
    "anomaly_detection_plot.png",

    dpi=300

)


plt.show()
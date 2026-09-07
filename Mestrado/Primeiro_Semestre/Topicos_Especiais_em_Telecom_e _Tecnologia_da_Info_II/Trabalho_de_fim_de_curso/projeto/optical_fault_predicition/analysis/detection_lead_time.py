import numpy as np
import json
from pathlib import Path


# ======================================================
# Diretório do experimento
# ======================================================

#experiment_dir = Path(
#    "experiments/lstm_forecast_hard_failure"
#)

#experiment_dir = Path(
#    "experiments/lstnet_forecast_hard_failure"
#)

#experiment_dir = Path(
#    "experiments/lstnet_v2_forecast_hard_failure"
#)

#experiment_dir = Path(
#    "experiments/attention_lstnet_forecast_hard_failure"
#)

experiment_dir = Path(
    "experiments/multitask_lstnet_forecast_hard_failure"
)    

# ======================================================
# Carrega dados
# ======================================================

errors = np.load(
    experiment_dir / "errors.npy"
)


labels = np.load(
    experiment_dir / "labels.npy"
)


with open(
    experiment_dir / "metrics.json",
    "r"
) as f:

    metrics = json.load(f)


threshold = metrics["Threshold"]



print("=" * 60)
print("DETECTION LEAD TIME ANALYSIS")
print("=" * 60)


print()

print(
    "Samples:",
    len(errors)
)

print(
    "Failures:",
    np.sum(labels)
)

print(
    "Threshold:",
    threshold
)



# ======================================================
# Pontos detectados
# ======================================================

alarms = errors > threshold



failure_points = np.where(
    labels == 1
)[0]



print()

print(
    "Failure events:",
    len(failure_points)
)



# ======================================================
# Calcula lead time
# ======================================================

lead_times = []


for failure in failure_points:


    # procura alarmes anteriores

    previous_alarms = np.where(

        alarms[:failure]

    )[0]


    if len(previous_alarms) == 0:

        continue


    last_alarm = previous_alarms[-1]


    lead_time = (

        failure
        -
        last_alarm

    )


    lead_times.append(
        lead_time
    )



# ======================================================
# Resultados
# ======================================================

print()


if len(lead_times) == 0:

    print(
        "No early detections found"
    )


else:


    lead_times = np.array(
        lead_times
    )


    print(
        "Detected failures:",
        len(lead_times)
    )


    print()


    print(
        "Average lead time:",
        np.mean(lead_times)
    )


    print(
        "Maximum lead time:",
        np.max(lead_times)
    )


    print(
        "Minimum lead time:",
        np.min(lead_times)
    )


    print()


    print(
        "Lead time distribution:"
    )


    unique, counts = np.unique(
        lead_times,
        return_counts=True
    )


    for u, c in zip(
        unique,
        counts
    ):

        print(
            f"{u} windows: {c}"
        )


    # salva resultado

    result = {

        "detected_failures":

            int(len(lead_times)),


        "average_lead_time":

            float(np.mean(lead_times)),


        "max_lead_time":

            int(np.max(lead_times)),


        "min_lead_time":

            int(np.min(lead_times))

    }


    with open(

        experiment_dir /
        "lead_time.json",

        "w"

    ) as f:

        json.dump(

            result,

            f,

            indent=4

        )


print()

print(
    "Finished."
)
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)


class ThresholdOptimizer:


    def __init__(
        self,
        errors,
        labels
    ):

        self.errors = np.array(errors)

        self.labels = np.array(labels)



    def search(
        self,
        steps=200
    ):


        thresholds = np.linspace(

            self.errors.min(),

            self.errors.max(),

            steps

        )


        best = None


        for threshold in thresholds:


            predictions = (
                self.errors > threshold
            ).astype(int)



            f1 = f1_score(

                self.labels,

                predictions,

                zero_division=0

            )


            precision = precision_score(

                self.labels,

                predictions,

                zero_division=0

            )


            recall = recall_score(

                self.labels,

                predictions,

                zero_division=0

            )


            if (

                best is None

                or

                f1 > best["F1"]

            ):


                best = {

                    "Threshold": float(threshold),

                    "Precision": float(precision),

                    "Recall": float(recall),

                    "F1": float(f1),

                    "Confusion Matrix":
                        confusion_matrix(
                            self.labels,
                            predictions
                        ).tolist()

                }


        return best
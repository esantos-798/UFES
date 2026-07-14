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



class AnomalyDetector:


    def __init__(

        self,

        model,

        test_loader,

        device

    ):


        self.model = model

        self.test_loader = test_loader

        self.device = device


        self.errors = None

        self.labels = None

        self.predictions = None

        self.threshold = None


        self.failure_scores = None

        self.alpha = 0.5



    ############################################################

    # Normaliza saída dos modelos

    ############################################################


    def extract_forecast(

        self,

        output

    ):


        """

        Compatível com:

        Tensor

        tuple:

            forecast, failure


        dict:

            {

              forecast:

              failure:

            }

        """


        failure_score = None



        if isinstance(

            output,

            torch.Tensor

        ):


            forecast = output



        elif isinstance(

            output,

            tuple

        ):


            forecast = output[0]


            if len(output) > 1:

                failure_score = output[1]



        elif isinstance(

            output,

            dict

        ):


            forecast = output["forecast"]


            if "failure" in output:

                failure_score = output["failure"]



        else:


            raise TypeError(

                f"Unsupported output type: {type(output)}"

            )



        return forecast, failure_score



    ############################################################

    # Prediction

    ############################################################


    def predict(self):


        self.model.eval()



        errors = []

        labels = []

        failure_scores = []



        with torch.no_grad():


            for batch in self.test_loader:



                ################################################

                # Dataset atual

                ################################################


                if len(batch) == 3:


                    X, y, failure = batch

                    label = failure



                else:


                    X, y = batch

                    label = y



                X = X.to(

                    self.device

                )


                y = y.to(

                    self.device

                )



                output = self.model(X)



                prediction, failure_score = self.extract_forecast(

                    output

                )



                ################################################

                # erro de previsão

                ################################################


                mse = torch.mean(
                    (prediction - y) ** 2,
                    dim=1
                )


                if failure_score is not None:

                    failure_score = failure_score.squeeze()


                    score = (

                        self.alpha * mse

                        +

                        (1-self.alpha) * failure_score

                    )

                else:

                    score = mse


                errors.extend(

                    score.cpu().numpy()

                )


                labels.extend(

                    label.cpu().numpy()

                )



                if failure_score is not None:


                    failure_scores.extend(

                        failure_score.cpu().numpy()

                    )



        self.errors = np.array(

            errors

        )


        self.labels = np.array(

            labels

        )



        if len(failure_scores):


            self.failure_scores = np.array(

                failure_scores

            )



        return (

            self.errors,

            self.labels

        )


    ############################################################
    # Threshold
    ############################################################


    def compute_threshold(self):


        normal_errors = self.errors[

            self.labels == 0

        ]



        self.threshold = np.percentile(

            normal_errors,

            95

        )



        return self.threshold



    ############################################################
    # Detection
    ############################################################


    def detect(self):


        self.predictions = (

            self.errors > self.threshold

        ).astype(int)



        return self.predictions



    ############################################################
    # Metrics
    ############################################################


    def compute_metrics(self):


        accuracy = accuracy_score(

            self.labels,

            self.predictions

        )


        precision = precision_score(

            self.labels,

            self.predictions,

            zero_division=0

        )


        recall = recall_score(

            self.labels,

            self.predictions,

            zero_division=0

        )


        f1 = f1_score(

            self.labels,

            self.predictions,

            zero_division=0

        )



        ########################################################

        # AUC

        ########################################################


        try:


            auc = roc_auc_score(

                self.labels,

                self.errors

            )


        except ValueError:


            auc = 0.0



        ########################################################

        # Confusion Matrix

        ########################################################


        tn, fp, fn, tp = confusion_matrix(

            self.labels,

            self.predictions

        ).ravel()



        far = (

            fp /

            (fp + tn)

            if (fp + tn) > 0

            else 0

        )


        mdr = (

            fn /

            (fn + tp)

            if (fn + tp) > 0

            else 0

        )



        return {


            "Threshold":

                float(self.threshold),


            "Accuracy":

                float(accuracy),


            "Precision":

                float(precision),


            "Recall":

                float(recall),


            "F1":

                float(f1),


            "AUC":

                float(auc),



            "False Alarm Rate":

                float(far),


            "Miss Detection Rate":

                float(mdr),



            "TN":

                int(tn),


            "FP":

                int(fp),


            "FN":

                int(fn),


            "TP":

                int(tp)

        }



    ############################################################
    # Lead Time
    ############################################################


    def lead_time(self):


        lead_times = []



        failures = np.where(

            self.labels == 1

        )[0]



        for idx in failures:



            start = max(

                0,

                idx - 50

            )



            detected = np.where(

                self.predictions[start:idx]

                ==

                1

            )[0]



            if len(detected):


                lead = idx - (

                    start +

                    detected[-1]

                )


                lead_times.append(

                    lead

                )



        return lead_times



    ############################################################
    # Complete evaluation
    ############################################################


    def evaluate(self):


        self.predict()



        self.compute_threshold()



        self.detect()



        metrics = self.compute_metrics()



        lead = self.lead_time()



        metrics.update({


            "Average Lead Time":

                (

                    float(np.mean(lead))

                    if len(lead)

                    else 0

                ),



            "Maximum Lead Time":

                (

                    int(np.max(lead))

                    if len(lead)

                    else 0

                ),



            "Minimum Lead Time":

                (

                    int(np.min(lead))

                    if len(lead)

                    else 0

                ),


            "Detected Failures":

                int(

                    np.sum(self.predictions)

                ),


            "Real Failures":

                int(

                    np.sum(self.labels)

                )

        })



        return metrics    
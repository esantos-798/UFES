import os
import json
import torch


class ExperimentSaver:


    def __init__(self, experiment):

        self.experiment = experiment

        self.path = os.path.join(
            "experiments",
            experiment.name
        )


        os.makedirs(
            self.path,
            exist_ok=True
        )



    def save_config(self):

        config = {

            "model":
                self.experiment.model,

            "task":
                self.experiment.task,

            "dataset":
                self.experiment.dataset,

            "epochs":
                self.experiment.epochs,

            "batch_size":
                self.experiment.batch_size,

            "lr":
                self.experiment.lr

        }


        with open(
            os.path.join(
                self.path,
                "config.json"
            ),
            "w"
        ) as f:

            json.dump(
                config,
                f,
                indent=4
            )



    def save_history(
        self,
        history
    ):

        with open(
            os.path.join(
                self.path,
                "history.json"
            ),
            "w"
        ) as f:

            json.dump(
                history,
                f,
                indent=4
            )



    def save_model(
        self,
        model
    ):


        torch.save(

            model.state_dict(),

            os.path.join(
                self.path,
                "model.pt"
            )

        )



    def save_metrics(
        self,
        metrics
    ):


        with open(
            os.path.join(
                self.path,
                "metrics.json"
            ),
            "w"
        ) as f:

            json.dump(
                metrics,
                f,
                indent=4
            )
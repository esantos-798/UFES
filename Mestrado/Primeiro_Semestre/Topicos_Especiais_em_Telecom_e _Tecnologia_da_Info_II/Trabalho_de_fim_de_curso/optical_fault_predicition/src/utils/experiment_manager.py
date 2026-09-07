import os
import json
import torch
from datetime import datetime


class ExperimentManager:


    def __init__(
        self,
        name,
        base_dir="experiments"
    ):

        self.name = name

        self.path = os.path.join(
            base_dir,
            name
        )

        os.makedirs(
            self.path,
            exist_ok=True
        )


        self.metadata = {

            "experiment": name,

            "created":
                datetime.now().isoformat()

        }



    def save_metrics(
        self,
        metrics
    ):


        file = os.path.join(
            self.path,
            "metrics.json"
        )


        with open(
            file,
            "w"
        ) as f:

            json.dump(
                metrics,
                f,
                indent=4
            )



    def save_history(
        self,
        history
    ):


        file = os.path.join(
            self.path,
            "history.json"
        )


        with open(
            file,
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


        file = os.path.join(
            self.path,
            "model.pt"
        )


        torch.save(

            model.state_dict(),

            file

        )



    def save_config(
        self,
        config
    ):


        file = os.path.join(
            self.path,
            "config.json"
        )


        with open(
            file,
            "w"
        ) as f:


            json.dump(
                config,
                f,
                indent=4
            )
from dataclasses import dataclass
from pathlib import Path
import json
import torch.nn as nn


@dataclass
class Experiment:

    # ===========================
    # Identificação
    # ===========================

    model: str
    task: str
    dataset: str


    # ===========================
    # Dados
    # ===========================

    input_size: int = 12
    output_size: int = 12
    sequence_length: int = 30


    # ===========================
    # Treinamento
    # ===========================

    batch_size: int = 64
    epochs: int = 30
    lr: float = 0.001
    patience: int = 5


    # ===========================
    # Modelo
    # ===========================

    hidden_size: int = 100
    dropout: float = 0.2

    cnn_channels: int = 32
    kernel_size: int = 3

    skip: int = 5

    d_model: int = 64
    nhead: int = 4
    num_layers: int = 1

    hidden_channels: int = 64
    


    # ===========================
    # Diretórios
    # ===========================

    experiment_root: str = "experiments"


    # ===========================
    # Nome
    # ===========================

    @property
    def name(self):

        return (
            f"{self.model}_"
            f"{self.task}_"
            f"{self.dataset}"
        )


    @property
    def output_dir(self):

        path = (
            Path(self.experiment_root)
            /
            self.name
        )

        path.mkdir(
            parents=True,
            exist_ok=True
        )

        return path



    @property
    def checkpoint(self):

        return (
            self.output_dir
            /
            "best_model.pt"
        )



    @property
    def history_file(self):

        return (
            self.output_dir
            /
            "history.csv"
        )



    @property
    def metrics_file(self):

        return (
            self.output_dir
            /
            "metrics.json"
        )



    @property
    def predictions_file(self):

        return (
            self.output_dir
            /
            "predictions.csv"
        )



    @property
    def config_file(self):

        return (
            self.output_dir
            /
            "config.json"
        )



    # ===========================
    # Loss
    # ===========================

    @property
    def criterion(self):

        if self.task == "forecast":

            return nn.HuberLoss()

        elif self.task == "classification":

            return nn.BCEWithLogitsLoss()

        else:

            raise ValueError(
                f"Unknown task {self.task}"
            )



    # ===========================
    # Modelo
    # ===========================

    @property
    def model_params(self):

        params = {

            "input_size":
                self.input_size,

            "output_size":
                self.output_size

        }


        # Classificação sempre gera 1 saída

        if self.task == "classification":

            params["output_size"] = 1



        if self.model in [

            "lstm",
            "gru",
            "bilstm"

        ]:

            params.update({

                "hidden_size":
                    self.hidden_size

            })


        elif self.model in [

            "cnn_lstm",
            "lstnet",
            "attention_lstnet"

        ]:

            params.update({

                "hidden_size":
                    self.hidden_size,

                "cnn_channels":
                    self.cnn_channels,

                "kernel_size":
                    self.kernel_size

            })


        elif self.model == "transformer":

            params.update({

                "d_model":
                    self.d_model,

                "nhead":
                    self.nhead,

                "num_layers":
                    self.num_layers,

                "dropout":
                    self.dropout

            })


        elif self.model == "tcn":

            params.update({

                "hidden_channels":
                    self.hidden_channels

            })


        return params



    @property
    def optimizer_params(self):

        return {

            "lr": self.lr

        }



    # ===========================
    # Salvar configuração
    # ===========================

    def save(self):

        config = {

            "model": self.model,

            "task": self.task,

            "dataset": self.dataset,

            "input_size":
                self.input_size,

            "output_size":
                self.output_size,

            "sequence_length":
                self.sequence_length,

            "batch_size":
                self.batch_size,

            "epochs":
                self.epochs,

            "lr":
                self.lr,

            "hidden_size":
                self.hidden_size

        }


        with open(

            self.config_file,

            "w",

            encoding="utf8"

        ) as f:

            json.dump(

                config,

                f,

                indent=4

            )
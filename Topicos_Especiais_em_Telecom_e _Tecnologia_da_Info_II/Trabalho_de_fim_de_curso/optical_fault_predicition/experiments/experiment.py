from dataclasses import dataclass, field
from pathlib import Path
import json
import torch.nn as nn


@dataclass
class Experiment:

    # ===========================
    # Identificação
    # ===========================

    model: str
    task: str                  # forecast | classification
    dataset: str               # hard_failure | soft_failure

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

    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2

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

        return f"{self.model}_{self.task}_{self.dataset}"

    # ===========================
    # Pasta
    # ===========================

    @property
    def output_dir(self):

        path = Path(self.experiment_root) / self.name

        path.mkdir(
            parents=True,
            exist_ok=True
        )

        return path

    # ===========================
    # Checkpoint
    # ===========================

    @property
    def checkpoint(self):

        return self.output_dir / "best_model.pt"

    # ===========================
    # Histórico
    # ===========================

    @property
    def history_file(self):

        return self.output_dir / "history.csv"

    # ===========================
    # Métricas
    # ===========================

    @property
    def metrics_file(self):

        return self.output_dir / "metrics.json"

    # ===========================
    # Predições
    # ===========================

    @property
    def predictions_file(self):

        return self.output_dir / "predictions.csv"

    # ===========================
    # Configuração
    # ===========================

    @property
    def config_file(self):

        return self.output_dir / "config.json"

    # ===========================
    # Loss
    # ===========================

    @property
    def criterion(self):

        if self.task == "forecast":

            return nn.HuberLoss()

        return nn.BCEWithLogitsLoss()

    # ===========================
    # Hiperparâmetros
    # ===========================

    @property
    def model_params(self):

        params = {

            "input_size": self.input_size,
            "output_size": self.output_size

        }

        if self.model in [

            "lstm",
            "gru",
            "bilstm"

        ]:

            params["hidden_size"] = self.hidden_size

        elif self.model in [

            "cnn_lstm",
            "lstnet",
            "attention_lstnet"

        ]:

            params.update({

                "hidden_size": self.hidden_size,
                "cnn_channels": self.cnn_channels,
                "kernel_size": self.kernel_size

            })

        elif self.model == "transformer":

            params.update({

                "d_model": self.d_model,
                "nhead": self.nhead,
                "num_layers": self.num_layers,
                "dropout": self.dropout

            })

        elif self.model == "tcn":

            params.update({

                "hidden_channels": self.hidden_channels

            })

        return params

    # ===========================
    # Optimizer
    # ===========================

    @property
    def optimizer_params(self):

        return {

            "lr": self.lr

        }

    # ===========================
    # Salvar Config
    # ===========================

    def save(self):

        config = self.__dict__.copy()

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
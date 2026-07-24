from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import json
import time


class History:

    def __init__(self, experiment):

        self.experiment = experiment

        self.records = []

        self.start_time = None

    # -------------------------------------------------

    def start_epoch(self):

        self.start_time = time.time()

    # -------------------------------------------------

    def end_epoch(

        self,

        epoch,

        train_loss,

        val_loss,

        lr

    ):

        elapsed = time.time() - self.start_time

        self.records.append({

            "epoch": epoch,

            "train_loss": train_loss,

            "val_loss": val_loss,

            "lr": lr,

            "time": elapsed

        })

    # -------------------------------------------------

    @property
    def dataframe(self):

        return pd.DataFrame(

            self.records

        )

    # -------------------------------------------------

    def save_csv(self):

        self.dataframe.to_csv(

            self.experiment.history_file,

            index=False

        )

    # -------------------------------------------------

    def save_json(self):

        with open(

            self.experiment.output_dir/"history.json",

            "w",

            encoding="utf8"

        ) as f:

            json.dump(

                self.records,

                f,

                indent=4

            )

    # -------------------------------------------------

    def plot_losses(self):

        df = self.dataframe

        plt.figure(figsize=(8,5))

        plt.plot(

            df["epoch"],

            df["train_loss"],

            label="Train"

        )

        plt.plot(

            df["epoch"],

            df["val_loss"],

            label="Validation"

        )

        plt.xlabel("Epoch")

        plt.ylabel("Loss")

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(

            self.experiment.output_dir/"loss.png"

        )

        plt.close()

    # -------------------------------------------------

    def plot_lr(self):

        df = self.dataframe

        plt.figure(figsize=(8,5))

        plt.plot(

            df["epoch"],

            df["lr"]

        )

        plt.xlabel("Epoch")

        plt.ylabel("Learning Rate")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(

            self.experiment.output_dir/"lr.png"

        )

        plt.close()

    # -------------------------------------------------

    def plot_time(self):

        df = self.dataframe

        plt.figure(figsize=(8,5))

        plt.bar(

            df["epoch"],

            df["time"]

        )

        plt.xlabel("Epoch")

        plt.ylabel("Seconds")

        plt.tight_layout()

        plt.savefig(

            self.experiment.output_dir/"time.png"

        )

        plt.close()

    # -------------------------------------------------

    def save_all(self):

        self.save_csv()

        self.save_json()

        self.plot_losses()

        self.plot_lr()

        self.plot_time()
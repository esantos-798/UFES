"""
Preprocessing pipeline for optical fault prediction datasets.

Transforma o dataset bruto:
Timestamp | Type | ID | BER | OSNR | InputPower | OutputPower | Failure

em uma série temporal multivariada:

Timestamp |
SPO1_BER |
SPO1_OSNR |
SPO2_BER |
SPO2_OSNR |
Ampli1_InputPower |
Ampli1_OutputPower |
...

Autor: Eduardo Ribeiro
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from config import HARD_FAILURE_DATASET
from sklearn.preprocessing import StandardScaler

# ============================
# Configurações iniciais
# ============================

WINDOW_SIZE = 30
HORIZON = 1


FEATURES = [
    "SPO1_BER",
    "SPO1_OSNR",
    "SPO2_BER",
    "SPO2_OSNR",

    "Ampli1_InputPower",
    "Ampli1_OutputPower",

    "Ampli2_InputPower",
    "Ampli2_OutputPower",

    "Ampli3_InputPower",
    "Ampli3_OutputPower",

    "Ampli4_InputPower",
    "Ampli4_OutputPower",
]


class OpticalFaultPreprocessor:

    def __init__(
        self,
        csv_file: str,
        window_size: int = WINDOW_SIZE,
        horizon: int = HORIZON,
    ):

        self.csv_file = Path(csv_file)

        self.window_size = window_size
        self.horizon = horizon

        self.scaler = StandardScaler()


    # ---------------------------------

    def load_data(self):

        print("Loading dataset...")

        df = pd.read_csv(self.csv_file)

        print("Raw shape:", df.shape)

        return df


    # ---------------------------------

    def pivot_network_state(self, df):

        """
        Converte medições por equipamento
        em um vetor de estado da rede.
        """

        print("\nPivoting network state...")


        rows = []


        timestamps = sorted(
            df["Timestamp"].unique()
        )


        for ts in timestamps:

            sample = {
                "Timestamp": ts,
                "Failure": 0
            }


            current = df[
                df["Timestamp"] == ts
            ]


            for _, row in current.iterrows():

                device = row["ID"]

                if row["Failure"] == 1:
                    sample["Failure"] = 1

                if "SPO1" in device:

                    sample["SPO1_BER"] = row["BER"]

                    sample["SPO1_OSNR"] = row["OSNR"]


                elif "SPO2" in device:

                    sample["SPO2_BER"] = row["BER"]

                    sample["SPO2_OSNR"] = row["OSNR"]


                elif "Ampli" in device:

                    sample[
                        f"{device}_InputPower"
                    ] = row["InputPower"]


                    sample[
                        f"{device}_OutputPower"
                    ] = row["OutputPower"]


            rows.append(sample)


        network_df = pd.DataFrame(rows)


        print(
            "After pivot:",
            network_df.shape
        )


        return network_df


    # ---------------------------------

    def clean_data(self, df):

        print("\nCleaning data...")


        df = df.sort_values(
            "Timestamp"
        )


        # manter apenas estados completos

        df = df.dropna()


        print(
            "After cleaning:",
            df.shape
        )


        return df


    # ---------------------------------

    def normalize(self, df):

        print("\nNormalizing...")


        values = df[FEATURES].values


        values = self.scaler.fit_transform(
            values
        )


        return values


    # ---------------------------------

    def create_sequences(
        self,
        X,
        labels
    ):

        print("\nCreating sequences...")

        sequences = []
        targets = []
        failure_targets = []


        for i in range(
            len(X)
            - self.window_size
            - self.horizon
        ):

            seq = X[
                i:i+self.window_size
            ]

            target = X[
                i+self.window_size
            ]


            failure = labels[
                i+self.window_size
            ]


            sequences.append(seq)

            targets.append(target)

            failure_targets.append(failure)

        print("Number sequences:", len(sequences))
        print("Number targets:", len(targets))
        print("Number failures:", len(failure_targets))
        
        return (
            np.array(sequences),
            np.array(targets),
            np.array(failure_targets)
        )

    # ---------------------------------

    def run(self):


        df = self.load_data()


        df = self.pivot_network_state(
            df
        )


        df = self.clean_data(
            df
        )


        labels = df["Failure"].values

        X = self.normalize(
            df
        )

        X_seq, y, failure = self.create_sequences(
            X,
            labels
        )


        print("\nFinal dataset")
        print("failure:", failure.shape)

        print(
            "X:",
            X_seq.shape
        )

        print(
            "y:",
            y.shape
        )


        return X_seq, y, failure



if __name__ == "__main__":


    processor = OpticalFaultPreprocessor(
        HARD_FAILURE_DATASET
    )

    X, y = processor.run()
from pathlib import Path
from datetime import datetime
import traceback

import pandas as pd
import torch

from src.runners.run_experiment import ExperimentRunner

import sys
import importlib

# Remove o Trainer antigo do cache de módulos do Python antes de qualquer execução
if 'src.training.trainer' in sys.modules:
    importlib.reload(sys.modules['src.training.trainer'])
    
from src.training.trainer import Trainer


class ExperimentRunnerGrid:

    def __init__(self, experiments):

        self.experiments = experiments

        self.results = []

        self.best_f1 = -1.0

        self.best_model = None

        Path("results/csv").mkdir(parents=True, exist_ok=True)
        Path("results/models").mkdir(parents=True, exist_ok=True)
        Path("results/logs").mkdir(parents=True, exist_ok=True)

    #########################################################

    def run(self):

        total = len(self.experiments)

        for i, exp in enumerate(self.experiments):

            print()
            print("=" * 80)
            print(f"Experiment {i+1}/{total}")
            print(exp)
            print("=" * 80)

            start = datetime.now()

            try:

                #################################################
                # Executa experimento
                #################################################

                #runner = ExperimentRunner(

                #    model_name=exp.model,

                #    dataset_name=exp.dataset,

                #    forecast_weight=exp.forecast_weight,

                #    failure_weight=exp.failure_weight,

                #    pos_weight=exp.pos_weight,

                #    alpha=exp.alpha,

                #    random_seed=exp.random_seed

                #)
                # Executa experimento
                runner = ExperimentRunner(exp)
                output = runner.run()  # Contém o dicionário empacotado

                # Extrai as métricas reais para passar para o resto do script
                metrics = output["metrics"] 

                elapsed = (datetime.now() - start).total_seconds()

                #################################################
                # Salva resultados
                #################################################

                result = {

                    **exp.__dict__,

                    **metrics,

                    "Status": "SUCCESS",

                    "ExecutionTime": elapsed

                }

                self.results.append(result)


                #################################################
                # Melhor modelo (CORRIGIDO)
                #################################################

                # 1. Captura o valor de F1 de forma segura
                f1_value = metrics.get('F1', metrics.get('f1', metrics.get('Failure_F1', 0.0)))

                # 2. Atualiza e exibe APENAS se for estritamente maior que o melhor F1 anterior
                if f1_value > self.best_f1:
                    self.best_f1 = f1_value  # Atualiza primeiro o estado global

                    filename = (
                        f"results/models/"
                        f"{exp.model}"
                        f"_fw{exp.forecast_weight}"
                        f"_fl{exp.failure_weight}"
                        f"_pw{exp.pos_weight}"
                        f"_a{exp.alpha:.1f}"
                        f"_F1_{f1_value:.4f}.pt"
                    )

                    checkpoint = {
                        "model_state_dict": runner.model.state_dict(),
                        "config": exp.__dict__,
                        "metrics": metrics
                    }

                    torch.save(checkpoint, filename)

                    # Print executado estritamente após a atualização da variável global
                    print()
                    print("=" * 40)
                    print("🔥 Novo melhor modelo encontrado e salvo!")
                    print(f"Dataset avaliado: {exp.dataset}")
                    print(f"F1-Score Atualizado: {self.best_f1:.4f}")
                    print("=" * 40)


            #################################################
            # Se der erro continua para o próximo experimento
            #################################################

            except Exception as e:

                elapsed = (datetime.now() - start).total_seconds()

                print(traceback.format_exc())

                self.results.append({

                    **exp.__dict__,

                    "Status": "FAILED",

                    "ExecutionTime": elapsed,

                    "Error": str(e)

                })

            #################################################
            # Salva CSV após cada experimento
            #################################################

            self.save_results()

        print()
        print("=" * 80)
        print("Todos os experimentos finalizados.")
        print("=" * 80)

    #########################################################

    def save_results(self):

        df = pd.DataFrame(self.results)

        if "F1" in df.columns:

            df = df.sort_values(

                by="F1",

                ascending=False,

                na_position="last"

            )

        df.to_csv(

            "results/csv/grid_results.csv",

            index=False

        )

        print(
            f"Resultados atualizados ({len(df)} experimentos)."
        )
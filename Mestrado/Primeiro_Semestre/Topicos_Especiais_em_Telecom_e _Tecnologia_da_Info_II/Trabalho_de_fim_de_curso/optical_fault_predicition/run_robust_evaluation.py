import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from copy import deepcopy

# Importações oficiais do seu projeto
from src.config import DATASETS  # Carrega os caminhos configurados
from experiments.experiment import Experiment
from src.runners.run_experiment import ExperimentRunner

def set_seed(seed):
    """Garante a reprodutibilidade fixando todas as sementes do pipeline."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_tuning_grid():
    # ------------------------------------------------------------------
    # 1. Definição do Experimento Base
    # ------------------------------------------------------------------
    # Criamos o experimento exatamente como você faz no test_run.py
    base_exp = Experiment(
        model="multitask_lstnet",
        task="forecast",          # O runner usa essa tarefa para carregar os loaders
        dataset="hard_failure",   # Usando o SoftFailure
        epochs=30,                # Reduzido temporariamente para um teste rápido (depois suba para 30 ou 50)
        batch_size=64,
        lr=0.001
    )
    
    # ------------------------------------------------------------------
    # 2. Definição do Grid de Hiperparâmetros de Loss e Sementes
    # ------------------------------------------------------------------
    # Defina aqui as sementes e pesos que deseja testar
    seeds = [42, 123, 999]                 # Comece com 3 sementes para testar a velocidade
    failure_weights = [0.1, 0.5, 1.0, 2.0]  # Pesos da perda multitask
    
    all_runs_results = []
    summary_results = []

    print("=" * 80)
    print("INICIANDO GRID SEARCH MULTITASK LSTNet (SEEDS + LOSS WEIGHTS)".center(80))
    print("=" * 80)

    for weight in failure_weights:
        print(f"\n>>>> [TESTANDO LAMBDA_2 = {weight}] <<<<\n")
        weight_metrics = []
        
        for seed in seeds:
            print(f"--- Rodando Seed {seed} ---")
            
            # Força o determinismo da rodada atual
            set_seed(seed)
            
            # Copia o experimento para isolar as configurações
            exp_run = deepcopy(base_exp)
            
            # Injeção dinâmica para contornar a propriedade 'name' sem setter
            class MutableExperiment(type(exp_run)):
                @property
                def name(self):
                    return self._custom_name
            
            exp_run.__class__ = MutableExperiment
            exp_run._custom_name = f"multitask_lstnet_weight_{weight}_seed_{seed}"
            
            # Configura os parâmetros alterados nesta rodada
            exp_run.seed = seed
            exp_run.failure_weight = weight
            
            # MultiTaskLSTNet não tem Sigmoid na cabeça de classificação
            # Usar BCEWithLogitsLoss é a opção matematicamente estável
            if not hasattr(exp_run, "failure_loss") or exp_run.failure_loss is None:
                exp_run.failure_loss = nn.BCEWithLogitsLoss()
            
            # Inicializa e executa usando o runner nativo do seu projeto
            runner = ExperimentRunner(exp_run)
            
            try:
                # Executa o treino, teste e avaliações completas
                metrics = runner.run()
                
                # Se as métricas retornaram com sucesso, registramos
                if metrics:
                    metrics_copy = deepcopy(metrics)
                    metrics_copy['seed'] = seed
                    metrics_copy['failure_weight'] = weight
                    
                    weight_metrics.append(metrics_copy)
                    all_runs_results.append(metrics_copy)
                
            except Exception as e:
                print(f"[ERRO] Falha crítica na execução (Seed: {seed}, Weight: {weight}): {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Consolidação estatística para o peso (lambda_2) atual
        if len(weight_metrics) > 0:
            df_weight = pd.DataFrame(weight_metrics)
            summary_row = {"failure_weight": weight}
            
            print(f"\n=== RESULTADOS CONSOLIDADOS (Weight = {weight}) ===")
            for col in df_weight.columns:
                if col not in ['seed', 'failure_weight'] and np.issubdtype(df_weight[col].dtype, np.number):
                    mean_val = df_weight[col].mean()
                    std_val = df_weight[col].std()
                    
                    summary_row[f"{col}_mean"] = mean_val
                    summary_row[f"{col}_std"] = std_val
                    
                    # Exibe no console as métricas cruciais da sua análise
                    if any(metric_name in col for metric_name in ['MSE', 'R2', 'F1', 'AUC', 'Recall', 'Precision']):
                        print(f"  * {col:25}: {mean_val:.4f} ± {std_val:.4f}")
            
            summary_results.append(summary_row)
            print("-" * 50)

    # ------------------------------------------------------------------
    # 3. Salva os arquivos de resultados consolidados
    # ------------------------------------------------------------------
    if len(all_runs_results) > 0:
        df_raw = pd.DataFrame(all_runs_results)
        df_summary = pd.DataFrame(summary_results)
        
        df_raw.to_csv("robust_raw_runs_results.csv", index=False)
        df_summary.to_csv("robust_summary_loss_tuning.csv", index=False)
        
        print("\n" + "="*80)
        print("PROCESSO CONCLUÍDO COM SUCESSO!".center(80))
        print("Arquivos CSV gerados na raiz do projeto:")
        print("  1. 'robust_raw_runs_results.csv' (Todas as rodadas individuais)")
        print("  2. 'robust_summary_loss_tuning.csv' (Média ± Desvio padrão por Peso)")
        print("="*80)
    else:
        print("\n[AVISO] Nenhuma rodada retornou resultados válidos para salvar.")

# DISPARO DO SCRIPT (Esta parte garante que o script realmente execute ao rodar)
if __name__ == "__main__":
    run_tuning_grid()
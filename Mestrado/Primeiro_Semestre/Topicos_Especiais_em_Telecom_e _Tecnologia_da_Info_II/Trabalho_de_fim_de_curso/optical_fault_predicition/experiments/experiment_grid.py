from itertools import product
from .experiment_config import ExperimentConfig

MODELS = ["lstm", "bilstm", "gru", "transformer", "tcn", "lstnet","attention_lstnet", "multitask_lstnet", "multitask_lstnet_attention", "multitask_lstnet_transformer", "multitask_lstnet_tcn"]
#MODELS = ["lstm", "tcn", "lstnet", "multitask_lstnet", "multitask_lstnet_transformer", "multitask_lstnet_tcn", "lstm_xgboost", "transformer_xgboost"]
#MODELS = ["lstm_xgboost", "transformer_xgboost", "lstnet_xgboost"]
#MODELS = ["lstnet", "multitask_lstnet"]
DATASETS = ["hard_failure", "soft_failure"]
FORECAST_WEIGHTS = [1]
FAILURE_WEIGHTS = [1, 2, 3, 5, 7, 10, 15]
#FAILURE_WEIGHTS = [2, 10]
#FAILURE_WEIGHTS = [10]
POS_WEIGHTS = [5]
ALPHAS = [0.5]

def generate_grid():
    experiments = []

    for values in product(
        MODELS,
        DATASETS,
        FORECAST_WEIGHTS,
        FAILURE_WEIGHTS,
        POS_WEIGHTS,
        ALPHAS
    ):
        experiments.append(
            ExperimentConfig(
                model=values[0],
                dataset=values[1],
                forecast_weight=values[2],
                failure_weight=values[3],
                pos_weight=values[4],
                alpha=values[5]
            )
        )

    return experiments
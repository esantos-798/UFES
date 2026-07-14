from experiments.experiment import Experiment
from src.runners.run_experiment import ExperimentRunner

import random
import numpy as np
import torch


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#exp = Experiment(

#    model="lstm",

#    task="classification",

#    dataset="hard"

#)
#models = ['bilstm',  'ccn_lstm',  'tcn',   'lstnet_v2',     'transformer']
#for mod in models:
#    exp = Experiment(
#        model=mod,
#        task="forecast",
#        dataset="hard_failure",
#        epochs=30,
#        batch_size=64,
#        lr=0.001
#    )
#exp = Experiment(
#    model="bilstm",
#    task="forecast",
#    dataset="soft_failure",
#    epochs=30,
#    batch_size=64,
#    lr=0.001
#)

exp = Experiment(
    model="attention_lstnet",
    task="forecast",
    dataset="hard_failure",
    epochs=30,
    batch_size=64,
    lr=0.001
)

#exp = Experiment(
#    model="lstnet_v2",
#    task="forecast",
#    dataset="hard_failure",
#    epochs=30,
#    batch_size=64,
#    lr=0.001
#)

#exp = Experiment(
#    model="attention_lstnet",
#    task="forecast",
#    dataset="hard_failure",
#    epochs=30,
#    batch_size=64,
#    lr=0.001
#)

#exp = Experiment(
#    model="mtl_lstnet",
#    task="forecast",
#    dataset="hard_failure",
#    epochs=30,
#    batch_size=64,
#    lr=0.001
#)

runner = ExperimentRunner(exp)

runner.run()
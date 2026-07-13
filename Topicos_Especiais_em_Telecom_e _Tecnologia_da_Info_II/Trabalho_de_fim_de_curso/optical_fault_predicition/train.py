import argparse
import importlib

import torch
import torch.nn as nn
import torch.optim as optim

from src.data.dataloader import get_dataloader
from src.models.factory import get_model
from src.training.runner import ExperimentRunner


import argparse

from configs.factory import get_config
from src.models.factory import get_model

from src.experiments.experiment import Experiment

experiment = Experiment(

    dataset="hard",

    task="classification",

    model="transformer"

)

train_loader, val_loader, test_loader = get_dataloader(
    task=args.task,
    dataset=args.dataset
)

model = build_model(
    args.model,
    task=args.task
).to(device)

criterion = build_loss(args.task)

optimizer = build_optimizer(model)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    experiment=experiment,
    task=args.task
)

trainer.fit(epochs=30)

metrics = Evaluator(
    model=model,
    test_loader=test_loader,
    device=device,
    experiment=experiment,
    task=args.task
).evaluate()

print(metrics)
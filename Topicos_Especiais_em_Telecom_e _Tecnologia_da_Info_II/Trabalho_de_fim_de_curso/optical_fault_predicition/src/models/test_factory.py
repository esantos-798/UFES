from configs.models import *
from src.models.model_factory import ModelFactory
from experiments.experiment import Experiment


exp = Experiment(
    model="lstm",
    task="classification",
    dataset="hard_failure"
)


model = ModelFactory.create(exp)


print(model)
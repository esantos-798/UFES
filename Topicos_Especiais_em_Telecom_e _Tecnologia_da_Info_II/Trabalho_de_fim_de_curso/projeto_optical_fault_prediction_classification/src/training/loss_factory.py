import torch.nn as nn


def build_loss(task):

    if task == "classification":

        return nn.BCEWithLogitsLoss()

    return nn.HuberLoss()
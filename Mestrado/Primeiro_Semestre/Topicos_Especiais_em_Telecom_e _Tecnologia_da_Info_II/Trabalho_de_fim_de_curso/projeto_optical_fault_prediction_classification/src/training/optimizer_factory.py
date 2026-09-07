import torch.optim as optim


def build_optimizer(
    model,
    lr=1e-3,
    optimizer_name="adam"
):

    if optimizer_name == "adam":

        return optim.Adam(
            model.parameters(),
            lr=lr
        )

    if optimizer_name == "adamw":

        return optim.AdamW(
            model.parameters(),
            lr=lr
        )

    if optimizer_name == "sgd":

        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9
        )

    raise ValueError(
        optimizer_name
    )
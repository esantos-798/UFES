import torch

def load_encoder_weights(model, checkpoint_path, ignore_layers=("fc",)):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    model.load_state_dict(
        {
            k: v
            for k, v in checkpoint.items()
            if not any(k.startswith(layer) for layer in ignore_layers)
        },
        strict=False
    )

    return model
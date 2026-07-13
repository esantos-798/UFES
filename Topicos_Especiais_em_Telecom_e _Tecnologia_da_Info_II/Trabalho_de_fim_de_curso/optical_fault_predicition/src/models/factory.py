from src.models.lstm import LSTM
from src.models.gru import GRU
from src.models.bilstm import BiLSTM
from src.models.cnn_lstm import CNNLSTM
from src.models.lstnet import LSTNet
from src.models.attention_lstnet import AttentionLSTNet
from src.models.transformer import TransformerClassifier
from src.models.tcn import TCN


def get_model(model_name, config):

    model_name = model_name.lower()

    if model_name == "lstm":
        return LSTM(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            output_size=config["output_size"]
        )

    elif model_name == "gru":
        return GRU(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            output_size=config["output_size"]
        )

    elif model_name == "bilstm":
        return BiLSTM(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            output_size=config["output_size"]
        )

    elif model_name == "cnn_lstm":
        return CNNLSTM(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            output_size=config["output_size"]
        )

    elif model_name == "lstnet":
        return LSTNet(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            output_size=config["output_size"]
        )

    elif model_name == "attention_lstnet":
        return AttentionLSTNet(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            output_size=config["output_size"]
        )

    elif model_name == "transformer":
        return TransformerClassifier(
            input_size=config["input_size"],
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
            output_size=config["output_size"]
        )

    elif model_name == "tcn":
        return TCN(
            input_size=config["input_size"],
            num_channels=config["num_channels"],
            output_size=config["output_size"]
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")
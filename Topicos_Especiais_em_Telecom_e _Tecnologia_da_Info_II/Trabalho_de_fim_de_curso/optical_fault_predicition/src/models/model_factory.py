from src.models.lstm import LSTM
from src.models.gru import GRU
from src.models.bilstm import BiLSTM

from src.models.cnn_lstm import CNNLSTM

from src.models.lstnet import LSTNet
from src.models.attention_lstnet import AttentionLSTNet

from src.models.transformer import Transformer

from src.models.tcn import TCN



# src/models/model_factory.py

import inspect


class ModelFactory:


    @staticmethod
    def create(exp):


        name = exp.model.lower()


        if name == "lstm":

            if exp.task == "classification":

                from src.models.lstm_classifier import LSTMClassifier

                model_class = LSTMClassifier

            else:

                from src.models.lstm import LSTM

                model_class = LSTM


        elif name == "gru":

            from src.models.gru import GRU

            model_class = GRU


        elif name == "bilstm":

            from src.models.bilstm import BiLSTM

            model_class = BiLSTM


        elif name == "cnn_lstm":

            from src.models.cnn_lstm import CNNLSTM

            model_class = CNNLSTM


        elif name == "lstnet":

            from src.models.lstnet import LSTNet

            model_class = LSTNet


        elif name == "attention_lstnet":

            from src.models.attention_lstnet import AttentionLSTNet

            model_class = AttentionLSTNet


        elif name == "transformer":

            from src.models.transformer import Transformer

            model_class = Transformer


        elif name == "tcn":

            from src.models.tcn import TCN

            model_class = TCN

        elif name == "lstnet_v2":

            from src.models.lstnet_v2 import LSTNetV2

            model_class = LSTNetV2

        elif name == "multitask_lstnet":

            from src.models.multitask_lstnet import MultiTaskLSTNet

            model_class = MultiTaskLSTNet   

        elif name == "mtl_lstnet":

            from src.models.mtl_lstnet import MTLLSTNet

            model_class = MTLLSTNet     

        else:

            raise ValueError(
                f"Unknown model {name}"
            )


        params = {

            "input_size": exp.input_size,

            "hidden_size": exp.hidden_size,

            "output_size": exp.output_size,

            "num_layers": exp.num_layers,

            "dropout": exp.dropout,

            "d_model": exp.d_model,

            "nhead": exp.nhead,

            "cnn_channels": exp.cnn_channels,

            "kernel_size": exp.kernel_size
        }

        # remove argumentos não suportados

        valid_params = inspect.signature(
            model_class.__init__
        ).parameters


        params = {

            k:v

            for k,v in params.items()

            if k in valid_params

        }


        return model_class(
            **params
        )
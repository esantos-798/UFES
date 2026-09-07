from config import (
    lstm,
    gru,
    bilstm,
    cnn_lstm,
    lstnet,
    attention_lstnet,
    transformer,
    tcn
)

CONFIGS = {

    "lstm": lstm.CONFIG,
    "gru": gru.CONFIG,
    "bilstm": bilstm.CONFIG,
    "cnn_lstm": cnn_lstm.CONFIG,
    "lstnet": lstnet.CONFIG,
    "attention_lstnet": attention_lstnet.CONFIG,
    "transformer": transformer.CONFIG,
    "tcn": tcn.CONFIG

}


def get_config(model):

    return CONFIGS[model.lower()]
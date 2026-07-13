from src.experiments.experiment import Experiment
from src.models.model_factory import ModelFactory



models = [

    "lstm",

    "gru",

    "bilstm",

    "cnn_lstm",

    "lstnet",

    "attention_lstnet",

    "transformer",

    "tcn"

]


for name in models:


    print(
        "\nTesting:",
        name
    )


    exp = Experiment(

        model=name

    )


    model = ModelFactory.create(

        exp

    )


    print(

        type(model).__name__

    )
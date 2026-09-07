import inspect
from src.models.hybrid_wrappers import HybridXGBoostWrapper

class ModelFactory:

    @staticmethod
    def create(exp):
        name = exp.model.lower()
        
        # 1. Detecta se é um experimento híbrido independente do Grid
        is_xgboost_hybrid = False
        if name.endswith("_xgboost"):
            is_xgboost_hybrid = True
            # Força o nome a virar o modelo base (ex: "lstm_xgboost" vira "lstm" para os ifs abaixo)
            name = name.replace("_xgboost", "") 

        # 2. Mapeamento das classes de Redes Neurais do seu ecossistema
        if name == "lstm":
            if getattr(exp, "task", "forecast") == "classification":
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
        elif name == "multitask_lstnet_attention":
            from src.models.multitask_lstnet_attention import MultiTaskLSTNetAttention
            model_class = MultiTaskLSTNetAttention
        elif name == "multitask_lstnet_transformer":
            from src.models.multitask_lstnet_transformer import MultiTaskLSTNetTransformer
            model_class = MultiTaskLSTNetTransformer   
        elif name == "multitask_lstnet_tcn":
            from src.models.multitask_lstnet_tcn import MultiTaskLSTNetTCN
            model_class = MultiTaskLSTNetTCN
        else:
            raise ValueError(f"Unknown model {name}")

        # 3. Mapeia dinamicamente todos os hiperparâmetros de arquitetura do ExperimentConfig
        params = {
            "input_size": exp.input_size,
            "hidden_size": exp.hidden_size,
            "output_size": exp.output_size,
            "num_layers": exp.num_layers,
            "dropout": exp.dropout,
            "d_model": exp.d_model,
            "nhead": exp.nhead,
            "cnn_channels": exp.cnn_channels,
            "kernel_size": exp.kernel_size,
            "experiment": exp  # Para modelos MultiTask que pedem o objeto de configuração
        }

        # 4. Filtra apenas propriedades aceitas pela assinatura de inicialização do __init__ alvo
        valid_params = inspect.signature(model_class.__init__).parameters
        params = {k: v for k, v in params.items() if k in valid_params}

        # 5. Instancia a rede neural base
        base_model_instance = model_class(**params)

        # 6. Se for um modelo híbrido, encapsula no Wrapper; senão, retorna a rede pura
        if is_xgboost_hybrid:
            return HybridXGBoostWrapper(base_model_instance, model_name=exp.model)
            
        return base_model_instance
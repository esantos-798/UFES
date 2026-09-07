# train_soft_failure_models.py
import torch
import torch.nn as nn
from src.data_processing import load_and_engineer_features, prepare_datasets
from src.models import (
    VanillaRNNBaseline, LSTMBaseline, GRUBaseline, BiLSTMBaseline,
    LSTNetModified, AttentionLSTNet, TransformerBaseline, SpatioTemporalDirectedGNN
)
from src.evaluation import calculate_thresholds

WINDOW_SIZE = 15
BATCH_SIZE = 64
EPOCHS = 15  # Reduzido para acelerar o ciclo de testes
DATA_PATH = "data/SoftFailure_dataset.csv"  # <-- Apontando para o Soft Failure
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("1. Preparando dados do cenário Soft Failure...")
df_pivoted = load_and_engineer_features(DATA_PATH)
train_loader, _, _, _, feature_names = prepare_datasets(df_pivoted, WINDOW_SIZE, BATCH_SIZE)
num_features = len(feature_names)

# Lista com as arquiteturas candidatas que vamos treinar para o cenário soft
models_to_train = [
    ("Vanilla RNN", VanillaRNNBaseline, "rnn_soft.pt"),
    ("LSTM Baseline", LSTMBaseline, "lstm_soft.pt"),
    ("GRU Baseline", GRUBaseline, "gru_soft.pt"),
    ("BiLSTM Baseline", BiLSTMBaseline, "bilstm_soft.pt"),
    ("LSTNet", LSTNetModified, "lstnet_soft.pt"),
    ("Attention-LSTNet", AttentionLSTNet, "attention_lstnet_soft.pt"),
    ("Transformer", TransformerBaseline, "transformer_soft.pt"),
    ("ST-GNN Direcionada", SpatioTemporalDirectedGNN, "st_gnn_soft.pt")
]

criterion = nn.L1Loss()

for name, model_class, save_name in models_to_train:
    print(f"\nTreinando {name} no cenário Soft Failure...")
    
    # Inicialização adequada por tipo de construtor
    if name in ["LSTNet", "Attention-LSTNet"]:
        model = model_class(input_dim=num_features, window_size=WINDOW_SIZE, hidden_dim=64).to(DEVICE)
    elif name in ["ST-GNN Direcionada"]:
        model = model_class(num_nodes=num_features, in_features=1, hidden_dim=32, output_dim=num_features).to(DEVICE)
    else:
        model = model_class(input_dim=num_features, hidden_dim=64, output_dim=num_features).to(DEVICE)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
            
    train_loss /= len(train_loader.dataset)
    print(f"-> {name} finalizado com Perda MAE: {train_loss:.4f}")
    
    # Calcula limiares calibrados para o cenário soft
    thresholds = calculate_thresholds(model, train_loader, DEVICE)
    
    torch.save({
        'model_state': model.state_dict(),
        'thresholds': thresholds,
        'features': feature_names
    }, save_name)

print("\n[SUCESSO] Todos os modelos de Deep Learning re-treinados e salvos para o cenário Soft Failure!")
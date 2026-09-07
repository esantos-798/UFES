# train_all_baselines.py
import os
import torch
import torch.nn as nn
from src.data_processing import load_and_engineer_features, prepare_datasets
from src.models import GRUBaseline, BiLSTMBaseline
from src.evaluation import calculate_thresholds

# --- CONFIGURAÇÕES ---
WINDOW_SIZE = 15
BATCH_SIZE = 64
EPOCHS = 20
DATA_PATH = "data/HardFailure_dataset.csv"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("1. Carregando dados de Hard Failure para o Treinamento...")
df_pivoted = load_and_engineer_features(DATA_PATH)
train_loader, val_loader, _, _, feature_names = prepare_datasets(df_pivoted, WINDOW_SIZE, BATCH_SIZE)

num_features = len(feature_names)
print(f" -> Total de features detectadas: {num_features}")

# Lista de experimentos: (Nome do Modelo, Classe do Modelo, Nome do Arquivo de Saída)
experimentos = [
    ("GRU", GRUBaseline, "gru_baseline.pt"),
    ("BiLSTM", BiLSTMBaseline, "bilstm_baseline.pt")
]

# --- LOOP DE TREINAMENTO UNIFICADO ---
for nome, modelo_class, arquivo_saida in experimentos:
    print(f"\n=========================================")
    print(f" Inicializando Treinamento do Modelo: {nome}")
    print(f"=========================================")
    
    # Inicializa o modelo atual
    model = modelo_class(input_dim=num_features, hidden_dim=64, output_dim=num_features).to(DEVICE)
    criterion = nn.L1Loss() # MAE absoluto conforme o artigo original
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Loop de épocas
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
        print(f"Época [{epoch+1}/{EPOCHS}] - Perda MAE: {train_loss:.4f}")
        
    print(f"\n[CALIBRANDO] Calculando limiares do percentil 99% para {nome}...")
    thresholds = calculate_thresholds(model, train_loader, DEVICE)
    
    # Salvando pesos e metadados
    torch.save({
        'model_state': model.state_dict(),
        'thresholds': thresholds,
        'features': feature_names
    }, arquivo_saida)
    
    print(f"[SUCESSO] Modelo {nome} salvo em '{arquivo_saida}'!")

print("\n[PROCESSO CONCLUÍDO] Todos os novos baselines foram treinados!")
# train_multitask_monster.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from src.models import MultiTaskFaultDiagnosisGNN

WINDOW_SIZE = 15
BATCH_SIZE = 32
EPOCHS = 35  
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Carga e preparação do dataset
df = pd.read_csv("data/Multitask_dataset.csv")
features = [c for c in df.columns if c not in ['Timestamp', 'Label_Det', 'Label_Loc', 'Label_Sev']]
num_nodes = len(features)

X_data = df[features].values
y_det = df['Label_Det'].values
y_loc = df['Label_Loc'].values
y_sev = df['Label_Sev'].values

X_windows, seq_det, seq_loc, seq_sev = [], [], [], []
for i in range(len(df) - WINDOW_SIZE):
    X_windows.append(X_data[i:i+WINDOW_SIZE])
    seq_det.append(y_det[i+WINDOW_SIZE-1])
    seq_loc.append(y_loc[i+WINDOW_SIZE-1])
    seq_sev.append(y_sev[i+WINDOW_SIZE-1])

X_tensor = torch.tensor(np.array(X_windows), dtype=torch.float32)
y_det_tensor = torch.tensor(np.array(seq_det), dtype=torch.float32)
y_loc_tensor = torch.tensor(np.array(seq_loc), dtype=torch.long)
y_sev_tensor = torch.tensor(np.array(seq_sev), dtype=torch.long)

# Shuffled Split para garantir a distribuição uniforme das falhas
dataset = TensorDataset(X_tensor, y_det_tensor, y_loc_tensor, y_sev_tensor)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

torch.manual_seed(42)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Salvamos os índices de teste para o avaliador usar exatamente o mesmo vetor
torch.save(test_dataset, "data/test_dataset_split.pt")

# --- CORREÇÃO OPERACIONAL DOS PESOS DA LOCALIZAÇÃO (FORÇANDO AS 17 CLASSES) ---
NUM_MODEL_CLASSES_LOC = 17 # Alinha perfeitamente com a saída fixa da cabeça da rede
labels_loc_arr = np.array(seq_loc)
class_counts = np.bincount(labels_loc_arr)

# Criamos um vetor de contagem expandido para acomodar todas as classes da rede
full_counts = np.zeros(NUM_MODEL_CLASSES_LOC)
full_counts[:len(class_counts)] = class_counts

# Aplica o inverso da frequência apenas onde há dados, evitando divisão por zero
weights_loc = np.zeros(NUM_MODEL_CLASSES_LOC)
active_classes = full_counts > 0
weights_loc[active_classes] = 1.0 / (full_counts[active_classes])
weights_loc = weights_loc / np.sum(weights_loc) * np.sum(active_classes)
weights_loc_tensor = torch.tensor(weights_loc, dtype=torch.float32).to(DEVICE)
# ------------------------------------------------------------------------------

# Pesos para severidade (0: Normal, 1: Soft, 2: Hard)
labels_sev_arr = np.array(seq_sev)
sev_counts = np.bincount(labels_sev_arr)
weights_sev = 1.0 / (sev_counts + 1e-5)
weights_sev = weights_sev / np.sum(weights_sev) * len(sev_counts)
weights_sev_tensor = torch.tensor(weights_sev, dtype=torch.float32).to(DEVICE)

# Peso para detecção binária (relação saudáveis / falhas)
pos_weight_value = (len(y_det) - np.sum(y_det)) / np.sum(y_det)
pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32).to(DEVICE)

# 2. Inicialização dos critérios penalizados
model = MultiTaskFaultDiagnosisGNN(num_nodes=num_nodes, in_features=1, hidden_dim=64).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0015, weight_decay=1e-4)

criterion_det = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
criterion_loc = nn.CrossEntropyLoss(weight=weights_loc_tensor)
criterion_sev = nn.CrossEntropyLoss(weight=weights_sev_tensor)

print("\n[TREINO] Inicializando balanceamento adaptativo com compatibilidade de 17 canais...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    
    for batch_X, b_det, b_loc, b_sev in train_loader:
        batch_X = batch_X.to(DEVICE)
        b_det, b_loc, b_sev = b_det.to(DEVICE), b_loc.to(DEVICE), b_sev.to(DEVICE)
        
        out_det, out_loc, out_sev = model(batch_X)
        
        loss_det = criterion_det(out_det.squeeze(), b_det)
        loss_loc = criterion_loc(out_loc, b_loc)
        loss_sev = criterion_sev(out_sev, b_sev)
        
        # Ponderação de perda multitarefa estável
        total_loss = (2.0 * loss_det) + (1.0 * loss_loc) + (1.0 * loss_sev)
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss.item()
        
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Época {epoch+1:02d}/{EPOCHS} -> Perda Ponderada Global: {epoch_loss/len(train_loader):.4f}")

torch.save({'model_state': model.state_dict(), 'features': features}, "multitask_monster_model.pt")
print("\n[SUCESSO] O modelo multitarefa foi re-treinado com alinhamento de shape!")
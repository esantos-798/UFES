import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from sklearn.metrics import r2_score

# Silenciar logs excessivos do Optuna para o terminal ficar limpo
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=== Inicializando Otimização da MLP com Optuna (4 Variáveis) ===")

# 1. Carregar os datasets
df_train = pd.read_csv('dataset_treinamento_split.csv')
df_test = pd.read_csv('dataset_teste_split.csv')

colunas_x = ['Var_A', 'Var_B', 'Var_C', 'Var_D']
X_train = torch.tensor(df_train[colunas_x].values, dtype=torch.float32)
y_train = torch.tensor(df_train['Alvo_Y'].values, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(df_test[colunas_x].values, dtype=torch.float32)
y_test = torch.tensor(df_test['Alvo_Y'].values, dtype=torch.float32).reshape(-1, 1)

# 2. Arquitetura Dinâmica da MLP
class DinamicaMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation_name, output_dim):
        super(DinamicaMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        
        if activation_name == 'ReLU':
            self.activation = nn.ReLU()
        elif activation_name == 'Tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Sigmoid()
            
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        return self.layer2(self.activation(self.layer1(x)))

# 3. Função Objetivo do Optuna
def objective(trial):
    # Espaço de busca (Hiperparâmetros candidatos)
    hidden_dim = trial.suggest_int('hidden_dim', 4, 32)
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    activation_name = trial.suggest_categorical('activation', ['ReLU', 'Tanh', 'Sigmoid'])
    epochs = trial.suggest_int('epochs', 100, 300)
    
    # Inicializar modelo sugerido pelo trial
    model = DinamicaMLP(input_dim=4, hidden_dim=hidden_dim, activation_name=activation_name, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Loop de treino do trial
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
    # Avaliação no conjunto de teste
    model.eval()
    with torch.no_grad():
        preds = model(X_test).numpy().flatten()
    
    # Queremos maximizar o R² Score
    r2 = r2_score(y_test.numpy().flatten(), preds)
    return r2

# 4. Executar o Estudo do Optuna
study = optuna.create_study(direction='maximize')
print("-> Rodando 30 tentativas de otimização estrutural...")
study.optimize(objective, n_trials=30)

print("\n=========================================================")
print("MELHORES HIPERPARÂMETROS ENCONTRADOS PELO OPTUNA:")
print("=========================================================")
print(f"Melhor R² Score: {study.best_value:.4f}")
for key, value in study.best_params.items():
    print(f"{key}: {value}")
print("=========================================================")
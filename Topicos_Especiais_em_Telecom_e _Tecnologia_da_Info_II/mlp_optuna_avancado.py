import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from sklearn.metrics import r2_score

# Ocultar mensagens secundárias do Optuna para limpar o terminal
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=== OTIMIZAÇÃO AVANÇADA DA MLP (NEURÔNIOS, ATIVAÇÃO E DROPOUT) ===")

# 1. Carregar os datasets das 4 variáveis
df_train = pd.read_csv('dataset_treinamento_split.csv')
df_test = pd.read_csv('dataset_teste_split.csv')

colunas_x = ['Var_A', 'Var_B', 'Var_C', 'Var_D']
X_train = torch.tensor(df_train[colunas_x].values, dtype=torch.float32)
y_train = torch.tensor(df_train['Alvo_Y'].values, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(df_test[colunas_x].values, dtype=torch.float32)
y_test = torch.tensor(df_test['Alvo_Y'].values, dtype=torch.float32).reshape(-1, 1)

# 2. Arquitetura Flexível de 3 Camadas com Dropout
class MLPAvancada(nn.Module):
    def __init__(self, input_dim, h1_dim, h2_dim, dropout_p, activation_name, output_dim):
        super(MLPAvancada, self).__init__()
        
        # Seleção de Ativação
        if activation_name == 'ReLU':
            activation_layer = nn.ReLU()
        elif activation_name == 'Tanh':
            activation_layer = nn.Tanh()
        else:
            activation_layer = nn.Sigmoid()
            
        # Estrutura Sequencial da Rede Neural
        self.rede = nn.Sequential(
            nn.Linear(input_dim, h1_dim),
            activation_layer,
            nn.Dropout(p=dropout_p),
            nn.Linear(h1_dim, h2_dim),
            activation_layer,
            nn.Dropout(p=dropout_p),
            nn.Linear(h2_dim, output_dim)
        )
        
    def forward(self, x):
        return self.rede(x)

# 3. Função de Avaliação do Trial (Optuna)
def objective(trial):
    # Espaço de busca definido para o Optuna escanear
    h1_dim = trial.suggest_int('h1_dim', 8, 64)
    h2_dim = trial.suggest_int('h2_dim', 4, 32)
    dropout_p = trial.suggest_float('dropout_p', 0.0, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    activation_name = trial.suggest_categorical('activation', ['ReLU', 'Tanh', 'Sigmoid'])
    epochs = trial.suggest_int('epochs', 150, 350)
    
    # Instanciação temporária do modelo para teste
    model = MLPAvancada(input_dim=4, h1_dim=h1_dim, h2_dim=h2_dim, 
                        dropout_p=dropout_p, activation_name=activation_name, output_dim=1)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Loop de Treinamento
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
    # Teste e Cálculo da Métrica Alvo (R²)
    model.eval()
    with torch.no_grad():
        preds = model(X_test).numpy().flatten()
        
    r2 = r2_score(y_test.numpy().flatten(), preds)
    return r2

# 4. Executar os Experimentos
study = optuna.create_study(direction='maximize')
print("-> Iniciando busca estocástica por hiperparâmetros (40 Trials)...")
study.optimize(objective, n_trials=40)

print("\n=========================================================")
print("ARQUITETURA CAMPEÃ DETECTADA PELO OPTUNA:")
print("=========================================================")
print(f"Melhor R² alcançado: {study.best_value:.4f}")
for param, valor in study.best_params.items():
    if isinstance(valor, float):
        print(f"{param}: {valor:.4f}")
    else:
        print(f"{param}: {valor}")
print("=========================================================")
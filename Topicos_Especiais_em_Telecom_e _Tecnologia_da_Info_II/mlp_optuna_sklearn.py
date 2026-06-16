import pandas as pd
import numpy as np
import optuna
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

# Silenciar logs do Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=== OTIMIZAÇÃO DA MLP VIA SCOIKIT-LEARN (SOLUÇÃO SEM DLL) ===")

# 1. Carregar os datasets
df_train = pd.read_csv('dataset_treinamento_split.csv')
df_test = pd.read_csv('dataset_teste_split.csv')

colunas_x = ['Var_A', 'Var_B', 'Var_C', 'Var_D']
X_train = df_train[colunas_x].values
y_train = df_train['Alvo_Y'].values
X_test = df_test[colunas_x].values
y_test = df_test['Alvo_Y'].values

# 2. Função Objetivo para o Optuna
def objective(trial):
    # Definindo espaço de busca de neurônios para as 2 camadas ocultas
    h1 = trial.suggest_int('h1_dim', 8, 64)
    h2 = trial.suggest_int('h2_dim', 4, 32)
    
    # O scikit-learn usa 'alpha' para regularização L2 (equivalente ao controle de overfitting do dropout)
    alpha = trial.suggest_float('alpha', 1e-5, 1e-2, log=True)
    lr_init = trial.suggest_float('lr', 1e-3, 1e-1, log=True)
    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic'])
    
    # Criando a MLP de 3 camadas (Entrada -> h1 -> h2 -> Saída)
    model = MLPRegressor(
        hidden_layer_sizes=(h1, h2),
        activation=activation,
        alpha=alpha,
        learning_rate_init=lr_init,
        max_iter=300,
        random_state=42
    )
    
    # Treinar
    model.fit(X_train, y_train)
    
    # Predizer e Avaliar
    preds = model.predict(X_test)
    return r2_score(y_test, preds)

# 3. Rodar a Otimização
study = optuna.create_study(direction='maximize')
print("-> Iniciando 40 rodadas de otimização pelo Scikit-Learn...")
study.optimize(objective, n_trials=40)

print("\n=========================================================")
print("ARQUITETURA CAMPEÃ DETECTADA PELO OPTUNA (SKLEARN):")
print("=========================================================")
print(f"Melhor R² alcançado: {study.best_value:.4f}")
for param, valor in study.best_params.items():
    print(f"{param}: {valor}")
print("=========================================================")
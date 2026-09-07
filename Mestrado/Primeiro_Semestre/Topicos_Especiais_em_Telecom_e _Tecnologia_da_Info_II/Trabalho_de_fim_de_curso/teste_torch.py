import os
import sys

# Força o Windows a enxergar as pastas de DLLs da .venv
venv_lib_dir = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib")
if os.path.exists(venv_lib_dir):
    os.add_dll_directory(venv_lib_dir)

try:
    import torch
    print("-> PyTorch carregado com SUCESSO! Versão:", torch.__version__)
except Exception as e:
    print("Erro ao carregar:", e)
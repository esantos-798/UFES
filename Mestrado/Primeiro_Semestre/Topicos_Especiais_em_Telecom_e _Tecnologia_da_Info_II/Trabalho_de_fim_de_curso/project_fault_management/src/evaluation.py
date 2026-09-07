# src/evaluation.py
import numpy as np
import torch

def calculate_thresholds(model, loader, device):
    """Calcula o percentil 99 do erro absoluto para cada feature individualmente."""
    model.eval()
    all_errors = []
    
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            preds = model(batch_X).cpu().numpy()
            targets = batch_y.numpy()
            
            # Erro absoluto por amostra e por feature
            error = np.abs(targets - preds)
            all_errors.append(error)
            
    all_errors = np.vstack(all_errors)
    # Percentil 99 coluna por coluna (para cada indicador de falha - FI)
    thresholds = np.percentile(all_errors, 99, axis=0)
    return thresholds
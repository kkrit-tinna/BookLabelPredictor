# metrics.py

import torch

def precision_at_k(model, data_loader, k: int = None, device: str):
    model.eval()
    total_precision = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_X, batch_Y in data_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)

            logits = model(batch_X)         
            probs = torch.sigmoid(logits)    

            sorted_index = probs.argsort(dim = 1, descending = True)
            sorted_labels = batch_Y.gather(1, sorted_index)

            cumsum = sorted_labels.cumsum(dim = 1)
            true_counts = sorted_labels.sum(dim = 1).long()
            
            mask = true_counts > 0
            if mask.sum().item() == 0:
                continue

            if k is None:
                rows = torch.arange(batch_Y.size(0), device=batch_Y.device)[mask]
                indices = (true_counts[mask] - 1)
                correct = cumsum[rows, indices]
                per_sample_precision = correct.float() / true_counts[mask].float()
            else:
                k = int(k)
                topk = sorted_labels[:, :k] 
                correct_all = topk.sum(dim=1)
            
                per_sample_precision = correct_all[mask].float() / float(k)

            total_precision += per_sample_precision.sum().item()
            total_count += per_sample_precision.numel()

    return (total_precision / total_count) if total_count > 0 else 0.0

def precision_at_ks(model, data_loader, k_list, device: str):
    """
    Compute Precision@k for multiple k values in one call.

    Returns: dict {k: precision_at_k}
    """
    results = {}
    for k in k_list:
        results[k] = precision_at_k(model, data_loader, k=k, device=device)
    return results

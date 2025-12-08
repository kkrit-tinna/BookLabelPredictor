# metrics.py

import torch

def precision_at_k(model, data_loader, k: int, device: str):
    model.eval()
    total_precision = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_X, batch_Y in data_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)

            logits = model(batch_X)           # [B, L]
            probs = torch.sigmoid(logits)     # [B, L]

            _, topk_indices = torch.topk(probs, k=k, dim=1)
            relevant = batch_Y.gather(1, topk_indices)   # [B, k]

            correct_per_sample = relevant.sum(dim=1)     # [B]
            precision_per_sample = correct_per_sample / k

            total_precision += precision_per_sample.sum().item()
            total_samples += batch_Y.size(0)

    return total_precision / total_samples


def precision_at_ks(model, data_loader, k_list, device: str):
    """
    Compute Precision@k for multiple k values in one call.

    Returns: dict {k: precision_at_k}
    """
    results = {}
    for k in k_list:
        results[k] = precision_at_k(model, data_loader, k=k, device=device)
    return results


def precision_at_k_per_sample(model, data_loader, device: str):
    """
    Per-sample Precision@k where k_i = number of true labels for sample i.

    For each sample i:
      k_i = sum(y_i)
      take top k_i predicted labels
      precision_i = (# correct in top k_i) / k_i

    Returns the average precision_i over all samples with at least one label.
    """
    model.eval()
    total_precision = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_X, batch_Y in data_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)

            logits = model(batch_X)           # [B, L]
            probs = torch.sigmoid(logits)     # [B, L]
            batch_size = batch_Y.size(0)

            for i in range(batch_size):
                true_mask = batch_Y[i] == 1
                k_i = int(true_mask.sum().item())

                # skip samples with no labels
                if k_i == 0:
                    continue

                # top k_i predictions for this sample
                topk_idx = torch.topk(probs[i], k=k_i, dim=0).indices  # [k_i]

                # how many of these are actually true labels
                relevant_i = batch_Y[i, topk_idx]                      # [k_i]
                correct_i = float(relevant_i.sum().item())

                precision_i = correct_i / k_i
                total_precision += precision_i
                total_samples += 1

    if total_samples == 0:
        return 0.0

    return total_precision / total_samples

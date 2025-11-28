import torch

def precision_at_k(model, data_loader, k: int, device: str):
    model.eval()
    total_precision = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_X, batch_Y in data_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)

            logits = model(batch_X)                  # [B, L]
            probs = torch.sigmoid(logits)           # [B, L]

            topk_vals, topk_indices = torch.topk(probs, k=k, dim=1)
            relevant = batch_Y.gather(1, topk_indices)   # [B, k]

            correct_per_sample = relevant.sum(dim=1)     # [B]
            precision_per_sample = correct_per_sample / k

            total_precision += precision_per_sample.sum().item()
            total_samples += batch_Y.size(0)

    return total_precision / total_samples

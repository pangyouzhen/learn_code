import torch
import torch.nn.functional as F


# simcse loss
def compute_loss(y_pred, tao=0.05, device="cuda"):
    idxs = torch.arange(0, y_pred.shape[0], device=device)
    y_true = idxs + 1 - idxs % 2 * 2
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    similarities = similarities - torch.eye(y_pred.shape[0], device=device) * 1e12
    similarities = similarities / tao
    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)


if __name__ == '__main__':
    batch_size = 64
    embedding = 768
    a = torch.randn(size=(batch_size * 2, embedding))
    print(compute_loss(a, device="cpu"))

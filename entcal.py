import torch


def batch_histc(batch, min_value=0.0, max_value=1.0, num_bins=16, adaptive=False):
    if batch.dim() > 2:
        batch = batch.flatten(1)
    if adaptive:
        max_value = batch.max(dim=1).values.unsqueeze(1)
        min_value = batch.min(dim=1).values.unsqueeze(1)
    q_step = (max_value - min_value) / num_bins
    q = ((batch - min_value) / q_step).floor().int().clamp_(0, num_bins-1)
    r = torch.arange(num_bins).reshape(1, -1, 1)
    hists = (q.unsqueeze(1) == r).sum(dim=2)
    return hists

def calculate_entropy(batch, min_value=0.0, max_value=1.0, num_bins=16, adaptive=False):
    """
    Calculate entropy for a batch of samples.
    """
    hists = batch_histc(batch, min_value, max_value, num_bins, adaptive)
    probs = hists / hists.sum(dim=1, keepdims=True)
    min_pos = torch.finfo(batch.dtype).resolution
    ent = (-probs * torch.log(probs.clamp_(min=min_pos))).sum(dim=1, keepdim=True)
    return ent
    

if __name__ == "__main__":
    a = torch.linspace(-3, 3, 100)[None,]
    b = torch.rand(1, 100)
    c = torch.empty_like(a).fill_(0.5)

    ent_a = calculate_entropy(a, -3, 3, 10)
    ent_b = calculate_entropy(b, 0, 1, 10)
    ent_c = calculate_entropy(c, 0, 1, 10)

    print("Entropy of A:", ent_a.item())
    print("Entropy of B:", ent_b.item())
    print("Entropy of C:", ent_c.item())

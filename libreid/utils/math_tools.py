import torch


def euclidean_dist(x, y):
    r"""Euclidean distance. The computing formula is `d(x,y) = || (\vec{x} - \vec{y})^T \codt (\vec{x} - \vec{y}) ||`

    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


def cosine_dist(x, y):
    r"""cosine similarity, range from [-1, 1]. The computing formula is `sim(x,y) = \frac{x \cdot y}{||x|| \cdot ||y||}`.

    Args:
        x(tensor): with shape [m, d]
        y(tensor): with shape [n, d]
    Returns:
        A distance matrix tensor with shape [m, n]
    """
    x = torch.nn.functional.normalize(x, dim=1, p=2)
    y = torch.nn.functional.normalize(y, dim=1, p=2)
    dist = x.matmul(y.t()).clamp(min=1e-12)
    return dist
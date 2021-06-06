import torch
import torch.nn.functional as F


class Exclusive(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, V_lookup_table):
        ctx.save_for_backward(inputs, targets, V_lookup_table)
        outputs = inputs.mm(V_lookup_table.t())
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets, V_lookup_table = ctx.saved_tensors
        grad_inputs = grad_outputs.mm(V_lookup_table) if ctx.needs_input_grad[0] else None
        for x, y in zip(inputs, targets):
            V_lookup_table[y] = F.normalize( (V_lookup_table[y] + x) / 2, p=2, dim=0)
        return grad_inputs, None, None


class ExLoss(torch.nn.Module):
    def __init__(self, t=1.0, weight=None):
        super(ExLoss, self).__init__()
        self.t = t
        self.weight = weight

    def create_lookup_table(self, num_clusters, feature_dim=1024):
        self.register_buffer("_lookup_table", torch.zeros(num_clusters, feature_dim, device="cuda:0"))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, inputs, targets):
        scores = Exclusive.apply(inputs, targets, self._lookup_table) * self.t
        loss = F.cross_entropy(scores, targets, weight=self.weight)
        return loss, scores


import torch


def make_optimizer(model, name, lr=0.1, **kwargs):
    """
    """
    param_group_other = []
    param_group_bias = []

    for key, param in model.named_parameters():
        if param.requires_grad:
            if "bias" in key:
                param_group_bias.append(param)
            else:
                param_group_other.append(param)

    # lr = kwargs['lr']
    lr_bias = kwargs.get('lr_bias', lr)
    weight_decay = kwargs.get('weight_decay', 0.0)
    weight_decay_bias = kwargs.get('weight_decay_bias', 0.0)
    momentum=kwargs.get('momentum', 0.0)

    params = []
    params += [{'params':param_group_other, "lr":lr, "weight_decay":weight_decay}]
    params += [{'params':param_group_bias, "lr":lr_bias, "weight_decay":weight_decay_bias}]

    if name == 'SGD':
        # optimizer = getattr(torch.optim, name)(model.parameters(), lr=kwargs['lr'], momentum=kwargs.get('momentum', 0.0))
        optimizer = getattr(torch.optim, name)(params, momentum=momentum)
    else:
        # optimizer = getattr(torch.optim, name)(model.parameters(), lr=kwargs['lr'])
        optimizer = getattr(torch.optim, name)(params)
    return optimizer


def clip_grad_value(parameters, clip_value):
    r"""Clips gradient of an iterable of parameters at specified value.
    Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        clip_value (float or int): maximum allowed value of the gradients.
            The gradients are clipped in the range
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    clip_value = float(clip_value)

    max_v = 0.0
    min_v = 0.0
    for p in filter(lambda p: p.grad is not None, parameters):
        # p.grad.data.clamp_(-clip_value, clip_value)   
        max_v = max(torch.max(p.grad.data).item(), max_v)
        min_v = min(torch.min(p.grad.data).item(), min_v)

    return max_v, min_v
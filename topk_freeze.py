import torch.nn as nn
from torch.optim import AdamW
import re
from transformers.optimization import get_scheduler

ALL_LAYERNORM_LAYERS = [nn.LayerNorm]

def find_layer_number(name):
    res = re.search('layer\.([0-9]+)\.', name)
    if not res:
        return None
    layer_num = res.group(1)
    return int(layer_num)


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

def create_optimizer_and_scheduler(model, k, train_args, max_steps):

    optimizer_kwargs = {"lr": train_args.learning_rate,
        "betas": (train_args.adam_beta1, train_args.adam_beta2),
        "eps": train_args.adam_epsilon
        }

    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    for n, p in model.named_parameters():
        layer_number = find_layer_number(n)
        if layer_number is not None and layer_number <= k:
            p.requires_grad = False

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)],
            "weight_decay": train_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)
    warmup_steps = max_steps * train_args.warmup_ratio


    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    return optimizer, lr_scheduler
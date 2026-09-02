from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_value, model_value in zip(
            self.model.state_dict().values(), model.state_dict().values()
        ):
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(self.decay).add_(model_value.detach(), alpha=1 - self.decay)
            else:
                ema_value.copy_(model_value)

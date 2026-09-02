from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def sample(model, agnostic, mask, garment=None, steps: int = 32, device: str = "cuda"):
    z = torch.randn(agnostic.shape, device=device)
    for i in range(steps):
        t = torch.full((agnostic.shape[0],), i / steps, device=device)
        velocity = model(z, t, agnostic, mask, garment)
        z = z + velocity / steps
    return z


def flow_matching_loss(model, z1, agnostic, mask, garment=None):
    batch_size = z1.shape[0]
    z0 = torch.randn_like(z1)
    t = torch.sigmoid(torch.randn(batch_size, device=z1.device))
    time = t.view(-1, 1, 1, 1)
    z_t = (1 - time) * z0 + time * z1
    target = z1 - z0
    prediction = model(z_t, t, agnostic, mask, garment)
    return F.mse_loss(prediction, target)


FMLoss = flow_matching_loss

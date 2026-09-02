from __future__ import annotations

from .dit import DiT


def build_model(model_config: dict, latent_size: tuple[int, int] | list[int]) -> DiT:
    return DiT(
        in_channels=model_config["in_channels"],
        emb_dim=model_config["embedding_dim"],
        latent_size=tuple(latent_size),
        depth=model_config["depth"],
        num_head=model_config["num_heads"],
        drop_p=model_config["attention_dropout"],
        forward_expansion=model_config["forward_expansion"],
        patch_size=model_config["patch_size"],
        forward_drop_p=model_config["forward_dropout"],
        cross_end=model_config["cross_attention_end"],
    )

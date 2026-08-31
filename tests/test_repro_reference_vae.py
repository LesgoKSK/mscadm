import torch

from repro.models.reference_vae import DumasReferenceVAE


def test_reference_vae_has_compatible_state_and_finite_loss() -> None:
    model = DumasReferenceVAE(condition_dim=5, target_dim=3, latent_dim=2, hidden_dim=4)
    loss = model.loss(torch.randn(6, 3), torch.randn(6, 5))
    assert torch.isfinite(loss["loss"])
    assert set(loss) == {"loss", "reconstruction", "kl"}

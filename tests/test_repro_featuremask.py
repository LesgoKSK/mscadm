import torch


def test_elementwise_mask_is_partial() -> None:
    torch.manual_seed(0)
    condition = torch.ones(64, 24, 20)
    masked = condition.masked_fill(torch.rand_like(condition) < 0.1, 0)
    fraction = float((masked == 0).float().mean())
    assert 0.08 < fraction < 0.12
    assert not torch.any((masked == 0).all(dim=(1, 2)))

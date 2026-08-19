import torch

from crowd_counter import DMCount


def test_dm_count_returns_one_density_channel():
    model = DMCount().eval()
    density = model(torch.zeros((1, 3, 32, 32)))
    assert tuple(density.shape) == (1, 1, 4, 4)

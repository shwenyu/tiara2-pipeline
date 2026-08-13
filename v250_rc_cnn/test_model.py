import torch

from model import RCShortResidualExpert


def test_zero_init_and_reverse_complement_invariance():
    model = RCShortResidualExpert(
        {"root": 3, "euk": 8, "prok": 2, "organelle": 2},
        embed_dim=8,
        channels=48,
        dropout=0.0,
    ).eval()
    tokens = torch.tensor([
        [0, 1, 2, 3, 0, 4, 4],
        [3, 3, 2, 1, 0, 2, 4],
    ])
    lengths = torch.tensor([5, 6])
    reverse = model.reverse_complement(tokens, lengths)
    assert reverse.tolist() == [
        [3, 0, 1, 2, 3, 4, 4],
        [1, 3, 2, 1, 0, 0, 4],
    ]
    with torch.inference_mode():
        outputs = model(tokens, lengths, rc_average=True)
    assert all(torch.count_nonzero(value) == 0 for value in outputs.values())

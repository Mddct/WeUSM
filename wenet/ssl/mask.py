import torch


def _sampler(pdf: torch.Tensor, num_samples: int):
    size = pdf.size()
    z = -torch.log(torch.rand(size))
    _, indices = torch.topk(pdf + z, num_samples)
    return indices


def compute_mask_indices(size,
                         mask_prob: float,
                         mask_length: int,
                         min_masks: int = 0):
    pass

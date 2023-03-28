from typing import Tuple
import torch


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    index = torch.arange(max_length, device=input.device)
    index_expand = index.unsqueeze(0)  # [1, max_length]
    length_expand = length.unsqueeze(1)  # [B, 1]
    return index_expand < length_expand


def upsampling(
        input: torch.Tensor, repeats: torch.Tensor,
        repeats_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
      Args:
          input: [B, L]
          repeats: [B, L]
          repetas_mask: [B, L]
    """
    B = input.size(0)
    repeats = repeats * repeats_mask
    new_seq_len = torch.sum(repeats, 1)
    new_seq_max_len = new_seq_len.max()

    repeats_mask_flat = repeats_mask.reshape([-1])
    repeats_mask_flat_idx = torch.nonzero(repeats_mask_flat)
    reteats_flat = repeats.reshape([-1])
    indices_value = torch.take_along_dim(reteats_flat, repeats_mask_flat_idx)

    input_flat = input.reshape([-1])
    input_flat_nonzero = torch.take_along_dim(input_flat,
                                              repeats_mask_flat_idx)
    input_updates = torch.repeat_interleave(input_flat_nonzero, indices_value)

    new_seq_mask = sequence_mask(new_seq_len, new_seq_max_len)
    new_seq_mask_flat = new_seq_mask.reshape([-1])
    new_index = torch.nonzero(new_seq_mask_flat)

    flat = torch.zeros(B * new_seq_max_len,
                       device=input.device,
                       dtype=input.dtype)
    zeros = torch.zeros(B * new_seq_max_len - input_flat_nonzero.size(0),
                        device=input.device,
                        dtype=new_index.dtype)

    input_updates = torch.cat([input_updates, zeros], dim=0)
    new_index = torch.cat([new_index.squeeze(1), zeros], dim=0)

    new_input = flat.scatter(0, new_index, input_updates)
    new_input[0] = input_updates[0]

    new_input = new_input.reshape(B, new_seq_max_len)

    return new_input, new_seq_mask

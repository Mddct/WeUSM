from typing import Callable
from torch.utils.data import IterDataPipe
import torchdata
import torch

from torchdata.datapipes import functional_datapipe


@functional_datapipe("try_map")
class TryMapIterDataPipe(IterDataPipe):

    def __new__(cls,
                datapipe: IterDataPipe,
                fn: Callable,
                input_col=None,
                output_col=None) -> None:

        def try_fn(x):
            try:
                return fn(x)
            except Exception as e:
                # eg: download error
                return None

        return datapipe.map(try_fn, input_col=input_col,
                            output_col=output_col)  # type: ignore


def dataset():
    """
    list file -> shuffle -> shard -> sort
            -> map read and torchaudio load
            -> filter -> spec_aug -> batch
    """
    pass

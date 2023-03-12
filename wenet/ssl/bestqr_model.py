import torch


class BestRQModel(torch.nn.Module):

    def __init__(self, encoder: torch.nn.Module) -> None:
        super().__init__()

    def forward(self):
        # should support nonstreamming and streamming
        # eg: full attenton and chunk or  dynamic chunk training
        pass

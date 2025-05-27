import torch

from train.model.base import AframeBase


class MLP(AframeBase):
    """Docstring for MLP"""

    def __init__(self, input_dim, embed_dim, layer_widths):
        super(MLP, self).__init__()
        self.input_dim = input_dim

        def block(in_feat, out_feat, normalize=True):
            layers = [torch.nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(torch.nn.BatchNorm1d(num_features=out_feat))
            layers.append(torch.nn.LeakyReLU())
            layers.append(torch.nn.Dropout(p=0.05, inplace=False))
            return layers

        self.fcblock = torch.nn.Sequential(
            *block(input_dim, layer_widths[0]),
            *[
                layers
                for i in range(len(layer_widths) - 1)
                for layers in block(layer_widths[i], layer_widths[i + 1])
            ],
            torch.nn.Linear(layer_widths[-1], embed_dim),
        )

    def forward(self, src):
        output = src.reshape(-1, self.input_dim)
        output = self.fcblock(output)
        return output

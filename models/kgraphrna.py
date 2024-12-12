# https://medium.com/@pytorch_geometric/link-prediction-on-heterogeneous-graphs-with-pyg-6d5c29677c70
# https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SAGEConv
# https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.to_hetero_transformer.to_hetero
# https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SAGEConv.html#torch_geometric.nn.conv.SAGEConv
# https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html#torch_geometric.nn.conv.GCNConv
from torch_geometric.nn import SAGEConv, to_hetero
from torch.nn import Linear
import os
import numpy as np
import random
import torch


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


# class EdgeDecoder(torch.nn.Module):  # NN decoder
#     def __init__(self, hidden_channels: int, srna: str = 'srna', mrna: str = 'mrna',):
#         super().__init__()
#         # 1 - define metadata
#         self.srna = srna
#         self.mrna = mrna
#         # 2 - build model
#         self.lin1 = Linear(2 * hidden_channels, hidden_channels)
#         self.lin2 = Linear(hidden_channels, 1)
#
#     def forward(self, z_dict, edge_label_index):
#         row, col = edge_label_index
#         z = torch.cat([z_dict[self.srna][row], z_dict[self.mrna][col]], dim=-1)
#
#         z = self.lin1(z).relu()
#         z = self.lin2(z)
#         return z.view(-1)


class EdgeDecoder(torch.nn.Module):  # dot-product decoder
    def __init__(self, srna: str = 'srna', mrna: str = 'mrna',):
        super().__init__()
        # 1 - define metadata
        self.srna = srna
        self.mrna = mrna

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        return (z_dict[self.srna][row] * z_dict[self.mrna][col]).sum(dim=-1)


class kGraphRNA(torch.nn.Module):
    def __init__(self, hidden_channels: int, seed: int = 2, srna: str = 'srna', mrna: str = 'mrna',
                 srna_to_mrna: str = 'targets'):
        super().__init__()
        # 1 - set seeds
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)  # Set a fixed value for the hash seed
        # 2 - define metadata
        self.srna = srna
        self.mrna = mrna
        self.srna_to_mrna = srna_to_mrna
        self.mrna_to_srna = f"rev_{srna_to_mrna}"
        metadata = ([self.srna, self.mrna],
                    [(self.srna, self.srna_to_mrna, self.mrna), (self.mrna, self.mrna_to_srna, self.srna)])
        # 3 - build model
        self.encoder = GNNEncoder(hidden_channels=hidden_channels, out_channels=hidden_channels)
        self.encoder = to_hetero(self.encoder, metadata, aggr='sum')
        self.decoder = EdgeDecoder(srna=self.srna, mrna=self.mrna)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)

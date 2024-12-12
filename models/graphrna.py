import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
# https://medium.com/@pytorch_geometric/link-prediction-on-heterogeneous-graphs-with-pyg-6d5c29677c70
# https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SAGEConv
# https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.to_hetero_transformer.to_hetero
# https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SAGEConv.html#torch_geometric.nn.conv.SAGEConv
# https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html#torch_geometric.nn.conv.GCNConv
from torch_geometric.nn import SAGEConv, to_hetero, HeteroConv, GCNConv
from typing import Tuple, List, Optional
import numpy as np
import random
import torch
import torch_geometric.transforms as T


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels: int = None):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        # self.conv1 = SAGEConv((-1, -1), hidden_channels)
        # self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        # x = self.conv1(x, edge_index).relu()
        # x = self.conv2(x, edge_index)
        return x


# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_srna: Tensor, x_mrna: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_srna = x_srna[edge_label_index[0]]
        edge_feat_mrna = x_mrna[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_srna * edge_feat_mrna).sum(dim=-1)


class GraphRNA(torch.nn.Module):
    def __init__(self, srna: str, mrna: str, srna_to_mrna: str, mrna_to_mrna: str, srna_num_embeddings: int,
                 mrna_num_embeddings: int, metadata: Tuple[List[str], List[Tuple[str, str, str]]], model_args: dict,
                 **kwargs):
        """

        Parameters
        ----------
        srna - str key for sRNA node type
        mrna - str key for mRNA node type
        srna_to_mrna - str key for sRNA-mRNA edge type to be predicted
        srna_num_embeddings - number of sRNA embeddings   # todo - check the meaning of that
        mrna_num_embeddings - number of mRNA embeddings
        metadata - ([<node_type_1>, <node_type_2>, ...],
                    [<edge_type_1>, <edge_type_2>, <edge_type_3>, ...])
        model_args
        """
        super().__init__()
        if kwargs.get('seed_torch', None) is not None:
            torch.random.set_rng_state(kwargs['seed_torch'])
        if kwargs.get('seed_numpy', None) is not None:
            np.random.set_state(kwargs['seed_numpy'])
        self.srna = srna
        self.mrna = mrna
        self.srna_to_mrna = srna_to_mrna
        self.mrna_to_mrna = mrna_to_mrna
        # self.movie_lin = torch.nn.Linear(20, hidden_channels)  # 20 is the num of mRNA features
        # self.mrna_lin = torch.nn.Linear(model_args['num_sim_feat'], model_args['hidden_channels'])
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for sRNAs and mRNAs
        self.srna_emb = torch.nn.Embedding(num_embeddings=srna_num_embeddings,
                                           embedding_dim=model_args['hidden_channels'])
        self.mrna_emb = torch.nn.Embedding(num_embeddings=mrna_num_embeddings,
                                           embedding_dim=model_args['hidden_channels'])
        print(self.srna_emb.weight.detach().cpu().numpy()[:6, 1])
        print(self.mrna_emb.weight.detach().cpu().numpy()[:6, 1])

        # instantiate homogeneous GNN
        self.convs = torch.nn.ModuleList()
        num_layers = 2
        out_channels = 2
        for _ in range(num_layers):
            conv = HeteroConv({
                (self.srna, self.srna_to_mrna, self.mrna): SAGEConv(model_args['hidden_channels'],
                                                                    model_args['hidden_channels']),
                # (self.srna, self.srna_to_mrna, self.mrna): SAGEConv((-1, -1)), model_args['hidden_channels']),
                (self.mrna, f"rev_{self.srna_to_mrna}", self.srna): SAGEConv(model_args['hidden_channels'],
                                                                             model_args['hidden_channels']),
                # (self.mrna, f"rev_{self.srna_to_mrna}", self.srna): SAGEConv((-1, -1), model_args['hidden_channels']),
                (self.mrna, self.mrna_to_mrna, self.mrna): GCNConv(-1, model_args['hidden_channels']),
                (self.mrna, f"rev_{self.mrna_to_mrna}", self.mrna): GCNConv(-1, model_args['hidden_channels'])
            }, aggr='sum')
            self.convs.append(conv)

        # self.lin = Linear(model_args['hidden_channels'], out_channels)

        # self.out_channels = 2  # Num of classes
        # self.gnn = GNN(model_args['hidden_channels'])
        #
        # # convert GNN model into a heterogeneous variant
        # self.gnn = to_hetero(module=self.gnn, metadata=metadata)

        self.classifier = Classifier()

    def forward(self, data: HeteroData, model_args: dict = None) -> Tensor:
        x_dict = {
          # "user": self.user_emb(data["user"].node_id),
          # "movie": self.movie_lin(data["movie"].x) + self.movie_emb(data["movie"].node_id),
          self.srna: self.srna_emb(data[self.srna].node_id),
          # data[self.mrna].x - should hold mRNA-mRNA similarity features
          # self.mrna: self.mrna_lin(data[self.mrna].x) + self.mrna_emb(data[self.mrna].node_id),
          self.mrna: self.mrna_emb(data[self.mrna].node_id),
        }

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        edge_index_dict = data.edge_index_dict
        # x_dict = self.gnn(x_dict, data.edge_index_dict)
        if model_args['add_sim']:
            edge_weight_dict = data.edge_attr_dict
            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict, edge_weight_dict)
                x_dict = {key: x.relu() for key, x in x_dict.items()}
        else:
            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict)
                x_dict = {key: x.relu() for key, x in x_dict.items()}

        pred = self.classifier(
            x_dict[self.srna],
            x_dict[self.mrna],
            data[self.srna, self.srna_to_mrna, self.mrna].edge_label_index,
        )

        return pred

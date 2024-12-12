import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv, GCNConv


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


# the final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions
class Classifier(torch.nn.Module):
    def forward(self, x_srna: Tensor, x_mrna: Tensor, edge_label_index: Tensor) -> Tensor:
        # convert node embeddings to edge-level representations
        edge_feat_srna = x_srna[edge_label_index[0]]
        edge_feat_mrna = x_mrna[edge_label_index[1]]

        # apply dot-product to get a prediction per supervision edge
        return (edge_feat_srna * edge_feat_mrna).sum(dim=-1)


class GraphRNA(torch.nn.Module):
    def __init__(self, srna: str, mrna: str, srna_to_mrna: str, srna_num_embeddings: int, mrna_num_embeddings: int,
                 model_args: dict):
        """

        Parameters
        ----------
        srna - str key for sRNA node type
        mrna - str key for mRNA node type
        srna_to_mrna - str key for sRNA-mRNA edge type to be predicted
        srna_num_embeddings - number of sRNA embeddings
        mrna_num_embeddings - number of mRNA embeddings
        model_args
        """
        super().__init__()
        self.srna = srna
        self.mrna = mrna
        self.srna_to_mrna = srna_to_mrna
        self.mrna_to_mrna = "similar"
        # learn two embedding matrices for sRNAs and mRNAs
        self.srna_emb = torch.nn.Embedding(num_embeddings=srna_num_embeddings,
                                           embedding_dim=model_args['hidden_channels'])
        self.mrna_emb = torch.nn.Embedding(num_embeddings=mrna_num_embeddings,
                                           embedding_dim=model_args['hidden_channels'])

        # instantiate GNN
        self.convs = torch.nn.ModuleList()
        num_layers = 2
        for _ in range(num_layers):
            conv = HeteroConv({
                (self.srna, self.srna_to_mrna, self.mrna): SAGEConv(model_args['hidden_channels'],
                                                                    model_args['hidden_channels']),
                (self.mrna, f"rev_{self.srna_to_mrna}", self.srna): SAGEConv(model_args['hidden_channels'],
                                                                             model_args['hidden_channels']),
                (self.mrna, self.mrna_to_mrna, self.mrna): GCNConv(-1, model_args['hidden_channels']),
                (self.mrna, f"rev_{self.mrna_to_mrna}", self.mrna): GCNConv(-1, model_args['hidden_channels'])
            }, aggr='sum')
            self.convs.append(conv)

        self.classifier = Classifier()

    def forward(self, data: HeteroData, model_args: dict = None) -> Tensor:
        x_dict = {
          self.srna: self.srna_emb(data[self.srna].node_id),
          self.mrna: self.mrna_emb(data[self.mrna].node_id),
        }

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        edge_index_dict = data.edge_index_dict
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

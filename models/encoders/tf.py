import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Module, Sequential, ModuleList, Linear, Conv1d
from torch_geometric.nn import radius_graph, knn_graph
from torch_scatter import scatter_sum, scatter_softmax
import numpy as np
from math import pi as PI

from ..common import GaussianSmearing, ShiftedSoftplus
from ..protein_features import ProteinFeatures


class AttentionInteractionBlock(Module):

    def __init__(self, hidden_channels, edge_channels, key_channels, num_heads=1):
        super().__init__()

        assert hidden_channels % num_heads == 0 
        assert key_channels % num_heads == 0

        self.hidden_channels = hidden_channels
        self.key_channels = key_channels
        self.num_heads = num_heads

        self.k_lin = Conv1d(hidden_channels, key_channels, 1, groups=num_heads, bias=False)
        self.q_lin = Conv1d(hidden_channels, key_channels, 1, groups=num_heads, bias=False)
        self.v_lin = Conv1d(hidden_channels, hidden_channels, 1, groups=num_heads, bias=False)

        self.weight_k_net = Sequential(
            Linear(edge_channels, key_channels//num_heads),
            ShiftedSoftplus(),
            Linear(key_channels//num_heads, key_channels//num_heads),
        )
        self.weight_k_lin = Linear(key_channels//num_heads, key_channels//num_heads)

        self.weight_v_net = Sequential(
            Linear(edge_channels, hidden_channels//num_heads),
            ShiftedSoftplus(),
            Linear(hidden_channels//num_heads, hidden_channels//num_heads),
        )
        self.weight_v_lin = Linear(hidden_channels//num_heads, hidden_channels//num_heads)

        self.centroid_lin = Linear(hidden_channels, hidden_channels)
        self.act = ShiftedSoftplus()
        self.out_transform = Linear(hidden_channels, hidden_channels)
        self.layernorm_attention = nn.LayerNorm(hidden_channels)
        self.layernorm_ffn = nn.LayerNorm(hidden_channels)

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x:  Node features, (N, H).
            edge_index: (2, E).
            edge_attr:  (E, H)
        """
        N = x.size(0)
        row, col = edge_index   # (E,) , (E,)

        # self-attention layer_norm
        y = self.layernorm_attention(x)

        # Project to multiple key, query and value spaces
        h_keys = self.k_lin(y.unsqueeze(-1)).view(N, self.num_heads, -1)    # (N, heads, K_per_head)
        h_queries = self.q_lin(y.unsqueeze(-1)).view(N, self.num_heads, -1) # (N, heads, K_per_head)
        h_values = self.v_lin(y.unsqueeze(-1)).view(N, self.num_heads, -1)  # (N, heads, H_per_head)

        # Compute keys and queries
        W_k = self.weight_k_net(edge_attr)  # (E, K_per_head)
        keys_j = self.weight_k_lin(W_k.unsqueeze(1) * h_keys[col])  # (E, heads, K_per_head)
        queries_i = h_queries[row]    # (E, heads, K_per_head)

        # Compute attention weights (alphas)
        qk_ij = (queries_i * keys_j).sum(-1)  # (E, heads)
        alpha = scatter_softmax(qk_ij, row, dim=0)

        # Compose messages
        W_v = self.weight_v_net(edge_attr)  # (E, H_per_head)
        msg_j = self.weight_v_lin(W_v.unsqueeze(1) * h_values[col])  # (E, heads, H_per_head)
        msg_j = alpha.unsqueeze(-1) * msg_j   # (E, heads, H_per_head)

        # Aggregate messages
        aggr_msg = scatter_sum(msg_j, row, dim=0, dim_size=N).view(N, -1) # (N, heads*H_per_head)
        x = aggr_msg + x
        y = self.layernorm_ffn(x)
        out = self.out_transform(self.act(y)) + x
        return out


class CFTransformerEncoder(Module):
    
    def __init__(self, hidden_channels=256, edge_channels=64, key_channels=128, num_heads=4, num_interactions=6, k=32, cutoff=10.0):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.edge_channels = edge_channels
        self.key_channels = key_channels
        self.num_heads = num_heads
        self.num_interactions = num_interactions
        self.k = k
        self.cutoff = cutoff

        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)
        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = AttentionInteractionBlock(
                hidden_channels=hidden_channels,
                edge_channels=edge_channels,
                key_channels=key_channels,
                num_heads=num_heads,
            )
            self.interactions.append(block)

    @property
    def out_channels(self):
        return self.hidden_channels

    def forward(self, node_attr, pos, batch):
        # edge_index = radius_graph(pos, self.cutoff, batch=batch, loop=False)
        edge_index = knn_graph(pos, k=self.k, batch=batch, flow='target_to_source')
        edge_length = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
        edge_attr = self.distance_expansion(edge_length)

        h = node_attr
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_attr)
        return h


# residue level graph transformer
class AAEmbedding(nn.Module):

    def __init__(self):
        super(AAEmbedding, self).__init__()

        self.hydropathy = {'#': 0, "I":4.5, "V":4.2, "L":3.8, "F":2.8, "C":2.5, "M":1.9, "A":1.8, "W":-0.9, "G":-0.4, "T":-0.7, "S":-0.8, "Y":-1.3, "P":-1.6, "H":-3.2, "N":-3.5, "D":-3.5, "Q":-3.5, "E":-3.5, "K":-3.9, "R":-4.5}
        self.volume = {'#': 0, "G":60.1, "A":88.6, "S":89.0, "C":108.5, "D":111.1, "P":112.7, "N":114.1, "T":116.1, "E":138.4, "V":140.0, "Q":143.8, "H":153.2, "M":162.9, "I":166.7, "L":166.7, "K":168.6, "R":173.4, "F":189.9, "Y":193.6, "W":227.8}
        self.charge = {**{'R':1, 'K':1, 'D':-1, 'E':-1, 'H':0.1}, **{x:0 for x in 'ABCFGIJLMNOPQSTUVWXYZ#'}}
        self.polarity = {**{x:1 for x in 'RNDQEHKSTY'}, **{x:0 for x in "ACGILMFPWV#"}}
        self.acceptor = {**{x:1 for x in 'DENQHSTY'}, **{x:0 for x in "RKWACGILMFPV#"}}
        self.donor = {**{x:1 for x in 'RKWNQHSTY'}, **{x:0 for x in "DEACGILMFPV#"}}
        ALPHABET = ['#', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y','V']
        self.embedding = torch.tensor([
            [self.hydropathy[aa], self.volume[aa] / 100, self.charge[aa], self.polarity[aa], self.acceptor[aa], self.donor[aa]]
            for aa in ALPHABET])#.cuda(4)

    def to_rbf(self, D, D_min, D_max, stride):
        D_count = int((D_max - D_min) / stride)
        D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
        D_mu = D_mu.view(1,-1)  # [1, K]
        D_expand = torch.unsqueeze(D, -1)  # [N, 1]
        return torch.exp(-((D_expand - D_mu) / stride) ** 2)

    def transform(self, aa_vecs):
        return torch.cat([
            self.to_rbf(aa_vecs[:, 0], -4.5, 4.5, 0.1),
            self.to_rbf(aa_vecs[:, 1], 0, 2.2, 0.1),
            self.to_rbf(aa_vecs[:, 2], -1.0, 1.0, 0.25),
            torch.sigmoid(aa_vecs[:, 3:] * 6 - 3),
        ], dim=-1)

    def dim(self):
        return 90 + 22 + 8 + 3

    def forward(self, x, raw=False):
        #B, N = x.size(0), x.size(1)
        #aa_vecs = self.embedding[x.view(-1)].view(B, N, -1)
        aa_vecs = self.embedding[x.view(-1)]
        rbf_vecs = self.transform(aa_vecs)
        return aa_vecs if raw else rbf_vecs


class TransformerLayer(nn.Module):
    def __init__(self, num_hidden, num_heads=4, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.dropout_attention = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)
        self.self_attention_norm = nn.LayerNorm(num_hidden)
        self.ffn_norm = nn.LayerNorm(num_hidden)

        self.attention = ResidueAttention(num_hidden, num_heads)
        self.ffn = PositionWiseFeedForward(num_hidden, num_hidden)

    def forward(self, h_V, h_E, E_idx):
        """ Parallel computation of full transformer layer """
        # Self-attention
        y = self.self_attention_norm(h_V)
        y = self.attention(y, h_E, E_idx)
        h_V = h_V + self.dropout_attention(y)

        # Position-wise feedforward
        y = self.ffn_norm(h_V)
        y = self.ffn(y)
        h_V = h_V + self.dropout_ffn(y)
        return h_V


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)

    def forward(self, h_V):
        h = F.relu(self.W_in(h_V))
        h = self.W_out(h)
        return h


class ResidueAttention(nn.Module):
    def __init__(self, num_hidden, num_heads=4):
        super(ResidueAttention, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden

        # Self-attention layers: {queries, keys, values, output}
        self.W_Q = nn.Linear(num_hidden, num_hidden, bias=False)
        self.W_K = nn.Linear(num_hidden*2, num_hidden, bias=False)
        self.W_V = nn.Linear(num_hidden*2, num_hidden, bias=False)
        self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)
        self.act = ShiftedSoftplus()
        self.layernorm = nn.LayerNorm(num_hidden)

    def forward(self, h_V, h_E, edge_index):
        """ Self-attention, graph-structured O(Nk)
        Args:
            h_V:            Node features           [N_batch, N_nodes, N_hidden]
            h_E:            Neighbor features       [N_batch, N_nodes, K, N_hidden]
            mask_attend:    Mask for attention      [N_batch, N_nodes, K]
        Returns:
            h_V:            Node update
        """

        # Queries, Keys, Values
        n_edges = h_E.shape[0]
        n_nodes = h_V.shape[0]
        n_heads = self.num_heads
        row, col = edge_index  # (E,) , (E,)

        d = int(self.num_hidden / n_heads)
        Q = self.W_Q(h_V).view([n_nodes, n_heads, 1, d])
        K = self.W_K(torch.cat([h_E, h_V[col]], dim=-1)).view([n_edges, n_heads, d, 1])
        V = self.W_V(torch.cat([h_E, h_V[col]], dim=-1)).view([n_edges, n_heads, d])
        # Attention with scaled inner product
        attend_logits = torch.matmul(Q[row], K).view([n_edges, n_heads]) # (E, heads)
        alpha = scatter_softmax(attend_logits, row, dim=0) / np.sqrt(d)
        # Compose messages
        msg_j = alpha.unsqueeze(-1) * V   # (E, heads, H_per_head)

        # Aggregate messages
        aggr_msg = scatter_sum(msg_j, row, dim=0, dim_size=n_nodes).view(n_nodes, -1) # (N, heads*H_per_head)
        h_V_update = self.W_O(self.act(aggr_msg))
        return h_V_update


class ResidueTF(nn.Module):
    def __init__(self, node_features, edge_features,
        hidden_dim, num_encoder_layers=2, k_neighbors=8):
        super(ResidueTF, self).__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        # Featurization layers
        self.residue_feat = AAEmbedding()
        self.features = ProteinFeatures(top_k=k_neighbors)

        # Embedding layers
        self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([TransformerLayer(hidden_dim, dropout=0.1) for _ in range(num_encoder_layers)])

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, S_id, R, residue_batch):
        # Prepare node and edge embeddings
        V, E, edge_index = self.features(X, S_id, residue_batch)
        V= torch.cat([V, self.residue_feat(R)], dim=-1)
        h_V = self.W_v(V)
        h_E = self.W_e(E)

        for layer in self.encoder_layers:
            h_V = layer(h_V, h_E, edge_index)
        return h_V


# hierachical graph transformer encoder
class HierEncoder(Module):
    def __init__(self, hidden_channels=256, edge_channels=64, key_channels=128, num_heads=4, num_interactions=6, k=32,
                 cutoff=10.0):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.edge_channels = edge_channels
        self.key_channels = key_channels
        self.num_heads = num_heads
        self.num_interactions = num_interactions
        self.k = k
        self.cutoff = cutoff

        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)
        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = AttentionInteractionBlock(
                hidden_channels=hidden_channels,
                edge_channels=edge_channels,
                key_channels=key_channels,
                num_heads=num_heads,
            )
            self.interactions.append(block)

        # Residue level settings
        self.residue_feat = AAEmbedding()
        self.features = ProteinFeatures(top_k=8)
        self.W_v = nn.Linear(hidden_channels+self.residue_feat.dim(), hidden_channels, bias=True)
        self.W_e = nn.Linear(self.features.feature_dimensions, hidden_channels, bias=True)
        self.residue_encoder_layers = nn.ModuleList([TransformerLayer(hidden_channels, dropout=0.1) for _ in range(3)])


    @property
    def out_channels(self):
        return self.hidden_channels

    def forward(self, node_attr, pos, batch, X=None, S_id=None, R=None, residue_batch=None, atom2residue=None, mask=None, node_level=False):
        edge_index = knn_graph(pos, k=self.k, batch=batch, flow='target_to_source')
        edge_length = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
        edge_attr = self.distance_expansion(edge_length)

        h = node_attr
        for interaction in self.interactions:
            h = interaction(h, edge_index, edge_attr)

        if node_level:
            return h

        E, residue_edge_index = self.features(X, S_id, residue_batch)
        h_protein = h[mask]
        V = torch.cat([self.residue_feat(R), scatter_sum(h_protein, atom2residue, dim=0)], dim=-1)
        h_V = self.W_v(V)
        h_E = self.W_e(E)

        for layer in self.residue_encoder_layers:
            h_V = layer(h_V, h_E, residue_edge_index)
        return h, h_V


if __name__ == '__main__':
    from torch_geometric.data import Data, Batch

    hidden_channels = 64
    edge_channels = 48
    key_channels = 32
    num_heads = 4

    data_list = []
    for num_nodes in [11, 13, 15]:
        data_list.append(Data(
            x = torch.randn([num_nodes, hidden_channels]),
            pos = torch.randn([num_nodes, 3]) * 2
        ))
    batch = Batch.from_data_list(data_list)

    model = CFTransformerEncoder(
        hidden_channels = hidden_channels,
        edge_channels = edge_channels,
        key_channels = key_channels,
        num_heads = num_heads,
    )
    out = model(batch.x, batch.pos, batch.batch)

    print(out)
    print(out.size())

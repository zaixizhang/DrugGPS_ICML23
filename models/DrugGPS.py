import sys

sys.path.append("..")
import numpy as np
from rdkit import RDConfig
import os
import torch
import torch.nn as nn
from torch.nn import Module, Linear, Embedding
from torch.nn import functional as F
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.data import Data, Batch
from rdkit.Chem import ChemicalFeatures

from .encoders import get_encoder, GNN_graphpred, MLP, WeightGNN, CFTransformerEncoder
from .common import *
from .vq import VQ
from utils import dihedral_utils, chemutils


ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable', 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}


class DrugGPS(Module):

    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim, vocab, device):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.device = device
        self.protein_atom_emb = Linear(protein_atom_feature_dim, config.hidden_channels)
        self.ligand_atom_emb = Linear(ligand_atom_feature_dim, config.hidden_channels)
        self.embedding = nn.Embedding(vocab.size() + 1, config.hidden_channels)
        self.encoder = get_encoder(config.encoder)
        self.comb_head = GNN_graphpred(num_layer=3, emb_dim=config.hidden_channels, JK='last',
                                       drop_ratio=0.5, graph_pooling='mean', gnn_type='gin')

        self.motif_mlp = MLP(in_dim=config.hidden_channels * 3, out_dim=config.hidden_channels, num_layers=1)
        self.alpha_mlp = MLP(in_dim=config.hidden_channels * 3, out_dim=1, num_layers=2)
        self.focal_mlp = MLP(in_dim=config.hidden_channels, out_dim=1, num_layers=1)
        self.dist_mlp = MLP(in_dim=protein_atom_feature_dim + ligand_atom_feature_dim, out_dim=1, num_layers=2)
        self.attach_mlp = MLP(in_dim=config.hidden_channels * 1, out_dim=1, num_layers=1)

        if config.subpocket_motif:
            self.mgraph = MGraph(config, vocab, device)
            self.WeightGNN = WeightGNN(num_layer=2, emb_dim=config.hidden_channels)

        self.smooth_cross_entropy = SmoothCrossEntropyLoss(reduction='mean', smoothing=0.1)
        self.pred_loss = nn.CrossEntropyLoss()
        self.comb_loss = nn.BCEWithLogitsLoss()
        self.three_hop_loss = torch.nn.MSELoss()
        self.focal_loss = nn.BCEWithLogitsLoss()
        self.dist_loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, protein_pos, protein_atom_feature, ligand_pos, ligand_atom_feature, batch_protein, batch_ligand, batch):
        h_protein = self.protein_atom_emb(protein_atom_feature)
        h_ligand = self.ligand_atom_emb(ligand_atom_feature)

        h_ctx, pos_ctx, batch_ctx, protein_mask = compose_context_stable(h_protein=h_protein, h_ligand=h_ligand,
                                                                         pos_protein=protein_pos, pos_ligand=ligand_pos,
                                                                         batch_protein=batch_protein,
                                                                         batch_ligand=batch_ligand)
        h_ctx, h_residue = self.encoder(node_attr=h_ctx, pos=pos_ctx, batch=batch_ctx, X=batch['residue_pos'],
                                        S_id=batch['res_idx'], R=batch['amino_acid'], residue_batch=batch['amino_acid_batch'], atom2residue=batch['atom2residue'], mask=protein_mask)
        focal_pred = self.focal_mlp(h_ctx)

        return focal_pred, protein_mask, h_ctx, pos_ctx, h_residue

    def forward_motif(self, h_ctx_focal, pos_ctx_focal, current_wid, current_atoms_batch, h_residue, residue_pos, amino_acid_batch):
        node_hiddens = scatter_add(h_ctx_focal, dim=0, index=current_atoms_batch)
        motif_hiddens = self.embedding(current_wid)

        center_pos = scatter_mean(pos_ctx_focal, dim=0, index=current_atoms_batch)
        residue_pos = residue_pos[:, 1, :]
        residue_index = torch.where(torch.norm(residue_pos - center_pos[amino_acid_batch], dim=1) < 6)
        residue_emb = torch.zeros_like(node_hiddens)
        added = scatter_add(h_residue[residue_index], dim=0, index=amino_acid_batch[residue_index])
        residue_emb[:added.shape[0]] = added

        #query global graph
        node_weight, motif_weight, global_weight = 0.8, 0.7, 1.5
        edge_index, edge_attr, node_feat = self.mgraph.output(residue_emb)
        residue_emb = self.WeightGNN(node_feat, self.embedding.weight, edge_index, edge_attr)[:residue_emb.shape[0]]

        pred_vecs = torch.cat([node_weight*node_hiddens, motif_weight*motif_hiddens, global_weight*residue_emb], dim=-1)
        pred_scores = torch.matmul(self.motif_mlp(pred_vecs), self.embedding.weight.transpose(1, 0))
        #_, preds = torch.max(pred_scores, dim=1)

        # random select in topk
        k = 5
        select_pool = torch.topk(pred_scores, k, dim=1)[1]
        index = torch.randint(k, (select_pool.shape[0],))
        preds = torch.cat([select_pool[i][index[i]].unsqueeze(0) for i in range(len(index))])
        return preds

    def forward_attach(self, mol_list, next_motif_smiles):
        cand_mols, cand_batch, new_atoms, one_atom_attach, intersection, attach_fail = chemutils.assemble(mol_list, next_motif_smiles)
        graph_data = Batch.from_data_list([chemutils.mol_to_graph_data_obj_simple(mol) for mol in cand_mols])
        comb_pred = self.comb_head(graph_data.x, graph_data.edge_index, graph_data.edge_attr, graph_data.batch).reshape(-1)
        slice_idx = torch.cat([torch.tensor([0]), torch.cumsum(cand_batch.bincount(), dim=0)], dim=0)
        # select max score
        #select = [(torch.argmax(comb_pred[slice_idx[i]:slice_idx[i + 1]]) + slice_idx[i]).item() for i in
        #          range(len(slice_idx) - 1)]

        # random select
        select = []
        for k in range(len(slice_idx) - 1):
            id = torch.multinomial(torch.exp(comb_pred[slice_idx[k]:slice_idx[k + 1]]).reshape(-1).float(), 1)
            select.append((id+slice_idx[k]).item())

        select_mols = [cand_mols[i] for i in select]
        new_atoms = [new_atoms[i] for i in select]
        one_atom_attach = [one_atom_attach[i] for i in select]
        intersection = [intersection[i] for i in select]
        return select_mols, new_atoms, one_atom_attach, intersection, attach_fail

    def forward_alpha(self, protein_pos, protein_atom_feature, ligand_pos, ligand_atom_feature, batch_protein,
                     batch_ligand, xy_index, rotatable, batch):
        # encode again
        h_protein = self.protein_atom_emb(protein_atom_feature)
        h_ligand = self.ligand_atom_emb(ligand_atom_feature)

        h_ctx, pos_ctx, batch_ctx, protein_mask = compose_context_stable(h_protein=h_protein, h_ligand=h_ligand,
                                                                         pos_protein=protein_pos, pos_ligand=ligand_pos,
                                                                         batch_protein=batch_protein,
                                                                         batch_ligand=batch_ligand)
        h_ctx, _ = self.encoder(node_attr=h_ctx, pos=pos_ctx, batch=batch_ctx, X=batch['residue_pos'],
                                        S_id=batch['res_idx'], R=batch['amino_acid'], residue_batch=batch['amino_acid_batch'], atom2residue=batch['atom2residue'], mask=protein_mask)
        h_ctx_ligand = h_ctx[~protein_mask]
        hx, hy = h_ctx_ligand[xy_index[:, 0]], h_ctx_ligand[xy_index[:, 1]]
        h_mol = scatter_add(h_ctx_ligand, dim=0, index=batch_ligand)
        h_mol = h_mol[rotatable]
        alpha = self.alpha_mlp(torch.cat([hx, hy, h_mol], dim=-1)) + self.alpha_mlp(torch.cat([hy, hx, h_mol], dim=-1))
        return alpha

    def get_loss(self, protein_pos, protein_atom_feature, ligand_pos, ligand_atom_feature, ligand_pos_torsion,
                 ligand_atom_feature_torsion, batch_protein, batch_ligand, batch_ligand_torsion, batch, update_mgraph=False, query_mgraph=False):
        self.device = protein_pos.device
        h_protein = self.protein_atom_emb(protein_atom_feature)
        h_ligand = self.ligand_atom_emb(ligand_atom_feature)

        loss_list = [0, 0, 0, 0, 0, 0]

        # Encode for motif prediction
        h_ctx, pos_ctx, batch_ctx, mask_protein = compose_context_stable(h_protein=h_protein, h_ligand=h_ligand,
                                                                         pos_protein=protein_pos, pos_ligand=ligand_pos,
                                                                         batch_protein=batch_protein,
                                                                         batch_ligand=batch_ligand)
        h_ctx, h_residue = self.encoder(node_attr=h_ctx, pos=pos_ctx, batch=batch_ctx, X=batch['residue_pos'],
                                        S_id=batch['res_idx'], R=batch['amino_acid'], residue_batch=batch['amino_acid_batch'], atom2residue=batch['atom2residue'], mask=mask_protein)  # (N_p+N_l, H)
        h_ctx_ligand = h_ctx[~mask_protein]
        h_ctx_protein = h_ctx[mask_protein]
        h_ctx_focal = h_ctx[batch['current_atoms']]
        pos_ctx_focal = pos_ctx[batch['current_atoms']]

        # Encode for torsion prediction
        if len(batch['y_pos']) > 0:
            h_ligand_torsion = self.ligand_atom_emb(ligand_atom_feature_torsion)
            h_ctx_torison, pos_ctx_torison, batch_ctx_torsion, mask_protein = compose_context_stable(h_protein=h_protein,
                                                                                                     h_ligand=h_ligand_torsion,
                                                                                                     pos_protein=protein_pos,
                                                                                                     pos_ligand=ligand_pos_torsion,
                                                                                                     batch_protein=batch_protein,
                                                                                                     batch_ligand=batch_ligand_torsion)
            h_ctx_torsion = self.encoder(node_attr=h_ctx_torison, pos=pos_ctx_torison, batch=batch_ctx_torsion, node_level=True)  # (N_p+N_l, H)
            h_ctx_ligand_torsion = h_ctx_torsion[~mask_protein]

        # next motif prediction
        node_hiddens = scatter_add(h_ctx_focal, dim=0, index=batch['current_atoms_batch'])
        center_pos = scatter_mean(pos_ctx_focal, dim=0, index=batch['current_atoms_batch'])
        residue_pos = batch['residue_pos'][:, 1, :]
        residue_index = torch.where(torch.norm(residue_pos - center_pos[batch['amino_acid_batch']], dim=1)<6)
        residue_emb = torch.zeros_like(node_hiddens)
        added = scatter_add(h_residue[residue_index], dim=0, index=batch['amino_acid_batch'][residue_index])
        residue_emb[:added.shape[0]] = added
        motif_hiddens = self.embedding(batch['current_wid'])
        if update_mgraph:
            self.mgraph.update_count(residue_emb.detach(), batch['next_wid'], batch['interaction'])
        if query_mgraph:
            edge_index, edge_attr, node_feat = self.mgraph.output(residue_emb.detach())
            residue_emb = self.WeightGNN(node_feat, self.embedding.weight, edge_index, edge_attr)[:residue_emb.shape[0]]
        pred_vecs = torch.cat([node_hiddens, motif_hiddens, residue_emb], dim=-1)
        pred_scores = torch.matmul(self.motif_mlp(pred_vecs), self.embedding.weight.transpose(1, 0))
        pred_loss = self.pred_loss(pred_scores, batch['next_wid'])
        loss_list[0] = pred_loss.item()

        # attachment prediction
        if len(batch['cand_labels']) > 0:
            cand_mols = batch['cand_mols']
            cand_emb = self.comb_head(cand_mols.x, cand_mols.edge_index, cand_mols.edge_attr, cand_mols.batch)
            #attach_pred = self.attach_mlp(torch.cat([cand_emb, residue_emb[batch['cand_mols_batch']]], dim=-1))
            attach_pred = self.attach_mlp(cand_emb)
            comb_loss = self.comb_loss(attach_pred, batch['cand_labels'].view(attach_pred.shape).float())
            loss_list[1] = comb_loss.item()
        else:
            comb_loss = 0

        # focal prediction
        focal_ligand_pred, focal_protein_pred = self.focal_mlp(h_ctx_ligand), self.focal_mlp(h_ctx_protein)
        focal_loss = self.focal_loss(focal_ligand_pred.reshape(-1), batch['ligand_frontier'].float()) +\
                     self.focal_loss(focal_protein_pred.reshape(-1), batch['protein_contact'].float())
        loss_list[2] = focal_loss.item()

        # distance matrix prediction
        if len(batch['true_dm']) > 0:
            input = torch.cat(
                [protein_atom_feature[batch['dm_protein_idx']], ligand_atom_feature[batch['dm_ligand_idx']]], dim=-1)
            pred_dist = self.dist_mlp(input)
            dm_loss = self.dist_loss(pred_dist, batch['true_dm'])/10
            loss_list[3] = dm_loss.item()
        else:
            dm_loss = 0

        # torsion prediction
        if len(batch['y_pos']) > 0:
            Hx = dihedral_utils.rotation_matrix_v2(batch['y_pos'])
            xn_pos = torch.matmul(Hx, batch['xn_pos'].permute(0, 2, 1)).permute(0, 2, 1)
            yn_pos = torch.matmul(Hx, batch['yn_pos'].permute(0, 2, 1)).permute(0, 2, 1)
            y_pos = torch.matmul(Hx, batch['y_pos'].unsqueeze(1).permute(0, 2, 1)).squeeze(-1)

            hx, hy = h_ctx_ligand_torsion[batch['ligand_torsion_xy_index'][:, 0]], h_ctx_ligand_torsion[
                batch['ligand_torsion_xy_index'][:, 1]]
            h_mol = scatter_add(h_ctx_ligand_torsion, dim=0, index=batch['ligand_element_torsion_batch'])
            alpha = self.alpha_mlp(torch.cat([hx, hy, h_mol], dim=-1))
            # rotate xn
            R_alpha = self.build_alpha_rotation(torch.sin(alpha).squeeze(-1), torch.cos(alpha).squeeze(-1))
            xn_pos = torch.matmul(R_alpha, xn_pos.permute(0, 2, 1)).permute(0, 2, 1)

            p_idx, q_idx = torch.cartesian_prod(torch.arange(3), torch.arange(3)).chunk(2, dim=-1)
            p_idx, q_idx = p_idx.squeeze(-1), q_idx.squeeze(-1)
            pred_sin, pred_cos = dihedral_utils.batch_dihedrals(xn_pos[:, p_idx],
                                                 torch.zeros_like(y_pos).unsqueeze(1).repeat(1, 9, 1),
                                                 y_pos.unsqueeze(1).repeat(1, 9, 1),
                                                 yn_pos[:, q_idx])
            dihedral_loss = torch.mean(
                dihedral_utils.von_Mises_loss(batch['true_cos'], pred_cos.reshape(-1), batch['true_sin'], pred_cos.reshape(-1))[batch['dihedral_mask']])
            torsion_loss = -dihedral_loss
            loss_list[4] = torsion_loss.item()
        else:
            torsion_loss = 0

        # dm: distance matrix
        loss = pred_loss + comb_loss + focal_loss + dm_loss + torsion_loss

        return loss, loss_list, residue_emb.detach()

    def build_alpha_rotation(self, alpha, alpha_cos=None):
        """
        Builds the alpha rotation matrix

        :param alpha: predicted values of torsion parameter alpha (n_dihedral_pairs)
        :return: alpha rotation matrix (n_dihedral_pairs, 3, 3)
        """
        H_alpha = torch.FloatTensor([[[1, 0, 0], [0, 0, 0], [0, 0, 0]]]).repeat(alpha.shape[0], 1, 1).to(self.device)

        if torch.is_tensor(alpha_cos):
            H_alpha[:, 1, 1] = alpha_cos
            H_alpha[:, 1, 2] = -alpha
            H_alpha[:, 2, 1] = alpha
            H_alpha[:, 2, 2] = alpha_cos
        else:
            H_alpha[:, 1, 1] = torch.cos(alpha)
            H_alpha[:, 1, 2] = -torch.sin(alpha)
            H_alpha[:, 2, 1] = torch.sin(alpha)
            H_alpha[:, 2, 2] = torch.cos(alpha)

        return H_alpha

class MGraph(Module):
    def __init__(self, config, vocab, device):
        super().__init__()
        self.device = device
        self.vq = VQ(config.num_prototypes, config.hidden_channels, device)
        self.motif_vocab_size = vocab.size()
        self.num_prototypes = config.num_prototypes
        self.prototype_count = torch.zeros(self.num_prototypes, self.motif_vocab_size)
        self.prototype_count_new = torch.zeros(self.num_prototypes, self.motif_vocab_size)
        self.construct_num_neighbors = 4
        self.cached_input=[]
    def cos_sim(self, input, topk=4):
        normalized_prototype = F.normalize(self.vq.get_k(), dim=1).to(self.device)
        cos = torch.matmul(F.normalize(input, dim=1), normalized_prototype.transpose(0, 1))
        index = torch.topk(cos, dim=1, k=topk).indices
        return index
    def init(self):
        self.prototype_count = self.prototype_count_new
        self.prototype_count_new = torch.zeros(self.num_prototypes, self.motif_vocab_size)
    def update_count(self, input, motif_lable, interaction):
        #cos similarity
        index = self.cos_sim(input, topk=4)
        self.prototype_count_new[index.reshape(-1), motif_lable] += interaction.cpu()
        self.cached_input.append(input.detach())
    def update_prototype(self):
        self.vq.update(torch.cat(self.cached_input))
        self.cached_input = []
    def output(self, input):
        dim_size = input.size(0)+self.prototype_count.size(0)+self.motif_vocab_size
        adj_matrix = torch.zeros(dim_size, dim_size).to(self.device)
        index = self.cos_sim(input, self.construct_num_neighbors)
        ind1 = index.transpose(0,1).reshape(-1)+input.size(0)
        ind0 = torch.arange(input.size(0)).repeat(self.construct_num_neighbors)
        adj_matrix[ind0, ind1] = 1
        #tf_idf
        sum_prototype = torch.sum(self.prototype_count, dim=1).unsqueeze(1).repeat(1, self.motif_vocab_size)
        sum_motif = torch.sum(torch.where(self.prototype_count > 0, 1, 0), dim=0).repeat(self.num_prototypes, 1)
        adj_matrix[input.size(0):-self.motif_vocab_size, input.size(0)+self.prototype_count.size(0):]\
            = self.prototype_count/(sum_prototype+1)*(torch.log((1+self.num_prototypes)/(1+sum_motif))+1)
        sparse_mat = adj_matrix.to_sparse()
        edge_index = sparse_mat.indices()
        edge_attr = sparse_mat.values()
        edge_index = torch.cat([edge_index, edge_index[torch.tensor([1, 0])]], dim=1)
        edge_attr = edge_attr.repeat(2)
        node_feat = torch.cat([input, self.vq.get_k()], dim=0)
        return edge_index, edge_attr, node_feat


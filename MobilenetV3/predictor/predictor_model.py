import math
import random
import torch
from torch import nn
from torch.nn import functional as F
from scipy.stats import spearmanr, kendalltau
from collections import defaultdict
import numpy as np
import igraph
from nas_201_api import NASBench201API as API201
import copy

from typing import Dict, List, Optional, Tuple

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, attn_mask):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if attn_mask is not None:
            attn.masked_fill_(attn_mask, -1e9)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        output, attn = self.attention(q, k, v, attn_mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output


def normalize_adj(adj):
    # Row-normalize matrix
    last_dim = adj.size(-1)
    rowsum = adj.sum(1, keepdim=True)
    return torch.div(adj, rowsum)


def graph_pooling(inputs, num_vertices):
    out = inputs.sum(1)
    return torch.div(out, num_vertices.unsqueeze(-1).expand_as(out))

class DirectedGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight1 = nn.Parameter(torch.zeros((in_features, out_features)))
        self.weight2 = nn.Parameter(torch.zeros((in_features, out_features)))
        self.fc1 = nn.Linear(out_features * 2, out_features)
        self.dropout = nn.Dropout(0.1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight1.data)
        nn.init.xavier_uniform_(self.weight2.data)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

    def forward(self, inputs, adj):
        norm_adj = normalize_adj(adj)
        output1 = F.relu(torch.matmul(norm_adj, torch.matmul(inputs.float(), self.weight1)))
        inv_norm_adj = normalize_adj(adj.transpose(1, 2))
        output2 = F.relu(torch.matmul(inv_norm_adj, torch.matmul(inputs.float(), self.weight2)))
        out = self.fc1(torch.cat([output1, output2], dim=2))
        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class ProjectionLayer(nn.Module):
    def __init__(self, raw_feature_dim, feature_dim):
        super().__init__()
        self.raw_feature_dim = raw_feature_dim
        self.feature_dim = feature_dim

    def forward(self, x, dynamic_params):
        weight = dynamic_params['projection_weight']
        bias = dynamic_params['projection_bias']
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        if weight.dim() == 2:
            weight = weight.unsqueeze(0)
            
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        # [B, 1, In] @ [B, Out, In].T --> [B, 1, Out]
        x_reshaped = x.unsqueeze(1)
        output = torch.bmm(x_reshaped, weight.transpose(1, 2)).squeeze(1)
        output = output + bias
        return output


class DynamicDatasetExtractor(nn.Module):

    def __init__(self, raw_feature_dim=1, feature_dim=128, hidden_dim=32, num_dynamic_layers=3):
        super().__init__()
        self.raw_feature_dim = raw_feature_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_dynamic_layers = num_dynamic_layers

        if self.raw_feature_dim != self.feature_dim:
            self.projection = ProjectionLayer(self.raw_feature_dim, self.feature_dim)
        else:
            self.projection = None

    def forward(self, raw_features, dynamic_params):

        if raw_features.dim() == 1:
            raw_features = raw_features.unsqueeze(0)
            
        if self.projection and 'projection_weight' in dynamic_params:
            current_features = self.projection(raw_features, dynamic_params)
        else:
            if raw_features.size(-1) != self.feature_dim:
                 raise ValueError("Input feature dimension mismatch and no projection layer defined.")
            current_features = raw_features

        for i in range(self.num_dynamic_layers):
            weight = dynamic_params[f'layer{i}_weight']
            bias = dynamic_params[f'layer{i}_bias']

            if weight.dim() == 2:
                weight = weight.unsqueeze(0)
            if bias.dim() == 1:
                bias = bias.unsqueeze(0)
                
            hidden = torch.bmm(current_features.unsqueeze(1), weight.transpose(1, 2)).squeeze(1)
            hidden = hidden + bias

            if i == self.num_dynamic_layers - 1:
                current_features = torch.tanh(hidden)
            else:
                current_features = F.relu(hidden)
                
        return current_features


class HyperNetwork(nn.Module):
    def __init__(self, arch_feature_dim, raw_feature_dim, dataset_feature_dim,
                 hidden_dim=128, num_layers=6, num_dynamic_layers=3):
        super().__init__()

        self.num_layers = num_layers
        self.dataset_feature_dim = dataset_feature_dim
        self.raw_feature_dim = raw_feature_dim
        self.hidden_dim = hidden_dim
        self.num_dynamic_layers = num_dynamic_layers
        self.total_params = self._calculate_total_params()

        hyper_layers = []
        input_dim = arch_feature_dim
        for i in range(num_layers):
            output_dim = self.hidden_dim if i < num_layers - 1 else self.total_params
            hyper_layers.append(nn.Linear(input_dim, output_dim))
            if i < num_layers - 1:
                hyper_layers.append(nn.ReLU())
            input_dim = output_dim
        self.hyper_mlp = nn.Sequential(*hyper_layers)

    def _calculate_total_params(self):
        projection_params = 0
        if self.raw_feature_dim != self.dataset_feature_dim:
            projection_params = (self.raw_feature_dim * self.dataset_feature_dim + self.dataset_feature_dim)

        core_params = 0
        
        if self.num_dynamic_layers == 0:
            return projection_params

        if self.num_dynamic_layers == 1:
            # only one layer： dataset_feature_dim -> dataset_feature_dim
            core_params = self.dataset_feature_dim * self.dataset_feature_dim + self.dataset_feature_dim
        else:
            # Layer 0 (Input Layer): dataset_feature_dim -> hidden_dim
            core_params += self.dataset_feature_dim * self.hidden_dim + self.hidden_dim
            
            # Layer 1 to L-2 (Middle Hidden Layers): hidden_dim -> hidden_dim
            num_middle_layers = self.num_dynamic_layers - 2
            params_per_middle_layer = self.hidden_dim * self.hidden_dim + self.hidden_dim
            core_params += max(0, num_middle_layers) * params_per_middle_layer

            # Layer L-1 (Output Layer): hidden_dim -> dataset_feature_dim
            core_params += self.hidden_dim * self.dataset_feature_dim + self.dataset_feature_dim
            
        return projection_params + core_params

    def forward(self, arch_features):
        flat_params = self.hyper_mlp(arch_features)
        dynamic_params = self._organize_params(flat_params)
        return dynamic_params

    def _organize_params(self, flat_params):
        batch_size = flat_params.size(0)
        dynamic_params = {}
        start = 0

        if self.raw_feature_dim != self.dataset_feature_dim:
            projection_weight_size = self.raw_feature_dim * self.dataset_feature_dim
            projection_bias_size = self.dataset_feature_dim

            dynamic_params['projection_weight'] = flat_params[:, start:start + projection_weight_size].view(batch_size,self.dataset_feature_dim,self.raw_feature_dim)
            start += projection_weight_size

            dynamic_params['projection_bias'] = flat_params[:, start:start + projection_bias_size].view(batch_size,self.dataset_feature_dim)
            start += projection_bias_size

        if self.num_dynamic_layers >= 1:
            
            # --- Layer 0 (Input Layer) ---
            input_dim = self.dataset_feature_dim
            output_dim = self.hidden_dim if self.num_dynamic_layers > 1 else self.dataset_feature_dim
            weight_size = input_dim * output_dim
            bias_size = output_dim
            
            dynamic_params[f'layer0_weight'] = flat_params[:, start:start + weight_size].view(batch_size, output_dim, input_dim)
            start += weight_size
            dynamic_params[f'layer0_bias'] = flat_params[:, start:start + bias_size].view(batch_size, output_dim)
            start += bias_size

        # --- Layer 1 to L-2 (Middle Hidden Layers) ---
        for i in range(1, self.num_dynamic_layers - 1):
            input_dim = self.hidden_dim
            output_dim = self.hidden_dim
            weight_size = input_dim * output_dim
            bias_size = output_dim

            dynamic_params[f'layer{i}_weight'] = flat_params[:, start:start + weight_size].view(batch_size, output_dim, input_dim)
            start += weight_size
            dynamic_params[f'layer{i}_bias'] = flat_params[:, start:start + bias_size].view(batch_size, output_dim)
            start += bias_size
        
        # --- Layer L-1 (Output Layer) --- 仅当 L >= 2 时存在
        if self.num_dynamic_layers >= 2:
            i = self.num_dynamic_layers - 1
            input_dim = self.hidden_dim
            output_dim = self.dataset_feature_dim
            weight_size = input_dim * output_dim
            bias_size = output_dim

            dynamic_params[f'layer{i}_weight'] = flat_params[:, start:start + weight_size].view(batch_size, output_dim, input_dim)
            start += weight_size
            dynamic_params[f'layer{i}_bias'] = flat_params[:, start:start + bias_size].view(batch_size, output_dim)

        return dynamic_params


class ConditionalDatasetFeatureExtractor(nn.Module):
    def __init__(self, arch_feature_dim=16, raw_dataset_feature_dim=1, dataset_feature_dim=32,
                 hidden_dim=64, num_hyper_layers=2, num_dynamic_layers=3):
        super().__init__()

        self.arch_feature_dim = arch_feature_dim
        self.raw_dataset_feature_dim = raw_dataset_feature_dim
        self.dataset_feature_dim = dataset_feature_dim
        self.hidden_dim = hidden_dim
        self.num_dynamic_layers = num_dynamic_layers

        self.hyper_net = HyperNetwork(
            arch_feature_dim=arch_feature_dim,
            raw_feature_dim=raw_dataset_feature_dim,
            dataset_feature_dim=dataset_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_hyper_layers,
            num_dynamic_layers=num_dynamic_layers
        )

        self.dataset_extractor = DynamicDatasetExtractor(
            raw_feature_dim=raw_dataset_feature_dim,
            feature_dim=dataset_feature_dim,
            hidden_dim=hidden_dim,
            num_dynamic_layers=num_dynamic_layers
        )

    def forward(self, arch_features, dataset_raw_features):
        dynamic_params = self.hyper_net(arch_features)
        dataset_features = self.dataset_extractor(dataset_raw_features, dynamic_params)
        return dataset_features
        
class EnhancedPredictorWithHyperNet(nn.Module):
    def __init__(self, arch_feature_dim=32, raw_dataset_feature_dim=1,final_feature_dim=32, hdim=32):
        super().__init__()

        self.conditional_extractor = ConditionalDatasetFeatureExtractor(
            arch_feature_dim=arch_feature_dim,
            raw_dataset_feature_dim=raw_dataset_feature_dim,
            dataset_feature_dim=final_feature_dim
        )

        self.task_linear = nn.Linear(final_feature_dim, hdim)
        self.arch_linear = nn.Linear(arch_feature_dim, hdim)

        self.transformer = MultiHeadAttention(
            n_head=8, d_model=hdim, d_k=hdim, d_v=hdim, dropout=0.1
        )
        self.lastact = nn.Sequential(nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        self.mlp_head = nn.Sequential(
            nn.PReLU(),
            nn.Linear(hdim, 1)
        )

        self.loss = PairwiseMarginRankingLoss()

        # arch encoder
        self.ks_map = {3: 0, 5: 1, 7: 2}
        self.e_map = {3: 0, 4: 1, 6: 2}
        self.d_map = {2: 0, 3: 1, 4: 2}

        self.initial_hidden = 15
        self.gcn_hidden = arch_feature_dim
        self.max_nodes = 22
        self.gcn_layers = 6

        self.gcn = [DirectedGraphConvolution(self.initial_hidden if i == 0 else self.gcn_hidden, self.gcn_hidden)for i in range(self.gcn_layers)]
        self.gcn = nn.ModuleList(self.gcn)

        self.gcn_norm = nn.ModuleList([nn.LayerNorm(self.gcn_hidden) for _ in range(self.gcn_layers)])

    def to_categorical(self, labels, num_classes):
        one_hot = np.zeros((len(labels), num_classes), dtype=np.float32)
        one_hot[np.arange(len(labels)), labels.astype(int)] = 1
        return one_hot

    def get_arch(self, arch_dict_batch):
        batch_size = len(arch_dict_batch)
        adjacency_batch = []
        operations_batch = []
        num_batch = []

        fixed_adj = np.zeros((self.max_nodes, self.max_nodes))
        for i in range(self.max_nodes - 1):
            fixed_adj[i, i + 1] = 1

        for arch_dict in arch_dict_batch:
            ks_list = np.array([self.ks_map[k] for k in arch_dict['ks']])
            e_list = np.array([self.e_map[e] for e in arch_dict['e']])

            d_list = []
            for i, depth in enumerate(arch_dict['d']):
                d_list.extend([self.d_map[depth]] * 4)
            d_list = np.array(d_list[:20])

            ks_onehot = self.to_categorical(ks_list, 3)
            e_onehot = self.to_categorical(e_list, 3)
            d_onehot = self.to_categorical(d_list, 3)

            node_features = np.concatenate([ks_onehot, e_onehot, d_onehot], axis=1)

            padding = np.zeros((20, 6))
            node_features = np.concatenate([node_features, padding], axis=1)

            full_features = np.zeros((self.max_nodes, self.initial_hidden))
            full_features[1:21, :] = node_features

            operations_batch.append(full_features)
            adjacency_batch.append(fixed_adj)
            num_batch.append(self.max_nodes)

        return (
            torch.tensor(np.array(num_batch)),
            torch.tensor(np.array(adjacency_batch), dtype=torch.float32),
            torch.tensor(np.array(operations_batch), dtype=torch.float32)
        )

    def get_arch_features(self, num_vert_batch, adjacency_batch, operations_batch):
        numv, adj, out = num_vert_batch.to(self.device), adjacency_batch.to(self.device), operations_batch.to(
            self.device)

        numv = torch.full_like(numv, self.max_nodes)

        adj_with_diag = normalize_adj(adj + torch.eye(self.max_nodes, device=adj.device))

        for layer in self.gcn:
            out = layer(out, adj_with_diag)

        out = graph_pooling(out, numv)

        return out


    def forward(self, num_vert_batch, adjacency_batch, operations_batch, dataset_raw_features):
        arch_features = self.get_arch_features(num_vert_batch, adjacency_batch, operations_batch)
        dataset_features = self.conditional_extractor(arch_features, dataset_raw_features)

        # 特征融合和预测（使用你现有的代码）
        #arch_proj = self.arch_linear(arch_features).unsqueeze(1)
        #task_proj = self.task_linear(dataset_features).unsqueeze(1)

        #arch_features = arch_features.unsqueeze(1)
        dataset_features = dataset_features.unsqueeze(1)

        #fused_features = torch.cat([arch_proj, task_proj], dim=1)#arch_proj, task_proj
        #fused_output = self.transformer(fused_features, fused_features, fused_features, attn_mask=None)

        #arch_out = fused_output[:, 0, :]
        arch_out = dataset_features
        y_pred = self.mlp_head(arch_out)

        return y_pred.squeeze()


class PairwiseMarginRankingLoss(nn.Module):

    def __init__(self, margin: float = 0.5):
        super(PairwiseMarginRankingLoss, self).__init__()
        if margin < 0:
            raise ValueError("Margin must be non-negative.")
        self.margin = margin

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:

        if outputs.dim() > 1:
            outputs = outputs.squeeze(-1)
        if labels.dim() > 1:
            labels = labels.squeeze(-1)

        device = outputs.device
        N = outputs.shape[0]

        outputs_i = outputs.unsqueeze(1).repeat(1, N)
        outputs_j = outputs.unsqueeze(0).repeat(N, 1)

        labels_i = labels.unsqueeze(1).repeat(1, N)
        labels_j = labels.unsqueeze(0).repeat(N, 1)

        true_diff_sign = torch.sign(labels_i - labels_j)
        is_comparable = (true_diff_sign != 0).float()
        tmp = (outputs_i - outputs_j) * true_diff_sign

        loss_matrix = torch.log(1 + torch.exp(self.margin - tmp))
        loss_matrix_filtered = loss_matrix * is_comparable

        num_pairs = torch.sum(is_comparable)

        if num_pairs.item() == 0:
            return torch.tensor(0.0, device=device)

        loss = torch.sum(loss_matrix_filtered) / num_pairs

        return loss


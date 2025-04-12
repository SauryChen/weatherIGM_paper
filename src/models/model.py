import torch
import einops
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from itertools import product

import sys
sys.path.append("/weatherIGM/src")

from utils.variables import vars_table, cons_table

class MLPLayers(nn.Module):
    def __init__(self, units=[1024, 512, 256, 128], weight_norm=False, nonlin=nn.LeakyReLU(), dropout=0.2):
        super(MLPLayers, self).__init__()
        self.nonlin = nonlin
        self.dropout = dropout

        sequence = []
        for u0, u1 in zip(units[:-1], units[1:]):
            sequence.append(weight_norm(nn.Linear(u0, u1)) if weight_norm else nn.Linear(u0, u1))
            sequence.append(self.nonlin)
            sequence.append(nn.Dropout(self.dropout))
        sequence = sequence[:-2]

        self.sequential = nn.Sequential(*sequence)

    def forward(self, X):
        X = self.sequential(X)
        return X


class GATLayer(nn.Module):
    def __init__(self, features, hidden, heads, edge_dim, dropout):
        super(GATLayer, self).__init__()
        self.__dict__.update(locals())

        self.gat = GATConv(features, hidden, heads, edge_dim=edge_dim)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, node_features, edge_index, edge_attr=None):
        return self.dropout(self.relu(self.gat(node_features, edge_index, edge_attr)))


class Block(nn.Module):
    def __init__(
        self, 
        input_dim: int = 39*3 + 6,
        target_dim: int = 39,
        n_layers: int = 2,
        hidden_dim: int = 128,
        heads: int = 4,
        edge_dim: int = 8,
        dropout: float = 0.1,
        non_local_embed: int = 128
    ):
        super(Block, self).__init__()
        self.__dict__.update(locals())

        self.gat_layers = nn.ModuleList([GATLayer(input_dim, hidden_dim, heads, edge_dim, dropout)])
        for _ in range(n_layers-1):
            self.gat_layers.append(GATLayer(hidden_dim*heads, hidden_dim, heads, edge_dim, dropout))
        
        self.conv = nn.Sequential(
            nn.Conv2d(hidden_dim*heads, out_channels=self.non_local_embed, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(self.non_local_embed),
            nn.LeakyReLU()
        )

        self.mlp = MLPLayers(
            units=[hidden_dim*heads+self.non_local_embed, 512, 256, target_dim],
            nonlin=nn.LeakyReLU(),
            dropout=self.dropout
        )

    def construct_local_map(self,x,H,W):
        x = x.reshape((-1, H, W, x.shape[-1]))
        x = einops.rearrange(x, 'b h w c -> b c h w')

        paddings = (1,1,1,1)
        local_map = F.pad(x, paddings, mode='constant', value=0)
        local_map[:,:,:,0] = local_map[:,:,:,-2]
        local_map[:,:,:,-1] = local_map[:,:,:,1]
        local_map[:,:,0,:] = torch.cat([x[:,:,1, W//2-1:], x[:,:,1,:W//2+1]], dim=-1)
        local_map[:,:,-1,:] = torch.cat([x[:,:,-2, W//2-1:], x[:,:,-2,:W//2+1]], dim=-1)
        return local_map

    def graph_convert(self, features):
        batch_size, in_channel, height, width = features.shape
        poles = torch.zeros((batch_size, 2, in_channel)).to(features.device)
        features = einops.rearrange(features, 'b c h w -> b (h w) c')
        features = torch.cat((poles, features), dim=1)
        return einops.rearrange(features, 'b hxw c -> (b hxw) c')

    def forward(self, x, x_cons, edge_index, edge_attr=None):
        
        current = x
        H = x.shape[2]
        W = x.shape[3]
        
        x = torch.cat((x, x_cons), dim=1)
        # x -> Graph
        x = self.graph_convert(x)
        for layer_module in self.gat_layers:
            x = layer_module(x, edge_index, edge_attr)

        x = x.reshape((-1, H*W+2, x.shape[-1]))
        x = x[:,2:,:]

        non_local = self.construct_local_map(x, H, W)
        non_local = self.conv(non_local)
        non_local = einops.rearrange(non_local, 'b c h w -> (b h w) c')

        x = einops.rearrange(x, 'b hw c -> (b hw) c')
        x = torch.cat((x, non_local), dim=1)

        x = self.mlp(x)
        x = x.reshape((-1, H, W, self.target_dim))
        delta_x = einops.rearrange(x, 'b h w c -> b c h w')

        return delta_x + current[:, -delta_x.shape[1]:] # current is [t-12, t-6, t]


class model(nn.Module):
    """
    This model add input x [B,C,H,W] before output, 
    which means the model learns the change between the current and next time step.
    """
    def __init__(self, img_size, traced_vars, constant_vars, use_time_emb, hidden, heads, edge_dim, dropout, n_layers, non_local_embed, n_blocks):
        super(model, self).__init__()
        self.__dict__.update(locals())

        self.input_dim = len(traced_vars) * 3 + \
            len(constant_vars) + int(use_time_emb)
        self.inner_block_dim = len(traced_vars) * 3
        
        self.n_layers = n_layers
        self.non_local_embed = non_local_embed

        # since x is [t-12, t-6, t], and we use u and v features in t to construct the edge features.
        self.u_feats_ind = 2 * len(traced_vars) + np.array([
            vars_table.u_component_of_wind_10m,
            vars_table.u_component_of_wind_50,
            vars_table.u_component_of_wind_250,
            vars_table.u_component_of_wind_500, 
            vars_table.u_component_of_wind_600, 
            vars_table.u_component_of_wind_700, 
            vars_table.u_component_of_wind_850, 
            vars_table.u_component_of_wind_925
        ])

        self.v_feats_ind = 2 * len(traced_vars) + np.array([
            vars_table.v_component_of_wind_10m,
            vars_table.v_component_of_wind_50,
            vars_table.v_component_of_wind_250,
            vars_table.v_component_of_wind_500, 
            vars_table.v_component_of_wind_600, 
            vars_table.v_component_of_wind_700, 
            vars_table.v_component_of_wind_850, 
            vars_table.v_component_of_wind_925
        ])

        self.static_graph = self.create_edge_features(img_size[0], img_size[1], less_edge=True)

        self.layers = nn.Sequential()
        for i in range(n_blocks - 1):
            self.layers.add_module(
                f"block_{i}", 
                Block(self.input_dim, self.inner_block_dim, n_layers, hidden, heads, edge_dim, dropout, non_local_embed)
            )
        self.layers.add_module(
            f"block_{n_blocks-1}", 
            Block(self.input_dim, len(traced_vars), n_layers, hidden, heads, edge_dim, dropout, non_local_embed)
        )


    def forward(self, x, x_cons, time_embedding, y, metric, lat):
        """
        Args:
            x: [B, Vi, H, W], assert Vi == 3*Vo + time_embedding + 5 constant features.
            y: [B, Vo, H, W] Ground Truth value
            time_embedding: 
            variables: input vars name listed in yaml file
            out_variables: output vars name listed in yaml file
        """
        # processing the time_embedding
        B, _, H, W = x.shape
        time_embedding = -torch.cos(2 * np.pi * time_embedding / 8760)
        time_embedding = time_embedding.reshape(B, 1, 1, 1).repeat(1, 1, H, W)
        input_cons = torch.cat((x_cons, time_embedding), dim=1)

        graph_info = self.expend_graph(self.static_graph, B).to(x.device)
        edge_feats = self.get_edge_features(x)

        for layer in self.layers:
            x = layer(x, input_cons, graph_info, edge_feats)
        if metric is None:
            loss = None
        else:
            # import pdb; pdb.set_trace()
            loss = [m(x, y, self.traced_vars, lat) for m in metric]

        return loss, x
        
    def create_edge_features(self, height, width, less_edge=True):
        grid_index = torch.from_numpy(np.arange(2,2+height*width).reshape(height,width))
        grid_index_pad = F.pad(grid_index, (1,1,1,1), 'constant', 0)

        grid_index_pad[1:-1,0] = grid_index[:,-1]
        grid_index_pad[1:-1,-1] = grid_index[:,0]
        
        grid_index_pad[0,:] = 0
        grid_index_pad[-1,:] = 1

        edge_start, edge_end = [], []
        count = 0
        if less_edge:
            directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]
            edge_feats = []
        else:
            directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for h, w in product(range(height), range(width)):
            for r, c in directions:
                edge_start.append(grid_index_pad[h+1, w+1].item())
                edge_end.append(grid_index_pad[h+1+r,w+1+c].item())
            count+=1
        edges = torch.stack([torch.tensor(edge_start), torch.tensor(edge_end)], dim=0)
        return edges

    def expend_graph(self, graph, batch_size):

        nodes_count = self.img_size[0] * self.img_size[1] + 2
        graph = graph.unsqueeze(0).repeat(batch_size, 1, 1)
        start_idx = torch.arange(0, batch_size) * nodes_count
        start_idx = start_idx.reshape(batch_size, 1, 1)
        graph += start_idx
        graph = einops.rearrange(graph, 'b n e -> n (b e)')
        return graph

    def get_edge_features(self, x):
        """
        Args:
            x: [B, Vi, H, W], assert Vi == 3*Vo + time_embedding + 5 constant features.
            y: [B, Vo, H, W] Ground Truth value
            time_embedding: 
            variables: input vars name listed in yaml file
            out_variables: output vars name listed in yaml file
        """
        u_feats, v_feats = x[:, self.u_feats_ind], x[:, self.v_feats_ind]
        u_feats_r = einops.rearrange(u_feats, 'b c h w -> (b h w) c').unsqueeze(1)
        v_feats_r = einops.rearrange(v_feats, 'b c h w -> (b h w) c').unsqueeze(1)
        
        edge_feats = torch.cat((v_feats_r, -u_feats_r, u_feats_r, -v_feats_r), dim=1)
        edge_feats_n = einops.rearrange(edge_feats, 'b t c -> (b t) c')
        return edge_feats_n
    
    def evaluate(self, x, x_cons, time_embedding, y, out_variables, transform, metrics, lat, clim, log_postfix):
        _, preds = self.forward(x, x_cons, time_embedding, y, metric=None, lat=lat)
        return [m(preds, y, transform, out_variables, lat, clim, log_postfix) for m in metrics]
    
    def evaluate_iter(self, x, x_cons, time_embedding, y, metrics, lat):
        with torch.no_grad():
            _, preds = self.forward(x, x_cons, time_embedding, y, metric=None, lat=lat)
        return preds


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from omegaconf import OmegaConf
    cfg = OmegaConf.load('/weatherIGM/src/configs/global_forecast_model.yaml').model.net.init_args

    def print_gpu_memory():
        allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
        reserved = torch.cuda.memory_reserved(device) / 1024**2    # MB
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"[Memory] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB | Peak: {max_allocated:.2f} MB")

    torch.cuda.reset_peak_memory_stats(device)

    model = model(
        **cfg
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    torch.cuda.empty_cache()

    rand_input = torch.rand((1, 39*3, 121, 240)).to(device)
    cons_input = torch.rand((1, 5, 121, 240)).to(device)
    y = torch.rand((1, 39, 121, 240)).to(device)
    time_embedding = torch.rand((1)).to(device)
    pred = model(rand_input, cons_input, time_embedding, y, None, None)
    print("After forward pass:")
    print_gpu_memory()

    print(1111)

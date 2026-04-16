# -*- coding:utf-8 -*-
"""
DynaSTGCN: Dynamic Spatial-Temporal Graph Convolutional Network
===============================================================
Built on the ASTGCN-r backbone with three architectural contributions:

  1. Channel Attention Gate (CAG)
     SE-Net style gating that learns to weight Flow, Speed, and Occupancy independently.

  2. Adaptive Learned Adjacency Matrix
     Two sets of learnable node embeddings E1 and E2 produce a data-driven adjacency matrix.

  3. Learnable Periodic Temporal Embedding
     A T-dimensional learnable positional embedding is broadcast across all nodes.

Baseline (in_channels=1) uses the original ASTGCN path with zero modification.
Ablation studies are controlled via boolean flags passed to the model initialization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import scaled_Laplacian, cheb_polynomial


# =============================================================================
# CONTRIBUTION 1: Channel Attention Gate
# =============================================================================

class ChannelAttentionGate(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 1):
        super(ChannelAttentionGate, self).__init__()
        mid = max(1, in_channels // reduction)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, N, C, T)
        gap = x.mean(dim=[1, 3])                        # (B, C) — global descriptor
        weights = self.excitation(gap)                  # (B, C) — channel gates
        weights = weights.unsqueeze(1).unsqueeze(-1)    # (B, 1, C, 1) — broadcast
        return x * weights                              # (B, N, C, T)


# =============================================================================
# CONTRIBUTION 2: Adaptive Learned Adjacency
# =============================================================================

class AdaptiveAdjacency(nn.Module):
    def __init__(self, num_of_vertices: int, emb_dim: int = 10, device='cpu'):
        super(AdaptiveAdjacency, self).__init__()
        self.E1 = nn.Parameter(torch.randn(num_of_vertices, emb_dim).to(device))
        self.E2 = nn.Parameter(torch.randn(emb_dim, num_of_vertices).to(device))

    def forward(self):
        # Returns: (N, N) soft adjacency matrix
        return F.softmax(F.relu(self.E1 @ self.E2), dim=-1)


# =============================================================================
# CONTRIBUTION 3: Learnable Periodic Temporal Embedding
# =============================================================================

class TemporalPositionalEmbedding(nn.Module):
    def __init__(self, num_of_timesteps: int, out_channels: int, device='cpu'):
        super(TemporalPositionalEmbedding, self).__init__()
        self.emb = nn.Parameter(
            torch.randn(out_channels, num_of_timesteps).to(device) * 0.01
        )

    def forward(self, x):
        # x: (B, N, C, T); emb: (C, T) → broadcast (1, 1, C, T)
        return x + self.emb.unsqueeze(0).unsqueeze(0)


# =============================================================================
# Standard ASTGCN Attention Layers (Baseline — unchanged)
# =============================================================================

class Spatial_Attention_layer(nn.Module):
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps).to(DEVICE))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps).to(DEVICE))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices).to(DEVICE))
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices).to(DEVICE))

    def forward(self, x):
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)
        product = torch.matmul(lhs, rhs)
        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))
        return F.softmax(S, dim=1)


class Temporal_Attention_layer(nn.Module):
    def __init__(self, DEVICE, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(DEVICE))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).to(DEVICE))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(DEVICE))
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(DEVICE))

    def forward(self, x):
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        rhs = torch.matmul(self.U3, x)
        product = torch.matmul(lhs, rhs)
        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))
        return F.softmax(E, dim=1)


# =============================================================================
# Enhanced Graph Convolution (Static + Adaptive)
# =============================================================================

class cheb_conv_withSAt(nn.Module):
    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE))
            for _ in range(K)
        ])
        self.alpha_raw = nn.Parameter(torch.tensor(0.0).to(self.DEVICE))

    def forward(self, x, spatial_attention, adaptive_adj=None):
        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape
        outputs = []

        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step]
            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)

            for k in range(self.K):
                T_k = self.cheb_polynomials[k]
                T_k_with_at = T_k.mul(spatial_attention)

                if adaptive_adj is not None:
                    alpha = torch.sigmoid(self.alpha_raw)
                    # Blend static Chebyshev graph with adaptive learned graph
                    T_k_with_at = alpha * T_k_with_at + (1.0 - alpha) * adaptive_adj

                theta_k = self.Theta[k]
                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)
                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))


# =============================================================================
# ASTGCN Block (adaptive_adj-aware)
# =============================================================================

class ASTGCN_block(nn.Module):
    def __init__(self, DEVICE, in_channels, K, nb_chev_filter, nb_time_filter,
                 time_strides, cheb_polynomials, num_of_vertices, num_of_timesteps):
        super(ASTGCN_block, self).__init__()
        self.TAt = Temporal_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.SAt = Spatial_Attention_layer(DEVICE, in_channels, num_of_vertices, num_of_timesteps)
        self.cheb_conv_SAt = cheb_conv_withSAt(K, cheb_polynomials, in_channels, nb_chev_filter)
        self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3),
                                   stride=(1, time_strides), padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1),
                                       stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)

    def forward(self, x, adaptive_adj=None):
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        temporal_At = self.TAt(x)
        x_TAt = torch.matmul(
            x.reshape(batch_size, -1, num_of_timesteps), temporal_At
        ).reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps)

        spatial_At = self.SAt(x_TAt)
        spatial_gcn = self.cheb_conv_SAt(x, spatial_At, adaptive_adj)

        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))
        x_residual = self.ln(
            F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)
        ).permute(0, 2, 3, 1)

        return x_residual


# =============================================================================
# DynaSTGCN — Full Model with Ablation Flags
# =============================================================================

class ASTGCN_submodule(nn.Module):
    def __init__(self, DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter,
                 time_strides, cheb_polynomials, num_for_predict, len_input, num_of_vertices,
                 use_channel_attention=True, use_adaptive_adj=True, use_temporal_emb=True):
        super(ASTGCN_submodule, self).__init__()

        self.in_channels = in_channels
        self.DEVICE = DEVICE
        
        # Save ablation flags
        self.use_channel_attention = use_channel_attention
        self.use_adaptive_adj = use_adaptive_adj
        self.use_temporal_emb = use_temporal_emb

        if in_channels == 3:
            # --- DynaSTGCN Enhancement Modules ---
            if self.use_channel_attention:
                self.channel_attention = ChannelAttentionGate(in_channels=3, reduction=1)
            
            # Channel Fusion is always required for 3-channel input to compress to 1 channel
            self.channel_fusion = nn.Conv2d(3, 1, kernel_size=(1, 1), bias=True)
            
            if self.use_adaptive_adj:
                self.adaptive_adj_module = AdaptiveAdjacency(num_of_vertices, emb_dim=10, device=DEVICE)
                
            if self.use_temporal_emb:
                self.temporal_emb = TemporalPositionalEmbedding(len_input, out_channels=1, device=DEVICE)
                
            block_in_channels = 1
        else:
            # --- Baseline: no enhancement modules ---
            self.channel_attention = None
            self.channel_fusion = None
            self.adaptive_adj_module = None
            self.temporal_emb = None
            block_in_channels = in_channels

        # ASTGCN blocks — same topology for both paths
        self.BlockList = nn.ModuleList([
            ASTGCN_block(DEVICE, block_in_channels, K, nb_chev_filter, nb_time_filter,
                         time_strides, cheb_polynomials, num_of_vertices, len_input)
        ])
        self.BlockList.extend([
            ASTGCN_block(DEVICE, nb_time_filter, K, nb_chev_filter, nb_time_filter,
                         1, cheb_polynomials, num_of_vertices, len_input // time_strides)
            for _ in range(nb_block - 1)
        ])

        self.final_conv = nn.Conv2d(
            int(len_input / time_strides), num_for_predict,
            kernel_size=(1, nb_time_filter)
        )
        self.to(DEVICE)

    def forward(self, x):
        adaptive_adj = None

        if self.in_channels == 3:
            # 1. Channel Attention Gate Ablation
            if self.use_channel_attention:
                x = self.channel_attention(x)                              # (B, N, 3, T)

            # 2. Channel Fusion — always active for 3-channel data
            x = self.channel_fusion(
                x.permute(0, 2, 1, 3)                                      # (B, 3, N, T)
            ).permute(0, 2, 1, 3)                                          # (B, N, 1, T)

            # 3. Temporal Positional Embedding Ablation
            if self.use_temporal_emb:
                x = self.temporal_emb(x)                                   # (B, N, 1, T)

            # 4. Adaptive Adjacency Ablation
            if self.use_adaptive_adj:
                adaptive_adj = self.adaptive_adj_module()                  # (N, N)

        # ASTGCN blocks (adaptive_adj=None falls back to original static behaviour)
        for block in self.BlockList:
            x = block(x, adaptive_adj)

        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        return output


# =============================================================================
# Model Factory
# =============================================================================

def make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter,
               time_strides, adj_mx, num_for_predict, len_input, num_of_vertices,
               use_channel_attention=True, use_adaptive_adj=True, use_temporal_emb=True):
    
    L_tilde = scaled_Laplacian(adj_mx)
    cheb_polynomials = [
        torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE)
        for i in cheb_polynomial(L_tilde, K)
    ]
    model = ASTGCN_submodule(
        DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter,
        time_strides, cheb_polynomials, num_for_predict, len_input, num_of_vertices,
        use_channel_attention, use_adaptive_adj, use_temporal_emb
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model
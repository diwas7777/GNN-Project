from __future__ import annotations

import math
from contextlib import nullcontext
from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F


class GatedTemporalConv(nn.Module):
    """Temporal gated convolution with residual gate mixing."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2, dilation: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.filter_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            dilation=(dilation, 1),
        )
        self.gate_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            dilation=(dilation, 1),
        )
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (self.filter_conv, self.gate_conv, self.residual_conv):
            nn.init.xavier_uniform_(module.weight, gain=math.sqrt(2.0))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, N, C] -> [B, C, T, N]
        residual = x.permute(0, 3, 1, 2).contiguous()
        pad = (self.kernel_size - 1) * self.dilation
        padded = F.pad(residual, (0, 0, pad, 0)).contiguous()
        gate = torch.sigmoid(self.gate_conv(padded)).contiguous()
        filtered = torch.tanh(self.filter_conv(padded)).contiguous()
        projected = self.residual_conv(residual).contiguous()
        out = gate * filtered + (1.0 - gate) * projected
        out = F.relu(self.norm(out))
        return out.permute(0, 2, 3, 1).contiguous()


class GraphAttentionHead(nn.Module):
    """Single graph attention head over a fixed or learned adjacency."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.6, alpha: float = 0.2):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features, bias=False)
        self.attn_src = nn.Linear(out_features, 1, bias=False)
        self.attn_dst = nn.Linear(out_features, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.residual = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight, gain=math.sqrt(2.0))
        nn.init.xavier_uniform_(self.attn_src.weight, gain=math.sqrt(2.0))
        nn.init.xavier_uniform_(self.attn_dst.weight, gain=math.sqrt(2.0))
        if isinstance(self.residual, nn.Linear):
            nn.init.xavier_uniform_(self.residual.weight, gain=math.sqrt(2.0))
            nn.init.constant_(self.residual.bias, 0.1)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        autocast_context = torch.amp.autocast("cuda", enabled=False) if x.device.type == "cuda" else nullcontext()
        with autocast_context:
            x_float = x.float().contiguous()
            h = F.linear(x_float, self.proj.weight.float()).contiguous()
            src_scores = F.linear(h, self.attn_src.weight.float())
            dst_scores = F.linear(h, self.attn_dst.weight.float()).transpose(1, 2).contiguous()
            scores = self.leaky_relu(src_scores + dst_scores).contiguous()
            edge_weights = adjacency.to(device=x.device, dtype=torch.float32).contiguous().clone()
            if edge_weights.dim() != 2 or edge_weights.shape[0] != x.shape[1] or edge_weights.shape[1] != x.shape[1]:
                raise ValueError(f"adjacency shape {tuple(edge_weights.shape)} does not match node count {x.shape[1]}")
            mask = edge_weights > 0
            log_edge_weights = torch.log(edge_weights.clamp_min(1e-6)).unsqueeze(0).contiguous()
            scores = (scores + log_edge_weights).contiguous()
            scores = scores.masked_fill(~mask.unsqueeze(0), torch.finfo(scores.dtype).min)
            attention = self.dropout(torch.softmax(scores, dim=-1)).contiguous()
            attended = torch.bmm(attention, h)
            if isinstance(self.residual, nn.Linear):
                residual = F.linear(x_float, self.residual.weight.float(), self.residual.bias.float())
            else:
                residual = x_float
            output = attended + self.bias.float() + residual
        return output.to(dtype=x.dtype)


class GraphAttentionLayer(nn.Module):
    """Multi-head graph attention layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int,
        dropout: float = 0.6,
        alpha: float = 0.2,
        concat: bool = True,
    ):
        super().__init__()
        self.concat = concat
        self.attention_heads = nn.ModuleList(
            [GraphAttentionHead(in_features, out_features, dropout=dropout, alpha=alpha) for _ in range(heads)]
        )

    @property
    def output_features(self) -> int:
        if self.concat:
            return len(self.attention_heads) * self.attention_heads[0].proj.out_features
        return self.attention_heads[0].proj.out_features

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        outputs = [head(x, adjacency) for head in self.attention_heads]
        if self.concat:
            return F.elu(torch.cat(outputs, dim=-1))
        return torch.stack(outputs, dim=0).mean(dim=0)


class STBlock(nn.Module):
    """Three gated temporal convolutions followed by graph attention."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        heads: int,
        dropout: float,
        final_attention: bool = False,
    ):
        super().__init__()
        self.temporal = nn.Sequential(
            GatedTemporalConv(channels, hidden_channels, dilation=1),
            GatedTemporalConv(hidden_channels, hidden_channels, dilation=2),
            GatedTemporalConv(hidden_channels, hidden_channels, dilation=1),
        )
        self.attention = GraphAttentionLayer(
            in_features=hidden_channels,
            out_features=hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=not final_attention,
        )
        attention_channels = self.attention.output_features
        self.channel_projection = (
            nn.Linear(attention_channels, hidden_channels) if attention_channels != hidden_channels else nn.Identity()
        )
        self.residual_projection = nn.Linear(channels, hidden_channels) if channels != hidden_channels else nn.Identity()
        self.norm = nn.BatchNorm2d(hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        residual = self.residual_projection(x)
        temporal = self.temporal(x)
        bsz, steps, nodes, channels = temporal.shape
        attention_input = temporal.reshape(bsz * steps, nodes, channels)
        attended = self.attention(attention_input, adjacency)
        attended = self.channel_projection(attended).reshape(bsz, steps, nodes, -1)
        out = self.dropout(attended + residual)
        out = self.norm(out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous()
        return F.relu(out)


class STGATPath(nn.Module):
    def __init__(
        self,
        input_features: int,
        hidden_channels: int,
        attention_heads: Sequence[int],
        blocks: int,
        dropout: float,
    ):
        super().__init__()
        if len(attention_heads) != blocks:
            raise ValueError("attention_heads must contain one value per block")
        layers = []
        channels = input_features
        for index in range(blocks):
            layers.append(
                STBlock(
                    channels=channels,
                    hidden_channels=hidden_channels,
                    heads=attention_heads[index],
                    dropout=dropout,
                    final_attention=index == blocks - 1,
                )
            )
            channels = hidden_channels
        self.blocks = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.blocks:
            out = block(out, adjacency)
        return out


class GatedFusion(nn.Module):
    """Learned fusion of physical-adjacency and adaptive-adjacency paths."""

    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Linear(channels, channels)
        self.physical_proj = nn.Linear(channels, channels)
        self.adaptive_proj = nn.Linear(channels, channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (self.gate, self.physical_proj, self.adaptive_proj):
            nn.init.xavier_uniform_(module.weight, gain=math.sqrt(2.0))
            nn.init.constant_(module.bias, 0.1)

    def forward(self, physical: torch.Tensor, adaptive: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(physical + adaptive))
        return torch.tanh(self.physical_proj(physical)) * gate + (1.0 - gate) * self.adaptive_proj(adaptive)


class STGAT(nn.Module):
    """Dual-path STGAT with physical and adaptive graph paths."""

    def __init__(
        self,
        num_nodes: int,
        input_features: int = 2,
        input_steps: int = 12,
        output_steps: int = 12,
        hidden_channels: int = 64,
        attention_heads: Sequence[int] = (4, 4, 4, 6),
        blocks: int = 4,
        dropout: float = 0.6,
    ):
        super().__init__()
        if input_steps <= 0 or output_steps <= 0:
            raise ValueError("input_steps and output_steps must be positive")
        self.num_nodes = num_nodes
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.physical_path = STGATPath(input_features, hidden_channels, attention_heads, blocks, dropout)
        self.adaptive_path = STGATPath(input_features, hidden_channels, attention_heads, blocks, dropout)
        self.adaptive_adjacency = nn.Parameter(torch.ones(num_nodes, num_nodes))
        self.fusion = GatedFusion(hidden_channels)
        self.head = nn.Sequential(
            nn.Linear(input_steps * hidden_channels, 512),
            nn.ReLU(),
            nn.Linear(512, output_steps),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.adaptive_adjacency)
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=math.sqrt(2.0))
                nn.init.constant_(module.bias, 0.1)

    def forward(self, x: torch.Tensor, physical_adjacency: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        physical_adjacency = physical_adjacency.contiguous()
        if x.dim() != 4:
            raise ValueError(f"x must have shape [batch, time, nodes, features], got {tuple(x.shape)}")
        if x.shape[1] != self.input_steps:
            raise ValueError(f"expected {self.input_steps} input steps, got {x.shape[1]}")
        if x.shape[2] != self.num_nodes:
            raise ValueError(f"expected {self.num_nodes} nodes, got {x.shape[2]}")

        adaptive_adjacency = torch.softmax(F.relu(self.adaptive_adjacency), dim=-1)
        physical = self.physical_path(x, physical_adjacency)
        adaptive = self.adaptive_path(x, adaptive_adjacency)
        fused = self.fusion(physical, adaptive)
        # [B, T, N, C] -> [B, N, T*C]
        node_embeddings = fused.permute(0, 2, 1, 3).reshape(x.shape[0], self.num_nodes, -1)
        prediction = self.head(node_embeddings)
        return prediction.permute(0, 2, 1).unsqueeze(-1).contiguous()


class STGATWithAdjacency(nn.Module):
    """Bind a fixed physical adjacency so DataParallel only scatters batches."""

    def __init__(self, model: STGAT, physical_adjacency: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("physical_adjacency", physical_adjacency)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.contiguous(), self.physical_adjacency.contiguous())

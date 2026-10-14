import torch
from torch import nn, Tensor

from .norms import RMSNorm, AdaRMSNorm
from .mha import MySimpleMHA, ProjOpt


class NormOrAdaNorm(nn.Module):
    def __init__(self, hdim, adaptive: bool = False):
        super().__init__()
        self.adaptive = adaptive
        if adaptive:
            self.norm = AdaRMSNorm(hdim)
        else:
            self.norm = RMSNorm(hdim)
    
    def forward(self, x: Tensor, film: Tensor = None):
        if self.adaptive:
            return self.norm(x, film)
        else:
            return self.norm(x)


class FFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x: Tensor):
        x = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, use_adaln=False, 
                 proj_opt=ProjOpt.Q_KV, bias=False, qk_norm=False, pe_type="rope",
                 ffn_expansion: int = 4):
        super().__init__()
        self.norm1 = NormOrAdaNorm(embed_dim, adaptive=use_adaln)
        self.attn = MySimpleMHA(
            embed_dim, num_heads, dropout=dropout, proj_opt=proj_opt,
            bias=bias, qk_norm=qk_norm, pe_type=pe_type
        )
        self.norm2 = NormOrAdaNorm(embed_dim, adaptive=use_adaln)
        self.ffn = FFN(embed_dim, embed_dim * ffn_expansion)
    
    def forward(
        self, 
        query, 
        value, 
        query_pe=None, 
        value_pe=None, 
        value_mask=None, 
        film=None,
        return_attn_weights=False,
        average_attn_weights=True
    ):
        residual = query
        query, attn_weight, value_mask = self.attn(
            x=self.norm1(query),
            c=self.norm1(value),
            x_pe=query_pe,
            c_pe=value_pe,
            attn_mask=value_mask,
            return_attn_weights=return_attn_weights,
            average_attn_weights=average_attn_weights
        )
        query = query + residual

        query = query + self.ffn(self.norm2(query, film=film))
        return query, attn_weight, value_mask


class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0, use_adaln=False, 
                 proj_opt=ProjOpt.QKV, bias=False, qk_norm=False, pe_type="rope",
                 ffn_expansion: int = 4):
        super().__init__()
        self.norm1 = NormOrAdaNorm(embed_dim, adaptive=use_adaln)
        self.attn = MySimpleMHA(
            embed_dim, num_heads, dropout=dropout, proj_opt=proj_opt,
            bias=bias, qk_norm=qk_norm, pe_type=pe_type
        )
        self.norm2 = NormOrAdaNorm(embed_dim, adaptive=use_adaln)
        self.ffn = FFN(embed_dim, embed_dim * ffn_expansion)
    
    def forward(
        self, 
        query, 
        query_pe=None, 
        query_mask=None, 
        film=None,
        return_attn_weights=False,
        average_attn_weights=True
    ):
        residual = query
        qnorm = self.norm1(query)
        query, attn_weight, query_mask = self.attn(
            x=qnorm,
            c=qnorm,
            x_pe=query_pe,
            c_pe=query_pe,
            attn_mask=query_mask,
            return_attn_weights=return_attn_weights,
            average_attn_weights=average_attn_weights
        )
        query = query + residual

        query = query + self.ffn(self.norm2(query, film=film))
        return query, attn_weight, query_mask


import torch
from torch import nn, Tensor


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)
    
    def _norm(self, x: Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor):
        original_dtype = x.dtype
        # to float32 first for numeric issues
        output = self._norm(x.float()).to(original_dtype)
        return (output * self.weight) if self.elementwise_affine else output


class FiLM(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.modulation = nn.Linear(embed_dim, 2 * embed_dim, bias=True)
        nn.init.constant_(self.modulation.weight, 0)
        nn.init.constant_(self.modulation.bias, 0)

    def forward(self, x: Tensor, t: Tensor):
        """
        Arguments:
        - x: (B, L, C)
        - t: (B, C) or (B, L, C)
        """
        assert x.dim() == 3
        assert t.dim() in (2, 3)
        if t.dim() == 2:
            t = t.unsqueeze(1)
        assert t.shape[1] == 1 or t.shape[1] == x.shape[1]
        scale, shift = torch.chunk(self.modulation(t), 2, dim=-1)
        x = x * (1 + scale) + shift
        return x


class AdaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(dim, eps, elementwise_affine=False)
        self.film = FiLM(dim)
    
    def forward(self, x: Tensor, t: Tensor):
        return self.film(self.norm(x), t)


if __name__ == "__main__":
    to_kwargs = {
        "device": "cuda:0",
        "dtype": torch.bfloat16
    }

    rms_torch = nn.RMSNorm(16).to(**to_kwargs)
    rms_mine = RMSNorm(16).to(**to_kwargs)

    x = torch.randn(16).to(**to_kwargs)
    print(x)

    y1 = rms_torch(x)
    y2 = rms_mine(x)

    delta = y1 - y2
    print(delta.abs().max())
    print(delta.abs().mean())



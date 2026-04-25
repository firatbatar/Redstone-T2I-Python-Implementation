import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
                    d_in = cfg["emb_dim"],
                    d_out = cfg["emb_dim"],
                    context_length = cfg["context_length"],
                    num_heads = cfg["n_heads"],
                    dropout = cfg["drop_rate"],
                    qkv_bias = cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert(d_out % num_heads == 0), \
                "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
                "mask",
                torch.triu(torch.ones(context_length, context_length), diagonal=1)
                )
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape # [batch, input_size, embedding_dim]
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
       
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def quantize(self):
        with torch.no_grad():
            scale_factor = (self.scale.abs().max().clamp(min=1e-8) / 127).reshape(1)
            scale_q = (self.scale.data / scale_factor).round().clamp(-128, 127).to(torch.int8)
            shift_factor = (self.shift.abs().max().clamp(min=1e-8) / 127).reshape(1)
            shift_q = (self.shift.data / shift_factor).round().clamp(-128, 127).to(torch.int8)
        del self._parameters['scale']
        del self._parameters['shift']
        self.register_buffer('scale', scale_q)
        self.register_buffer('shift', shift_q)
        self.register_buffer('scale_factor', scale_factor)
        self.register_buffer('shift_factor', shift_factor)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        if self.scale.dtype == torch.int8:
            scale = self.scale.to(x.dtype) * self.scale_factor.to(x.dtype)
            shift = self.shift.to(x.dtype) * self.shift_factor.to(x.dtype)
        else:
            scale = self.scale
            shift = self.shift
        return scale * norm_x + shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
                GELU(),
                nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
                )

    def forward(self, x):
        return self.layers(x)

__all__ = ["TransformerBlock", "MultiHeadAttention", "LayerNorm", "GELU", "FeedForward"]
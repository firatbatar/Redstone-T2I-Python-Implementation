# Initialize the tokenizer
import torch
import torch.nn as nn

from tokenizer import MinecraftTokenizer
from dataset_loader import MinecraftDataloader
from transformerblock import TransformerBlock, LayerNorm

with open("vocab.txt", "r", encoding="utf-8") as f:
        words = f.read()
all_words = words.split('\n')[:-1]
vocab = {token:integer for integer, token in enumerate(all_words)}
tokenizer = MinecraftTokenizer(vocab)

"""
dataloader = MinecraftDataloader(raw_text, 
                                 tokenizer, 
                                 batch_size=4, 
                                 max_length=4, 
                                 stride=2, 
                                 shuffle=False)

# dataloader test code
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
"""

# under construction
cfg = {
    "vocab_size": 362,
    "context_length": 1026,
    "emb_dim": 256,      # can test 128,256,512. increase leads to overfit.
    "n_heads": 8,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

class MinecraftGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
                *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]) 
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.output_layer = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size = seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
                torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

torch.manual_seed(123)
model = MinecraftGPT(cfg)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

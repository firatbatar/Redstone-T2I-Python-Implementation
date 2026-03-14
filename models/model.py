# Initialize the tokenizer
import numpy as np
import torch
import torch.nn as nn

from .tokenizer import MinecraftTokenizer
from .dataset_loader import MinecraftDataloader
from .quickdraw_manager import QuickdrawManager
from .transformerblock import TransformerBlock, LayerNorm

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
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
                torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.output_layer(x)
        return logits
    
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

     

"""
Below is code for testing functionality of model.
"""
def _main():
    cfg = {
        "vocab_size": 347,   # 345 classes + black and white bits
        "context_length": 256,
        "emb_dim": 256,      # can test 128,256,512. increase leads to overfit.
        "n_heads": 8,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }
   
    # Load dataset and initialize tokenizer
    with open("models/vocab.txt", "r", encoding="utf-8") as f:
        words = f.read()
    all_words = words.split('\n')[:-1]
    vocab = {token:integer for integer, token in enumerate(all_words)}
    tokenizer = MinecraftTokenizer(vocab)

    torch.manual_seed(123)
    model = MinecraftGPT(cfg)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")


    """
    Training Loop 
    """
    manager = QuickdrawManager()
    train_data = manager.sample_images(n=50000, seed=42)
    val_data = manager.sample_images(n=5000, seed=123)

    train_loader = MinecraftDataloader(
        train_data, tokenizer,
        batch_size=4, max_length=cfg["context_length"], stride=cfg["context_length"],
        drop_last=True, shuffle=True, num_workers=0
    )
    val_loader = MinecraftDataloader(
        val_data, tokenizer,
        batch_size=4, max_length=cfg["context_length"], stride=cfg["context_length"],
        drop_last=False, shuffle=False, num_workers=0
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    print("Train loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)

    print("\nValidation loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)

__all__ = ["MinecraftGPT", "_main"]

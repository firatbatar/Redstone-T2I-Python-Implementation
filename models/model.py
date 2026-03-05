# Initialize the tokenizer
import torch
import torch.nn as nn

from tokenizer import MinecraftTokenizer
from dataset_loader import MinecraftDataloader
from transformerblock import MinecraftTransformer

with open("vocab.txt", "r", encoding="utf-8") as f:
        words = f.read()
all_words = words.split('\n')[:-1]
vocab = {token:integer for integer, token in enumerate(all_words)}
tokenizer = MinecraftTokenizer(vocab)

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



# Embedding Layer
vocab_size = len(vocab) + 16 + 1
output_dim = 256        # can test 128,256,512. increase leads to overfit.

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

context_length = 1026 
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))

input_embeddings = token_embeddings + pos_embeddings






# under construction
MC_GPT_CONFIG = {
    "vocab_size": 362,
    "context_length": 1026,
    "emb_dim": 256,      # can test 128,256,512. increase leads to overfit.
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

class Minecraft_GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
                *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]) 
        #  ...

    def forward(self, in_idx):
        # under construction

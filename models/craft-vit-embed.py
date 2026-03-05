"""
CraftViT-Embed — Text-to-Image Transformer (no text encoder)
=============================================================
Identical to craft-vit.py except the text encoder is replaced by a plain
nn.Embedding lookup. Text tokens are projected directly into the decoder's
cross-attention with no transformer processing over the text sequence.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CRAFT_VIT_CONFIG = {
    "vocab_size":       364,    # vocabulary consists of 345 words, a separator '|', [PAD], [POS] and the possible pixel values (0-15)
    "num_bins":         16,     # distinct pixel values (2 = binary black/white)
    "img_size":         32,     # image is img_size x img_size pixels
    "text_max_len":     32,     # max number of text tokens
    "emb_dim":          256,    # embedding dimension for all layers
    "n_heads":          8,      # attention heads (emb_dim must be divisible)
    "n_decoder_layers": 4,
    "drop_rate":        0.1,
    "qkv_bias":         False,
}

# Derived fields — computed once here rather than scattered through the model
CRAFT_VIT_CONFIG["img_seq_len"] = CRAFT_VIT_CONFIG["img_size"] ** 2   # 1024
CRAFT_VIT_CONFIG["bos_token"]   = CRAFT_VIT_CONFIG["num_bins"]        # index just past pixel values


# ---------------------------------------------------------------------------
# Building Blocks  (identical style to llm-from-scratch)
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps   = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean   = x.mean(dim=-1, keepdim=True)
        var    = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
            nn.Dropout(cfg["drop_rate"]),
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    """
    Self-attention or cross-attention, causal-masking optional.

    - For self-attention:    call forward(x)        — Q, K, V all from x
    - For cross-attention:   call forward(x, kv=c)  — Q from x, K/V from c
    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False, causal=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out     = d_out
        self.num_heads = num_heads
        self.head_dim  = d_out // num_heads
        self.causal    = causal

        self.W_query  = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key    = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value  = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout  = nn.Dropout(dropout)

        if causal:
            self.register_buffer(
                "mask",
                torch.triu(torch.ones(context_length, context_length), diagonal=1)
            )

    def forward(self, x, kv=None):
        b, num_tokens, _ = x.shape
        src = kv if kv is not None else x    # cross-attention: K/V come from src
        kv_len = src.shape[1]

        queries = self.W_query(x)
        keys    = self.W_key(src)
        values  = self.W_value(src)

        # Split into heads
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys    = keys.view(b, kv_len,     self.num_heads, self.head_dim).transpose(1, 2)
        values  = values.view(b, kv_len,   self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        # Apply causal mask only during self-attention (kv is None)
        if self.causal and kv is None:
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
            attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)


# ---------------------------------------------------------------------------
# Decoder Block  (causal self-attention + cross-attention to text embeddings)
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        img_ctx = cfg["img_seq_len"] + 1   # +1 accounts for the BOS token

        # Step 1 — causal self-attention over image tokens generated so far
        self.self_att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=img_ctx,
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"],
            causal=True,
        )
        # Step 2 — cross-attention: image tokens (Q) attend to text embeddings (K, V)
        self.cross_att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=img_ctx,
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"],
            causal=False,
        )
        self.ff    = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.norm3 = LayerNorm(cfg["emb_dim"])
        self.drop  = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, context):
        # Causal self-attention
        shortcut = x
        x = self.norm1(x)
        x = self.self_att(x)
        x = self.drop(x) + shortcut

        # Cross-attention to text
        shortcut = x
        x = self.norm2(x)
        x = self.cross_att(x, kv=context)
        x = self.drop(x) + shortcut

        # Feed-forward
        shortcut = x
        x = self.norm3(x)
        x = self.ff(x)
        return self.drop(x) + shortcut


# ---------------------------------------------------------------------------
# Image Decoder
# ---------------------------------------------------------------------------

class ImageDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["num_bins"] + 1, cfg["emb_dim"])  # +1 for BOS
        self.pos_emb = nn.Embedding(cfg["img_seq_len"] + 1, cfg["emb_dim"])
        self.drop    = nn.Dropout(cfg["drop_rate"])
        self.blocks  = nn.ModuleList(
            [DecoderBlock(cfg) for _ in range(cfg["n_decoder_layers"])]
        )
        self.norm     = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["num_bins"])

    def forward(self, pixel_ids, context):
        _, seq_len = pixel_ids.shape
        x = self.tok_emb(pixel_ids) + self.pos_emb(
            torch.arange(seq_len, device=pixel_ids.device)
        )
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, context)
        x = self.norm(x)
        return self.out_head(x)   # (B, seq_len, num_bins)


# ---------------------------------------------------------------------------
# Full CraftViT-Embed Model
# ---------------------------------------------------------------------------

class CraftViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg        = cfg
        self.text_embed = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.image_decoder = ImageDecoder(cfg)

    def forward(self, text_ids, pixel_ids):
        context = self.text_embed(text_ids)               # (B, text_len, emb_dim)
        logits  = self.image_decoder(pixel_ids, context)  # (B, img_seq, num_bins)
        return logits

    @torch.no_grad()
    def generate(self, text_ids, device, temperature=1.0):
        """Autoregressively generate a flat pixel sequence from a text prompt."""
        self.eval()
        cfg     = self.cfg
        bos     = cfg["bos_token"]
        context = self.text_embed(text_ids.to(device))

        # Start with only the BOS token
        pixel_ids = torch.full((text_ids.shape[0], 1), bos, dtype=torch.long, device=device)

        for _ in range(cfg["img_seq_len"]):
            logits      = self.image_decoder(pixel_ids, context)
            next_logits = logits[:, -1, :] / temperature          # (B, num_bins)
            probs       = torch.softmax(next_logits, dim=-1)
            next_token  = torch.multinomial(probs, num_samples=1)  # (B, 1)
            pixel_ids   = torch.cat([pixel_ids, next_token], dim=1)

        return pixel_ids[:, 1:]   # drop BOS → (B, img_seq_len)


# ---------------------------------------------------------------------------
# Tokenizer  (word-level, no external dependencies)
# ---------------------------------------------------------------------------

class SimpleTokenizer:
    def __init__(self):
        self.word_to_idx = {"<pad>": 0, "<unk>": 1}
        self.idx_to_word = {0: "<pad>", 1: "<unk>"}

    def build_vocab(self, texts):
        for text in texts:
            for word in text.lower().split():
                if word not in self.word_to_idx:
                    idx = len(self.word_to_idx)
                    self.word_to_idx[word] = idx
                    self.idx_to_word[idx]  = word

    def encode(self, text, max_len):
        tokens  = [self.word_to_idx.get(w, 1) for w in text.lower().split()]
        tokens  = tokens[:max_len]
        tokens += [0] * (max_len - len(tokens))   # right-pad with <pad>
        return tokens

    @property
    def vocab_size(self):
        return len(self.word_to_idx)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TextImageDataset(Dataset):
    """
    texts:  list[str]         — one caption per image
    images: list[LongTensor]  — flat pixel sequences of length img_seq_len,
                                values in [0, num_bins)
    """

    def __init__(self, texts, images, tokenizer, cfg):
        bos = cfg["bos_token"]
        self.text_ids  = [
            torch.tensor(tokenizer.encode(t, cfg["text_max_len"]), dtype=torch.long)
            for t in texts
        ]
        # Prepend BOS to each image sequence
        self.pixel_ids = [
            torch.cat([torch.tensor([bos]), img.long()])
            for img in images
        ]

    def __len__(self):
        return len(self.text_ids)

    def __getitem__(self, idx):
        # Teacher-forcing: input is BOS+pixels[:-1], target is pixels
        pixel = self.pixel_ids[idx]
        return self.text_ids[idx], pixel[:-1], pixel[1:]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model, dataloader, optimizer, device, num_epochs=10):
    model.train()
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for text_ids, pixel_in, pixel_target in dataloader:
            text_ids     = text_ids.to(device)
            pixel_in     = pixel_in.to(device)
            pixel_target = pixel_target.to(device)

            logits = model(text_ids, pixel_in)              # (B, seq_len, num_bins)
            loss   = loss_fn(
                logits.view(-1, model.cfg["num_bins"]),
                pixel_target.view(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}  loss={total_loss / len(dataloader):.4f}")


# ---------------------------------------------------------------------------
# Entry point / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torchvision
    import torchvision.transforms as transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = CRAFT_VIT_CONFIG.copy()

    # --- Load CIFAR-100, convert to binary 32x32 ---
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    raw = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)

    texts  = [f"a photo of a {raw.classes[label]}" for _, label in raw]
    images = [(img > 0.5).long().view(-1) for img, _ in raw]  # binarize → flat

    # --- Build vocabulary ---
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(texts)
    cfg["vocab_size"] = tokenizer.vocab_size
    print(f"Vocabulary size: {cfg['vocab_size']}")

    # --- Dataset / DataLoader ---
    dataset    = TextImageDataset(texts, images, tokenizer, cfg)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    # --- Model ---
    model = CraftViT(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"CraftViT-Embed parameters: {total_params:,}")

    # --- Train ---
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(model, dataloader, optimizer, device, num_epochs=5)

    # --- Generate a sample ---
    model.eval()
    prompt     = tokenizer.encode("a photo of a cat", cfg["text_max_len"])
    text_ids   = torch.tensor([prompt], dtype=torch.long)
    pixels     = model.generate(text_ids, device)
    print(f"Generated shape: {pixels.shape}")          # (1, 1024)
    print(f"Unique values:   {pixels.unique().tolist()}")

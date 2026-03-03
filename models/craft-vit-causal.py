"""
CraftViT-Causal — Unified Sequence Text-to-Image Transformer
=============================================================
Text tokens and pixel tokens are concatenated into a single sequence.
A single causal mask lets every token attend to everything before it —
words see earlier words, pixels see all words and all earlier pixels.

No encoder/decoder split. No cross-attention. Just a GPT stack.

  [t0, t1, ..., t_T,  BOS, p0, p1, ..., p_{N-1}]
   ─── text tokens ───  ─────── pixel tokens ──────
   ◄──────────── single causal transformer ────────►

Pixel IDs are offset into the shared vocabulary space so a single
nn.Embedding covers both token types.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CRAFT_VIT_CONFIG = {
    "vocab_size":   512,    # text vocabulary (built from training data)
    "num_bins":     2,      # distinct pixel values (2 = binary black/white)
    "img_size":     32,     # image is img_size x img_size pixels
    "text_max_len": 32,     # max number of text tokens
    "emb_dim":      256,    # embedding dimension for all layers
    "n_heads":      8,      # attention heads (emb_dim must be divisible)
    "n_layers":     8,      # single transformer stack (replaces separate enc + dec)
    "drop_rate":    0.1,
    "qkv_bias":     False,
}

# Derived fields
CRAFT_VIT_CONFIG["img_seq_len"] = CRAFT_VIT_CONFIG["img_size"] ** 2           # 1024
CRAFT_VIT_CONFIG["bos_token"]   = CRAFT_VIT_CONFIG["num_bins"]                 # 2  (in pixel-local space)
CRAFT_VIT_CONFIG["total_vocab"] = CRAFT_VIT_CONFIG["vocab_size"] \
                                 + CRAFT_VIT_CONFIG["num_bins"] + 1            # text + pixels + BOS
CRAFT_VIT_CONFIG["max_seq_len"] = CRAFT_VIT_CONFIG["text_max_len"] \
                                 + CRAFT_VIT_CONFIG["img_seq_len"] + 1         # 1057


# ---------------------------------------------------------------------------
# Building Blocks  (identical to llm-from-scratch)
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
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out     = d_out
        self.num_heads = num_heads
        self.head_dim  = d_out // num_heads

        self.W_query  = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key    = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value  = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout  = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, _ = x.shape

        queries = self.W_query(x)
        keys    = self.W_key(x)
        values  = self.W_value(x)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys    = keys.view(b, num_tokens,   self.num_heads, self.head_dim).transpose(1, 2)
        values  = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)


# ---------------------------------------------------------------------------
# Transformer Block  (identical to llm-from-scratch's TransformerBlock)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["max_seq_len"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff         = FeedForward(cfg)
        self.norm1      = LayerNorm(cfg["emb_dim"])
        self.norm2      = LayerNorm(cfg["emb_dim"])
        self.drop       = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop(x) + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        return self.drop(x) + shortcut


# ---------------------------------------------------------------------------
# Full CraftViT-Causal Model
# ---------------------------------------------------------------------------

class CraftViT(nn.Module):
    """
    Single causal transformer over [text tokens | pixel tokens].

    Pixel token IDs are offset by vocab_size so both token types share
    one embedding table. The output head only predicts pixel values (num_bins).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg      = cfg
        self.tok_emb  = nn.Embedding(cfg["total_vocab"], cfg["emb_dim"])
        self.pos_emb  = nn.Embedding(cfg["max_seq_len"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.blocks   = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head   = nn.Linear(cfg["emb_dim"], cfg["num_bins"])

    def _embed(self, text_ids, pixel_ids):
        """Offset pixel IDs and concatenate with text IDs into one sequence."""
        pixel_ids_offset = pixel_ids + self.cfg["vocab_size"]
        full_ids = torch.cat([text_ids, pixel_ids_offset], dim=1)
        _, seq_len = full_ids.shape
        x = self.tok_emb(full_ids) + self.pos_emb(
            torch.arange(seq_len, device=full_ids.device)
        )
        return self.drop_emb(x)

    def forward(self, text_ids, pixel_ids):
        """
        text_ids:  (B, T)   — text token indices  [0, vocab_size)
        pixel_ids: (B, N)   — pixel token indices  [0, num_bins]  (num_bins = BOS)

        Returns logits (B, N, num_bins) for the pixel positions only.
        """
        x = self._embed(text_ids, pixel_ids)          # (B, T+N, emb_dim)
        x = self.blocks(x)
        x = self.final_norm(x)
        # Logits for pixel positions: positions T onward predict the next pixel
        T = text_ids.shape[1]
        return self.out_head(x[:, T:, :])             # (B, N, num_bins)

    @torch.no_grad()
    def generate(self, text_ids, device, temperature=1.0):
        """Autoregressively generate a flat pixel sequence from a text prompt."""
        self.eval()
        cfg = self.cfg
        bos = cfg["bos_token"]

        text_ids  = text_ids.to(device)
        pixel_ids = torch.full((text_ids.shape[0], 1), bos, dtype=torch.long, device=device)

        for _ in range(cfg["img_seq_len"]):
            logits      = self.forward(text_ids, pixel_ids)  # (B, cur_pix_len, num_bins)
            next_logits = logits[:, -1, :] / temperature
            probs       = torch.softmax(next_logits, dim=-1)
            next_token  = torch.multinomial(probs, num_samples=1)   # (B, 1)
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
        tokens += [0] * (max_len - len(tokens))
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
        # Prepend BOS to each image sequence (pixel-local IDs, offset happens in model)
        self.pixel_ids = [
            torch.cat([torch.tensor([bos]), img.long()])
            for img in images
        ]

    def __len__(self):
        return len(self.text_ids)

    def __getitem__(self, idx):
        # Teacher-forcing: input is BOS+pixels[:-1], target is pixels (without BOS)
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

            # logits shape: (B, N, num_bins) — pixel positions only
            logits = model(text_ids, pixel_in)
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
    images = [(img > 0.5).long().view(-1) for img, _ in raw]

    # --- Build vocabulary ---
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(texts)
    cfg["vocab_size"]   = tokenizer.vocab_size
    cfg["total_vocab"]  = cfg["vocab_size"] + cfg["num_bins"] + 1
    print(f"Vocabulary size: {cfg['vocab_size']}  |  Total token space: {cfg['total_vocab']}")

    # --- Dataset / DataLoader ---
    dataset    = TextImageDataset(texts, images, tokenizer, cfg)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    # --- Model ---
    model = CraftViT(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"CraftViT-Causal parameters: {total_params:,}")

    # --- Train ---
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(model, dataloader, optimizer, device, num_epochs=5)

    # --- Generate a sample ---
    model.eval()
    prompt   = tokenizer.encode("a photo of a cat", cfg["text_max_len"])
    text_ids = torch.tensor([prompt], dtype=torch.long)
    pixels   = model.generate(text_ids, device)
    print(f"Generated shape: {pixels.shape}")
    print(f"Unique values:   {pixels.unique().tolist()}")

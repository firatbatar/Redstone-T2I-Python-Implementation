"""
RedstoneTransformer - Text-to-Image (32x32 binary, CIFAR-100)
A small encoder-decoder transformer designed for eventual Minecraft implementation.
Target: ~2-5M parameters.
"""

import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms

# ============================================================
# Config
# ============================================================

IMG_SIZE = 32
NUM_PIXELS = IMG_SIZE * IMG_SIZE  # 1024
MAX_TEXT_LEN = 32
NUM_BINS = 2  # binary: 0 (black), 1 (white)
BOS_TOKEN = 2  # decoder start token (outside 0/1 pixel range)
NUM_PIXEL_TOKENS = 3  # 0, 1, BOS

# ============================================================
# Tokenizer
# ============================================================

class SimpleTokenizer:
    """Word-level tokenizer. Minimal and easy to reimplement without libraries."""

    def __init__(self):
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.idx2word = {0: "<pad>", 1: "<unk>"}
        self.pad_id = 0
        self.unk_id = 1

    def build_vocab(self, texts):
        for text in texts:
            for word in text.lower().split():
                if word not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word

    @property
    def vocab_size(self):
        return len(self.word2idx)

    def encode(self, text, max_len=MAX_TEXT_LEN):
        tokens = [self.word2idx.get(w, self.unk_id) for w in text.lower().split()]
        tokens = tokens[:max_len]
        tokens += [self.pad_id] * (max_len - len(tokens))
        return tokens

    def save(self, path):
        with open(path, "w") as f:
            for word, idx in sorted(self.word2idx.items(), key=lambda x: x[1]):
                f.write(f"{word}\t{idx}\n")

    def load(self, path):
        self.word2idx = {}
        self.idx2word = {}
        with open(path) as f:
            for line in f:
                word, idx = line.strip().split("\t")
                idx = int(idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        self.pad_id = self.word2idx["<pad>"]
        self.unk_id = self.word2idx["<unk>"]

# ============================================================
# Dataset
# ============================================================

class TextImageDataset(Dataset):
    def __init__(self, prompts, pixels):
        self.prompts = prompts  # list of token lists
        self.pixels = pixels    # list of np arrays (1024,)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.prompts[idx], dtype=torch.long),
            torch.tensor(self.pixels[idx], dtype=torch.long),
        )


def load_cifar100():
    """Load CIFAR-100 as grayscale, threshold to binary pixels, and generate text prompts."""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform,
    )
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform,
    )
    classes = trainset.classes

    def extract(dataset, desc):
        prompts = []
        pixels = []
        for img_tensor, label in tqdm(dataset, desc=desc):
            # img_tensor is (1, 32, 32) in [0, 1]
            arr = img_tensor.squeeze(0).numpy()  # (32, 32)
            binary = (arr >= 0.5).astype(np.uint8).flatten()  # threshold to 0/1
            label_name = classes[label].replace('_', ' ')
            prompts.append(f"a photo of a {label_name}")
            pixels.append(binary)
        return prompts, pixels

    train_prompts, train_pixels = extract(trainset, "Loading CIFAR-100 train")
    test_prompts, test_pixels = extract(testset, "Loading CIFAR-100 test")

    return train_prompts, train_pixels, test_prompts, test_pixels

# ============================================================
# Model
# ============================================================

def sinusoidal_positional_encoding(max_len, d_model):
    """Precompute sinusoidal positional encodings.

    No learnable parameters - easy to reimplement in Minecraft.
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (max_len, d_model)


class RedstoneTransformer(nn.Module):
    """Encoder-decoder transformer for text-to-image generation.

    Architecture kept intentionally simple for Minecraft portability:
    - Sinusoidal positional encoding (no learned params)
    - Standard multi-head attention
    - Simple feedforward blocks
    """

    def __init__(
        self,
        vocab_size,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=4,
        dim_feedforward=512,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # --- Embeddings ---
        self.text_emb = nn.Embedding(vocab_size, d_model)
        self.pixel_emb = nn.Embedding(NUM_PIXEL_TOKENS, d_model)  # 0, 1, BOS

        # --- Positional Encoding (fixed, not learned) ---
        self.register_buffer("text_pe", sinusoidal_positional_encoding(MAX_TEXT_LEN, d_model))
        self.register_buffer("pixel_pe", sinusoidal_positional_encoding(NUM_PIXELS + 1, d_model))

        # --- Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # --- Decoder ---
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # --- Output head ---
        self.output_proj = nn.Linear(d_model, NUM_BINS)  # predict 0 or 1

    def forward(self, text_tokens, pixel_input):
        """
        text_tokens: (B, T_text)  - tokenized prompt
        pixel_input: (B, T_pix)   - pixel sequence with BOS prepended
        Returns: (B, T_pix, NUM_BINS) logits
        """
        B, Tt = text_tokens.shape
        _, Tp = pixel_input.shape

        # Encode text
        src = self.text_emb(text_tokens) + self.text_pe[:Tt]
        memory = self.encoder(src)

        # Decode pixels
        tgt = self.pixel_emb(pixel_input) + self.pixel_pe[:Tp]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(Tp, device=tgt.device)
        out = self.decoder(tgt, memory, tgt_mask=causal_mask)

        return self.output_proj(out)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ============================================================
# Training
# ============================================================

def train_model(model, train_loader, val_loader, epochs, device, lr=1e-3):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]")
        for text, pixels in pbar:
            text, pixels = text.to(device), pixels.to(device)

            # Build decoder input: [BOS, p0, p1, ..., p_{N-2}]
            bos = torch.full((pixels.size(0), 1), BOS_TOKEN, dtype=torch.long, device=device)
            dec_input = torch.cat([bos, pixels[:, :-1]], dim=1)

            optimizer.zero_grad()
            logits = model(text, dec_input)  # (B, 4096, 2)
            loss = criterion(logits.transpose(1, 2), pixels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train = train_loss / len(train_loader)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for text, pixels in val_loader:
                text, pixels = text.to(device), pixels.to(device)
                bos = torch.full((pixels.size(0), 1), BOS_TOKEN, dtype=torch.long, device=device)
                dec_input = torch.cat([bos, pixels[:, :-1]], dim=1)
                logits = model(text, dec_input)
                val_loss += criterion(logits.transpose(1, 2), pixels).item()

        avg_val = val_loss / max(len(val_loader), 1)
        print(f"Epoch {epoch}: train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

        # Save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "best_model.pt")
            print(f"  -> Saved best model (val_loss={avg_val:.4f})")

    return model

# ============================================================
# Inference
# ============================================================

@torch.no_grad()
def generate_image(model, prompt, tokenizer, device="cpu", temperature=0.8):
    """Generate a 32x32 binary image from a text prompt, pixel by pixel."""
    model.eval()
    model.to(device)

    # Tokenize prompt
    tokens = tokenizer.encode(prompt)
    text_tensor = torch.tensor([tokens], dtype=torch.long, device=device)

    # Start with BOS
    generated = [BOS_TOKEN]

    for _ in tqdm(range(NUM_PIXELS), desc="Generating", leave=True):
        dec_input = torch.tensor([generated], dtype=torch.long, device=device)
        logits = model(text_tensor, dec_input)
        last_logits = logits[0, -1, :NUM_BINS]  # only 0/1 logits

        if temperature <= 0:
            next_pixel = last_logits.argmax().item()
        else:
            probs = torch.softmax(last_logits / temperature, dim=0)
            next_pixel = torch.multinomial(probs, 1).item()

        generated.append(next_pixel)

    pixels = np.array(generated[1:], dtype=np.uint8).reshape(IMG_SIZE, IMG_SIZE)
    return pixels


def display_image(img, title="Generated"):
    plt.figure(figsize=(3, 3))
    plt.imshow(img, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.title(title)
    plt.show()

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="RedstoneTransformer T2I")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--enc_layers", type=int, default=2)
    parser.add_argument("--dec_layers", type=int, default=4)
    parser.add_argument("--dim_ff", type=int, default=512)
    parser.add_argument("--generate", type=str, default=None, help="Prompt to generate (skip training)")
    parser.add_argument("--load", type=str, default=None, help="Path to saved model weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load CIFAR-100 ---
    train_prompts, train_pixels, test_prompts, test_pixels = load_cifar100()
    all_prompts = train_prompts + test_prompts
    print(f"Loaded {len(train_prompts)} train, {len(test_prompts)} test samples")

    # --- Tokenizer ---
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(all_prompts)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # --- Build model ---
    model = RedstoneTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_ff,
    )
    print(f"Model parameters: {model.count_parameters():,}")

    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=device))
        print(f"Loaded weights from {args.load}")

    # --- Generate mode ---
    if args.generate:
        vocab_path = "vocab.txt"
        if os.path.exists(vocab_path):
            tokenizer.load(vocab_path)
        img = generate_image(model, args.generate, tokenizer, device=device)
        display_image(img, title=args.generate)
        return

    # --- Tokenize ---
    train_tokens = [tokenizer.encode(p) for p in tqdm(train_prompts, desc="Tokenizing train")]
    val_tokens = [tokenizer.encode(p) for p in tqdm(test_prompts, desc="Tokenizing test")]

    train_ds = TextImageDataset(train_tokens, train_pixels)
    val_ds = TextImageDataset(val_tokens, test_pixels)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # --- Train ---
    model = train_model(model, train_loader, val_loader, args.epochs, device, args.lr)

    # --- Save tokenizer ---
    tokenizer.save("vocab.txt")

    # --- Generate a sample ---
    test_prompt = test_prompts[0]
    print(f"\nGenerating for: '{test_prompt}'")
    img = generate_image(model, test_prompt, tokenizer, device=device)
    display_image(img, title=test_prompt)

    # Show ground truth for comparison
    gt = test_pixels[0].reshape(IMG_SIZE, IMG_SIZE)
    display_image(gt, title="Ground Truth")


if __name__ == "__main__":
    main()

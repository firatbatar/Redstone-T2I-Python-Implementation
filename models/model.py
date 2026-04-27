from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import json

# For visualization of generated images
import matplotlib.pyplot as plt
import numpy as np

from .tokenizer import MinecraftTokenizer
from .dataset_loader import MinecraftDataloader
from .quickdraw_manager import QuickdrawManager
from .transformerblock import TransformerBlock, LayerNorm

CONFIG_PATH = Path(__file__).parent / "config.json"

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


def generate_image(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]       # idx is (batch, n_tokens)
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]               # (batch, n_tokens, vocab_size) -> (batch, vocab_size)
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)       # (batch, 1)
        idx = torch.cat((idx, idx_next), dim=1) # append sampled index to running sequence. idx has shape (batch, n_tokens+1)

    return idx

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # New (not in book): numerical stability tip to get equivalent results on mps device
            # subtract rowwise max before softmax
            logits = logits - logits.max(dim=-1, keepdim=True).values
            
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)
 
        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    # (batch, n_tokens, vocab_size) -> (batch*n_tokens, vocab_size), (batch_size, n_tokens) -> (batch_size, n_tokens)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_image(model, tokenizer, device, word):
    model.eval()
    # batch size will be 1 for printing since it is a single word only.
    context_size = model.pos_emb.weight.shape[0]
    word_id = tokenizer.word_to_id[word]
    encoded = torch.tensor([[word_id]], device=device)
    with torch.no_grad():
        token_ids = generate(
            model=model,
            idx=encoded,
            max_new_tokens=context_size-1,
            context_size=context_size,
            top_k=2,    # Only choose between black and white
            temperature=1.2
        )

    pixels = tokenizer.decode_pixels(token_ids.squeeze(0).cpu())  # squeeze out batch dimension.

    side = int(len(pixels) ** 0.5)
    grid = np.array(pixels, dtype=np.uint8).reshape(side, side)
    img = 1 - grid  # invert: pixel=1 → black (0), background=0 → white (1)

    import subprocess
    from datetime import datetime

    out_dir = Path(__file__).parent.parent / "generated"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{word}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(word)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    subprocess.Popen(["xdg-open", str(path)])
    model.train()


def train_model(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_word, tokenizer):
    
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_image(model, tokenizer, device, start_word)

    return train_losses, val_losses, track_tokens_seen


def _main():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")        
    cfg = json.loads(CONFIG_PATH.read_text())


    # Load dataset and initialize tokenizer
    vocab_file = Path(__file__).parent / "vocab.txt"
    if not vocab_file.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_file}")
    words = vocab_file.read_text().split('\n')[:-1]
    tokenizer = MinecraftTokenizer(words)


    # Set up device (GPU if available, otherwise CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Use PyTorch 2.9 or newer for stable mps results
        major, minor = map(int, torch.__version__.split(".")[:2])
        if (major, minor) >= (2, 9):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print(f"Using {device} device.")


    # Set random seed for reproducibility and initialize model
    torch.manual_seed(123)
    model = MinecraftGPT(cfg)
    model.to(device)
    # Print total number of parameters in the model
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")


    # Initialize dataset and data loaders
    manager = QuickdrawManager()
    train_data = manager.sample_images(n=345000, seed=42)
    val_data = manager.sample_images(n=34500, seed=123)

    train_loader = MinecraftDataloader(
        train_data, tokenizer,
        batch_size=64, max_length=cfg["context_length"] - 1,
        drop_last=True, shuffle=True, num_workers=0
    )
    val_loader = MinecraftDataloader(
        val_data, tokenizer,
        batch_size=64, max_length=cfg["context_length"] - 1,
        drop_last=False, shuffle=False, num_workers=0
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    torch.manual_seed(123)  # For reproducibility due to the shuffling in the data loader

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)
    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)


    # Main training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00175, weight_decay=0.1)
    num_epochs = 3
    train_losses, val_losses, tokens_seen = train_model(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=100, eval_iter=20,
        start_word="apple", tokenizer=tokenizer
    )

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        },
        "model_and_optimizer.pth"
        )


__all__ = ["MinecraftGPT", "_main"]

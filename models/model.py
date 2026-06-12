from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import json

# For visualization of generated images
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from datetime import datetime


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
        _, seq_len = in_idx.shape
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


def generate_and_print_image(model, tokenizer, device, word,
                             temperatures=(1.0,),
                             top_ks=(4,)):
    """
    Generates an image for the given word using different temperature and top_k settings,
    and saves the resulting grid of images to a file.

    The generated image grid is also displayed using the default image viewer and in Jupyter notebooks (if available).
    """
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    word_id = tokenizer.word_to_id[word]

    fig, axes = plt.subplots(len(top_ks), len(temperatures),
                             figsize=(3 * len(temperatures), 3 * len(top_ks)))
    axes = np.array(axes).reshape(len(top_ks), len(temperatures))
    for row, top_k in enumerate(top_ks):
        for col, temp in enumerate(temperatures):
            encoded = torch.tensor([[word_id]], device=device)
            with torch.no_grad():
                token_ids = generate(
                    model=model,
                    idx=encoded,
                    max_new_tokens=context_size - 1,
                    context_size=context_size,
                    top_k=top_k,
                    temperature=temp,
                )
            pixels = tokenizer.decode(token_ids.squeeze(0).cpu())
            grid = np.array(pixels, dtype=np.uint8).reshape(QuickdrawManager.IMG_SIZE, QuickdrawManager.IMG_SIZE)
            img = 1 - grid  # invert: pixel=1 → black (0), background=0 → white (1)

            ax = axes[row][col]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(f"t={temp}", fontsize=8)
            ax.axis("off")
            if col == 0:
                ax.text(-0.1, 0.5, f"k={top_k}", transform=ax.transAxes,
                        fontsize=8, va="center", ha="right")

    fig.suptitle(word)
    plt.tight_layout()

    out_dir = Path(__file__).parent.parent / "generated"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{word}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    try:
        subprocess.Popen(["xdg-open", str(path)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        pass
    try:
        from IPython.display import display, Image as IPImage
        display(IPImage(str(path)))
    except ImportError:
        pass
    model.train()


def train_model(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs,
                       eval_freq, eval_iter, start_word, tokenizer):
    """
    Train the model and evaluate on the training and validation set every eval_freq steps.
    Also generates an image for the start word after each epoch.

    Returns lists of training losses, validation losses, and tokens seen at each evaluation step.
    """

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
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

        # Save model and optimizer state after each epoch, and generate an image for the start word
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            },
            f"model_and_optimizer_{epoch}.pth"
            )

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
    tokenizer = MinecraftTokenizer(words, patch_size=cfg.get("patch_size", 1))


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

    torch.manual_seed(123)
    model = MinecraftGPT(cfg)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")


    manager = QuickdrawManager()
    train_data = manager.sample_images(n=5000000, seed=42)
    val_data = manager.sample_images(n=500000, seed=123)

    train_loader = MinecraftDataloader(
        train_data, tokenizer,
        batch_size=128, max_length=cfg["context_length"] - 1,
        drop_last=True, shuffle=True, num_workers=0
    )
    val_loader = MinecraftDataloader(
        val_data, tokenizer,
        batch_size=128, max_length=cfg["context_length"] - 1,
        drop_last=False, shuffle=False, num_workers=0
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")


    torch.manual_seed(123)  # For reproducibility due to the shuffling in the data loader

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=20)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=20)
    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.1)
    num_epochs = 3
    total_steps = num_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

    """
    # Load model and optimizer state from checkpoint if you want to continue training from a previous run. 
    # Make sure to set weights_only=False to also load the optimizer state, which is important for resuming training with the same learning rate schedule.
    checkpoint_path = Path("model_and_optimizer_2.pth")
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path}")
    """

    train_model(
        model, train_loader, val_loader, optimizer, scheduler, device,
        num_epochs=num_epochs, eval_freq=100, eval_iter=20,
        start_word="apple", tokenizer=tokenizer
    )


__all__ = ["MinecraftGPT", "_main"]

"""
ViT-Tiny for CIFAR-100
======================
Self-contained Vision Transformer (~5.7M params) trained on CIFAR-100.
No external dependencies beyond PyTorch + torchvision.

Usage:
    python models/vit-tiny.py                         # default 200 epochs
    python models/vit-tiny.py --epochs 1 --batch_size 64  # quick smoke-test
"""

import argparse
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Image → patch embeddings via a strided convolution."""

    def __init__(self, img_size: int = 32, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 192):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)  →  (B, num_patches, embed_dim)
        x = self.proj(x)          # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)          # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)     # (B, num_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN → MHSA → residual → LN → MLP → residual."""

    def __init__(self, embed_dim: int = 192, num_heads: int = 3, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class ViTTiny(nn.Module):
    """
    Vision Transformer — tiny config for CIFAR-100.

    img_size=32, patch_size=4  →  64 patches
    embed_dim=192, depth=12, num_heads=3, mlp_ratio=4
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        num_classes: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)                                         # (B, 64, 192)

        cls = self.cls_token.expand(B, -1, -1)                          # (B, 1, 192)
        x = torch.cat([cls, x], dim=1)                                  # (B, 65, 192)
        x = self.pos_drop(x + self.pos_embed)

        x = self.blocks(x)
        x = self.norm(x)

        return self.head(x[:, 0])                                        # CLS token → logits


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


def build_dataloaders(batch_size: int, num_workers: int = 4, data_root: str = "./data"):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    train_set = datasets.CIFAR100(data_root, train=True,  download=True, transform=train_transform)
    val_set   = datasets.CIFAR100(data_root, train=False, download=True, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device):
    """Return (avg_loss, top1_accuracy_percent)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with autocast(enabled=device.type == "cuda"):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, 100.0 * correct / total


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, 100.0 * correct / total


def build_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """Cosine annealing with linear warmup."""
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs
    )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ViT-Tiny CIFAR-100 trainer")
    p.add_argument("--epochs",        type=int,   default=200)
    p.add_argument("--batch_size",    type=int,   default=128)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight_decay",  type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=int,   default=10)
    p.add_argument("--dropout",       type=float, default=0.1)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--data_root",     type=str,   default="./data")
    p.add_argument("--save_dir",      type=str,   default=".")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}  |  AMP: {use_amp}")

    # Model
    model = ViTTiny(dropout=args.dropout).to(device)
    n_params = count_parameters(model)
    print(f"ViT-Tiny parameters: {n_params:,}")

    # Data
    train_loader, val_loader = build_dataloaders(
        args.batch_size, args.num_workers, args.data_root
    )

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = build_scheduler(optimizer, args.warmup_epochs, args.epochs)
    scaler = GradScaler(enabled=use_amp)

    best_acc = 0.0
    best_path  = os.path.join(args.save_dir, "vit_tiny_cifar100_best.pt")
    final_path = os.path.join(args.save_dir, "vit_tiny_cifar100_final.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, device, use_amp
        )
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.2f}% | "
            f"val loss {val_loss:.4f} acc {val_acc:.2f}% | "
            f"lr {lr_now:.2e}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"epoch": epoch, "model": model.state_dict(), "val_acc": val_acc}, best_path)
            print(f"  → New best: {best_acc:.2f}%  (saved to {best_path})")

    torch.save({"epoch": args.epochs, "model": model.state_dict(), "val_acc": val_acc}, final_path)
    print(f"\nTraining complete. Best val acc: {best_acc:.2f}%")
    print(f"Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()

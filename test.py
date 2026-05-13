"""Visualize 1000 images per category at 16x16, plus side-by-side resolution comparison."""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from models.quickdraw_manager import QuickdrawManager

SEED = 42
N = 1000
IMG = QuickdrawManager.IMG_SIZE  # 16


def load_raw(manager, category, n, seed):
    npz_path = manager.data_folder / f"{category}.npz"
    total = manager._sizes[category]
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(total, size=n, replace=False))
    with np.load(npz_path, mmap_mode="r") as f:
        arr = f[f.files[0]]
        return arr[indices]  # (N, 784) original 28x28


def to_16x16(raw, category):
    encoded = [QuickdrawManager.encode_img_data(category, row) for row in raw]
    return np.stack([QuickdrawManager.decode_img_data(e)[1] for e in encoded])


def save_grid(images, size, title, out_path, cols=40):
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.5, rows * 0.5))
    for i, ax in enumerate(axes.flatten()):
        if i < n:
            ax.imshow(images[i].reshape(size, size), cmap="gray_r", interpolation="nearest")
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


def save_sidebyside(raw, category, out_path, n_show=50, cols=10):
    """Show n_show images as 28x28 (top row) vs 16x16 (bottom row) pairs."""
    raw_show = raw[:n_show]
    images_16 = to_16x16(raw_show, category)

    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 0.6, rows * 1.2))

    for i in range(n_show):
        r, c = divmod(i, cols)
        axes[r * 2][c].imshow(raw_show[i].reshape(28, 28), cmap="gray_r", interpolation="nearest")
        axes[r * 2][c].axis("off")
        axes[r * 2 + 1][c].imshow(images_16[i].reshape(16, 16), cmap="gray_r", interpolation="nearest")
        axes[r * 2 + 1][c].axis("off")

    # Hide unused axes
    for i in range(n_show, rows * cols):
        r, c = divmod(i, cols)
        axes[r * 2][c].axis("off")
        axes[r * 2 + 1][c].axis("off")

    fig.suptitle(f"{category} — top: 28×28  |  bottom: 16×16", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


manager = QuickdrawManager()

# Apple
apple_raw = load_raw(manager, "apple", N, SEED)
apple_16 = to_16x16(apple_raw, "apple")
save_grid(apple_16, IMG, f"Apple — {N} images @ 16×16 (seed={SEED})", Path("apple_train_images.png"))
save_sidebyside(apple_raw, "apple", Path("apple_sidebyside.png"))

# Eiffel Tower
eiffel_raw = load_raw(manager, "The Eiffel Tower", N, SEED)
eiffel_16 = to_16x16(eiffel_raw, "The Eiffel Tower")
save_grid(eiffel_16, IMG, f"The Eiffel Tower — {N} images @ 16×16 (seed={SEED})", Path("eiffel_train_images.png"))
save_sidebyside(eiffel_raw, "The Eiffel Tower", Path("eiffel_sidebyside.png"))

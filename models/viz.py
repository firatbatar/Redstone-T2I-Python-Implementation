"""Visualize a random image from the QuickDraw dataset for a given category."""

import sys
import numpy as np
from pathlib import Path
from .quickdraw_manager import QuickdrawManager
import matplotlib.pyplot as plt

DATA_FOLDER = Path(__file__).parent / "quickdraw"

def main():
    if len(sys.argv) < 2:
        print("Usage: python viz.py <category> [index]")
        print("       python viz.py airplane")
        print("       python viz.py airplane 42")
        sys.exit(1)

    category = sys.argv[1]
    npz_path = DATA_FOLDER / f"{category}.npz"

    if not npz_path.exists():
        raise FileNotFoundError(f"Category '{category}' not found in {DATA_FOLDER}")

    with np.load(npz_path, mmap_mode="r") as f:
        arr = f[f.files[0]]

    if len(sys.argv) >= 3:
        idx = int(sys.argv[2])
    else:
        idx = np.random.randint(0, arr.shape[0])

    img = arr[idx].reshape(28, 28)
    display = 1 - img  # invert: pixel=1 → black (0), background=0 → white (1)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(display, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"{category} (index {idx})")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

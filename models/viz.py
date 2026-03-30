"""Visualize a random image from the QuickDraw dataset for a given category."""

import sys
import numpy as np
from pathlib import Path
from .quickdraw_manager import QuickdrawManager

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
    print(f"{category} (index {idx})")
    print()
    for row in img:
        print("".join("#" if p else "." for p in row))


if __name__ == "__main__":
    main()

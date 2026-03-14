from pathlib import Path
import json
import numpy as np

from time import time

DATA_FOLDER = Path(__file__).parent / "quickdraw"

class QuickdrawManager:
    def __init__(self, data_folder: Path = DATA_FOLDER):
        if not data_folder.exists():
            raise FileNotFoundError(f"Data folder not found: {data_folder}")

        self.data_folder = data_folder
        if (self.data_folder / "_data_shape.json").exists():
            self.sizes = json.loads((self.data_folder / "_data_shape.json").read_text())
        else:
            self.sizes = {}
            for npz_path in sorted(self.data_folder.glob("*.npz")):
                category = npz_path.stem    
                with np.load(npz_path, mmap_mode="r") as f:
                    arr = f[f.files[0]]
                    self.sizes[category] = arr.shape[0]

            (data_folder / "_data_shape.json").write_text(json.dumps(self.sizes))
        
        self._sizes = self.sizes
        self.unseen_indices = {}
        for category, count in self.sizes.items():
            self.unseen_indices[category] = set(range(count))

    def __len__(self) -> int:
        return sum(self.sizes.values())
    
    def reset(self) -> None:
        self.sizes = self._sizes
        for category in self.sizes.keys():
            self.unseen_indices[category] = set(range(self.sizes[category]))

    def split_by_percentage(self, percentage: float) -> list[tuple[str, np.ndarray]]:
        if not 0 <= percentage <= 1:
            raise ValueError("Percentage must be between 0 and 1.")
        
        n = int(len(self) * percentage)
        return self.sample_images(n, seed=None)

    def sample_images(self, n: int, seed: int | None = None) -> list[tuple[str, np.ndarray]]:
        total = sum(self.sizes.values())
        n = min(n, total)

        rng = np.random.default_rng(seed)

        # Sample each category portionally
        counts = [int(n * s / total) for s in self.sizes.values()]
        if sum(counts) < n:
            counts = [int(n * s / total) + 1 for s in self.sizes.values()]
        counts = counts[:n]

        try:
            self.sizes = {category: count - counts[i] for i, (category, count) in enumerate(self.sizes.items())}
        except IndexError:
            pass

        # Sample rows per category
        all_images: list[tuple[str, np.ndarray]] = []

        for category, count in zip(self.sizes.keys(), counts):
            npz_path = self.data_folder / f"{category}.npz"
            with np.load(npz_path, mmap_mode="r") as f:
                arr = f[f.files[0]]
                indices = rng.choice(list(self.unseen_indices[category]), size=count, replace=False)
                self.unseen_indices[category] -= set(indices)
                indices.sort()  # sequential access is faster on mmap
                all_images.append(QuickdrawManager.encode_img_data(category, arr[indices]))

        # Shuffle
        perm = rng.permutation(len(all_images))
        images = [all_images[i] for i in perm]

        return images
    
    def encode_img_data(label: str, img: np.ndarray) -> tuple[str, int]:
        """Pack image data into a single integer."""
        img_list = img.reshape(-1).astype(str).tolist()
        img_str = "".join(img_list)
        img_int = int(img_str, 2)
        return (label, img_int)

    def decode_img_data(data: tuple[str, int]) -> tuple[str, np.ndarray[int]]:
        """Unpack image data from a single integer."""
        label, img_int = data
        img_str = bin(img_int)[2:]
        img_list = [int(x) for x in img_str]
        img = np.array(img_list)
        return label, img

if __name__ == "__main__":
    n = 10
    manager = QuickdrawManager()
    # print(manager.sizes)
    seen_imgs = set()
    for i in range(10):
        print(f"Sampling {n:,} images ...")
        imgs = manager.sample_images(n, seed=42)
        for img in imgs:
            if img in seen_imgs:
                print("Duplicate image found!")
            seen_imgs.add(img)
    print(len(seen_imgs))

__all__ = ["QuickdrawManager"]

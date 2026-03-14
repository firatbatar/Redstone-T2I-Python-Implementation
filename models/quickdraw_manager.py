from pathlib import Path
import json
import numpy as np

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
        
        self._sizes = self.sizes.copy()
        self.unseen_indices = {}
        for category, count in self.sizes.items():
            self.unseen_indices[category] = set(range(count))

    def __len__(self) -> int:
        return sum(self.sizes.values())
    
    def reset(self) -> None:
        self.sizes = self._sizes.copy()
        for category in self.sizes.keys():
            self.unseen_indices[category] = set(range(self.sizes[category]))

    def sample_images(self, n: int, seed: int | None = None) -> list[tuple[str, np.ndarray]]:
        total = sum(self.sizes.values())
        n = min(n, total)

        rng = np.random.default_rng(seed)

        # Sample each category portionally
        # This calculation still a bit problematic,
        # but with extra calmping later on it works
        counts = [int(n * s / total) for s in self.sizes.values()]
        deficit = n - sum(counts)
        if deficit > 0:
            deficit_indices = rng.choice(len(counts), size=deficit, replace=False)
            for i in deficit_indices:
                counts[i] += 1
        elif deficit < 0:
            surplus_indices = rng.choice(len(counts), size=-deficit, replace=False)
            for i in surplus_indices:
                counts[i] -= 1

        # Sample rows per category
        all_images: list[tuple[str, np.ndarray]] = []

        for category, count in zip(self.sizes.keys(), counts):
            actual_count = min(count, self.sizes[category])
            if actual_count == 0:
                continue
            npz_path = self.data_folder / f"{category}.npz"
            with np.load(npz_path, mmap_mode="r") as f:
                arr = f[f.files[0]]
                indices = rng.choice(list(self.unseen_indices[category]), size=actual_count, replace=False)
                self.unseen_indices[category] -= set(indices)
                self.sizes[category] -= len(indices)
                indices.sort()  # sequential access is faster on mmap
                for i in indices:
                    all_images.append(QuickdrawManager.encode_img_data(category, arr[i]))

        # Shuffle
        perm = rng.permutation(len(all_images))
        images = [all_images[i] for i in perm]

        return images
    
    @staticmethod
    def encode_img_data(label: str, img: np.ndarray) -> tuple[str, int]:
        """Pack image data into a single integer."""
        img_list = img.reshape(-1).astype(str).tolist()
        img_str = "".join(img_list)
        img_int = int(img_str, 2)
        return (label, img_int)

    @staticmethod
    def decode_img_data(data: tuple[str, int]) -> tuple[str, np.ndarray[int]]:
        """Unpack image data from a single integer."""
        label, img_int = data
        img_str = bin(img_int)[2:]
        if len(img_str) < 28*28:
            img_str = "0" * (28*28 - len(img_str)) + img_str
        # I think this is not needed as the bin 
        # function returns a min length binary string
        # but just in case
        elif len(img_str) > 28*28:
            img_str = img_str[:28*28]
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
        imgs = manager.sample_images(n, seed=42 + i)
        for img in imgs:
            if img in seen_imgs:
                print("Duplicate image found!")
            seen_imgs.add(img)
    print(len(seen_imgs))

__all__ = ["QuickdrawManager"]

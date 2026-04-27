import numpy as np
import torch
from .quickdraw_manager import QuickdrawManager

IMG_SIZE = 28

class MinecraftTokenizer:
    def __init__(self, vocab, patch_size=1):
        self.word_to_id = {word: i for i, word in enumerate(vocab)}
        self.pixel_start_id = len(vocab)
        self.patch_size = patch_size
        self._patches_per_dim = IMG_SIZE // patch_size
        self._pixels_per_patch = patch_size * patch_size
        self._bit_shifts = np.arange(self._pixels_per_patch - 1, -1, -1)

    def encode(self, img_data: tuple[str, int]):
        word, img = QuickdrawManager.decode_img_data(img_data)
        tokens = [self.word_to_id[word]]
        p = self._patches_per_dim
        grid = np.array(img, dtype=np.uint8).reshape(IMG_SIZE, IMG_SIZE)
        patches = grid.reshape(p, self.patch_size, p, self.patch_size).transpose(0, 2, 1, 3)
        flat = patches.reshape(p * p, self._pixels_per_patch)
        values = (flat << self._bit_shifts).sum(axis=1)
        tokens.extend([int(v) + self.pixel_start_id for v in values])
        return torch.tensor(tokens)

    def decode_pixels(self, token_ids):
        """Returns a flat list of pixel values (0 or 1) of length IMG_SIZE^2."""
        p = self._patches_per_dim
        patch_values = np.array([int(t) - self.pixel_start_id for t in token_ids[1:]])
        bits = (patch_values[:, None] >> self._bit_shifts) & 1
        patches = bits.reshape(p, p, self.patch_size, self.patch_size)
        grid = patches.transpose(0, 2, 1, 3).reshape(IMG_SIZE, IMG_SIZE)
        return grid.flatten().tolist()

__all__ = ["MinecraftTokenizer"]

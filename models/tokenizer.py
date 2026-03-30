# Vocabulary consists of 345 class words and 2 binary pixel tokens (0 and 1).

import torch
from .quickdraw_manager import QuickdrawManager

class MinecraftTokenizer:
    def __init__(self, vocab):
        self.word_to_id = {word: i for i, word in enumerate(vocab)}
        # Map binary pixel values (0/1) to unique IDs starting after vocab tokens
        self.pixel_start_id = len(vocab) 

    def encode(self, img_data: tuple[str, int]):
        word, img = QuickdrawManager.decode_img_data(img_data)
        tokens = [self.word_to_id[word]]
        tokens.extend([p + self.pixel_start_id for p in img])
        return torch.tensor(tokens)
    
    def decode_pixels(self, token_ids):
        """Converts IDs back to binary pixel values (0 or 1)."""
        # Slice the tensor to get only the pixel part
        pixel_ids = token_ids[1:]
        return [p - self.pixel_start_id for p in pixel_ids]

__all__ = ["MinecraftTokenizer"]

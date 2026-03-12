# Vocabulary consists of '|', 16 pixel values (0-15 grayscale), and 345 class words.

import torch

class MinecraftTokenizer:
    def __init__(self, vocab):
        self.word_to_id = {word: i for i, word in enumerate(vocab)}
        # Map 0-1 grayscale to unique IDs starting after vocab tokens
        self.pixel_start_id = 345 

    def encode(self, word, pixel_array):
        tokens = [self.word_to_id[word]]
        tokens.extend([p + self.pixel_start_id for p in pixel_array])
        return torch.tensor(tokens)
    
    def decode_pixels(self, token_ids):
        """Converts IDs back to grayscale values (0-15)"""
        # Slice the tensor to get only the pixel part
        pixel_ids = token_ids[1:]
        return [p - self.pixel_start_id for p in pixel_ids]

__all__ = ["MinecraftTokenizer"]
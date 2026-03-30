import torch
from torch.utils.data import Dataset, DataLoader
from .tokenizer import MinecraftTokenizer

class MinecraftDataset(Dataset):
    def __init__(self, img_data, tokenizer, max_length):
        self.input_ids = []
        self.target_ids = []

        items = img_data if isinstance(img_data, list) else [img_data]
        for item in items:
            token_ids = tokenizer.encode(item)
            input_chunk = token_ids[0:max_length]
            target_chunk = token_ids[1:max_length+1]
            self.input_ids.append(input_chunk)
            self.target_ids.append(target_chunk)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def MinecraftDataloader(img_data, tokenizer, batch_size=4, max_length=256, shuffle=True, drop_last=True, num_workers=0):
    dataset = MinecraftDataset(img_data, tokenizer, max_length)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            num_workers=num_workers
                            )
    return dataloader

__all__ = ["MinecraftDataloader"]

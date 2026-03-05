# Initialize the tokenizer
import torch
from tokenizer import MinecraftTokenizer
from dataset_loader import MinecraftDataloader

with open("vocab.txt", "r", encoding="utf-8") as f:
        words = f.read()
all_words = words.split('\n')[:-1]
vocab = {token:integer for integer, token in enumerate(all_words)}
tokenizer = MinecraftTokenizer(vocab)

dataloader = MinecraftDataloader(raw_text, 
                                 tokenizer, 
                                 batch_size=4, 
                                 max_length=4, 
                                 stride=2, 
                                 shuffle=False)

# test code
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)




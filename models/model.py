# Initialize the tokenizer
import torch
from tokenizer import MinecraftTokenizer
with open("vocab.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

all_words = raw_text.split('\n')[:-1]
vocab = {token:integer for integer, token in enumerate(all_words)}

tokenizer = MinecraftTokenizer(vocab)

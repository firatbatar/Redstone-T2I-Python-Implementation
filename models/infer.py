import sys
from pathlib import Path
import torch
import json

from .tokenizer import MinecraftTokenizer
from .model import MinecraftGPT, generate_and_print_image

CONFIG_PATH = Path(__file__).parent / "config.json"

def infer(word, checkpoint_path="model_and_optimizer.pth"):
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    else:
        cfg = json.loads(CONFIG_PATH.read_text())


    with open(Path(__file__).parent / "vocab.txt", "r", encoding="utf-8") as f:
        vocab = f.read().split('\n')[:-1]
    tokenizer = MinecraftTokenizer(vocab)

    if word not in tokenizer.word_to_id:
        print(f"Unknown word '{word}'. Available words are in models/vocab.txt.")
        sys.exit(1)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        major, minor = map(int, torch.__version__.split(".")[:2])
        if (major, minor) >= (2, 9):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    print(f"Using {device} device.")

    model = MinecraftGPT(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)

    generate_and_print_image(model, tokenizer, device, word)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m models.infer <word> [checkpoint_path]")
        sys.exit(1)
    word = sys.argv[1]
    checkpoint = sys.argv[2] if len(sys.argv) > 2 else "model_and_optimizer.pth"
    infer(word, checkpoint)

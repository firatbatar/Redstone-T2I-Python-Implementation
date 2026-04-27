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

    vocab_file = Path(__file__).parent / "vocab.txt"
    if not vocab_file.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_file}")
    words = vocab_file.read_text().split('\n')[:-1]
    tokenizer = MinecraftTokenizer(words, patch_size=cfg.get("patch_size", 1))

    if word not in tokenizer.word_to_id:
        raise ValueError(f"Unknown word '{word}'. Available words are in models/vocab.txt.")

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
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

    has_affine = any(v.__class__.__name__ == "AffineQuantizedTensor" for v in state_dict.values())
    has_layernorm_q = any('scale_factor' in k for k in state_dict.keys())
    is_quantized = has_affine or has_layernorm_q

    if is_quantized:
        import torch.nn as nn
        from torchao.quantization import quantize_, Int8WeightOnlyConfig
        from .transformerblock import LayerNorm
        quantize_(model, Int8WeightOnlyConfig(), filter_fn=lambda m, _: isinstance(m, nn.Linear))
        quantize_(model, Int8WeightOnlyConfig(), filter_fn=lambda m, _: isinstance(m, nn.Embedding))
        for module in model.modules():
            if isinstance(module, LayerNorm):
                module.quantize()

    model.load_state_dict(state_dict, assign=is_quantized)
    model.to(device)

    generate_and_print_image(model, tokenizer, device, word)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m models.infer <word> [checkpoint_path]")
        sys.exit(1)
    word = sys.argv[1]
    checkpoint = sys.argv[2] if len(sys.argv) > 2 else "model_and_optimizer.pth"
    infer(word, checkpoint)

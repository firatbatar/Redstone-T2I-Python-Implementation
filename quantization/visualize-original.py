import torch

""""
Model architecture
{
    "vocab_size": 347,      # 345 classes + black and white bit
    "context_length": 785,  # prompt + 784 pixels
    "emb_dim": 256,         # can test 128, 256, 512. increase leads to overfit.
    "n_heads": 8,
    "n_layers": 6,
    "drop_rate": 0.1,
    "qkv_bias": False
}

Checkpoint structure
{
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict()
}
"""
# Load the archive
checkpoint = torch.load("model_and_optimizer_0.pth", map_location="cpu", weights_only=False)

# Access the dictionary containing the actual weights
model_weights = checkpoint["model_state_dict"]

import io

print(f"{'Layer Name':<50} | {'Shape'}")
print("-" * 70)

for layer_name, weights in model_weights.items():
    print(f"{layer_name:<50} | {list(weights.size())}")

print("-" * 70)

buf = io.BytesIO()
torch.save(model_weights, buf)
print(f"Total Parameters: {sum(p.numel() for p in model_weights.values()):,}")
print(f"Total Weight Size (serialized): {buf.tell() / (1024**2):.2f} MB")
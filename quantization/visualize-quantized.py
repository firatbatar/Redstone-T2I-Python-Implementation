import io
import torch

model_weights = torch.load("model_int8.pth", map_location="cpu", weights_only=False)

print(f"{'Layer Name':<50} | {'Shape':<30} | {'Type'}")
print("-" * 95)

for layer_name, weights in model_weights.items():
    print(f"{layer_name:<50} | {str(list(weights.size())):<30} | {weights.__class__.__name__}")

print("-" * 95)

buf = io.BytesIO()
torch.save(model_weights, buf)
print(f"Total Parameters: {sum(p.numel() for p in model_weights.values()):,}")
print(f"Total Weight Size (serialized): {buf.tell() / (1024**2):.2f} MB")

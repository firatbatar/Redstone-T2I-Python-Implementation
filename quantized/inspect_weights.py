import sys
import torch

def inspect(path: str) -> None:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)

    print(f"Checkpoint: {path}\n")
    print(f"{'Layer':<60}  {'Shape':<30}  {'dtype':<15}  {'min':>10}  {'max':>10}  {'mean':>10}")
    print("-" * 120)
    for name, tensor in state.items():
        t = tensor.float()
        print(f"{name:<60}  {str(tensor.shape):<30}  {str(tensor.dtype):<15}  {t.min().item():>10.4f}  {t.max().item():>10.4f}  {t.mean().item():>10.4f}")
    print(f"\nTotal tensors: {len(state)}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "model_and_optimizer_5M.pth"
    inspect(path)

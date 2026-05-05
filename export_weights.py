import os
import sys
import torch

LAYERS = 6
HEADS = 8
MLP_SCALE = 4
EMBED_SIZE = 256
HEAD_SIZE = EMBED_SIZE // HEADS  # 32
VOCAB_SIZE = 347
CONTEXT = 785
MATMUL_FIXED_POINT = 18
FIXED_POINT_SIZE = 24
FIXED_POINT_MASK = (1 << FIXED_POINT_SIZE) - 1

OUT_DIR = "weights2/weight_files"


def _decode_weight_byte(b):
    neg = (b >= 128)
    w = b % 128
    if w < 64:
        big, small, shift = w // 8, w % 8, 8
    elif w < 96:
        w2 = w - 64
        big, small, shift = 4 + w2 // 8, w2 % 8, 7
    elif w < 112:
        w2 = w - 96
        big, small, shift = 2 + w2 // 8, w2 % 8, 5
    elif w < 120:
        w2 = w - 112
        big, small, shift = 1 + w2 // 8, w2 % 8, 3
    else:
        w2 = w - 120
        big, small, shift = 1 + w2 // 8, w2 % 8, 2
    value = big * (2.0 ** -shift) + small * (2.0 ** -(shift + 3))
    return -value if neg else value


# All 256 representable weight values
WEIGHT_TABLE = [_decode_weight_byte(b) for b in range(256)]
MAX_WEIGHT = max(abs(v) for v in WEIGHT_TABLE)  # ~0.469


def quantize_weight(v):
    return min(range(256), key=lambda b: (WEIGHT_TABLE[b] - v) ** 2)


def write_weight_matrix(path, tensor):
    """Quantize (rows, cols) float tensor to 1-byte-per-weight binary file."""
    rows, cols = tensor.shape
    data = tensor.float().numpy()
    clipped = sum(1 for i in range(rows) for j in range(cols) if abs(data[i, j]) > MAX_WEIGHT)
    if clipped:
        print(f"  WARNING: {clipped}/{rows*cols} weights exceed ±{MAX_WEIGHT:.3f} and will be clipped — {path}")
    with open(path, "wb") as f:
        for i in range(rows):
            for j in range(cols):
                f.write(bytes([quantize_weight(float(data[i, j]))]))


def write_bias(path, tensor):
    """Write 1D bias as 24-bit signed fixed-point (MATMUL_FIXED_POINT fractional bits)."""
    with open(path, "wb") as f:
        for v in tensor.float().numpy():
            fixed = int(round(float(v) * (1 << MATMUL_FIXED_POINT))) & FIXED_POINT_MASK
            f.write(fixed.to_bytes(3, byteorder="little"))


def write_embedding(path, tensor):
    """Write (n, EMBED_SIZE) embedding table as 24-bit fixed-point."""
    rows = tensor.shape[0]
    with open(path, "wb") as f:
        for i in range(rows):
            for j in range(EMBED_SIZE):
                fixed = int(round(float(tensor[i, j]) * (1 << MATMUL_FIXED_POINT))) & FIXED_POINT_MASK
                f.write(fixed.to_bytes(3, byteorder="little"))


def write_layernorm(path, scale, shift):
    """Write LayerNorm scale and shift to a single file.

    Scale: stored as scale * 2^22 (emulator loads as cur // 2, giving scale * 2^21,
           then shifts >> 21 after multiply to recover scale * normalized_value).
    Shift: stored as shift * 2^18 (added directly to fixed-point output).
    """
    with open(path, "wb") as f:
        for v in scale.float().numpy():
            stored = int(round(float(v) * (1 << 22))) & FIXED_POINT_MASK
            f.write(stored.to_bytes(3, byteorder="little"))
        for v in shift.float().numpy():
            stored = int(round(float(v) * (1 << MATMUL_FIXED_POINT))) & FIXED_POINT_MASK
            f.write(stored.to_bytes(3, byteorder="little"))


def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "model.pt"
    print(f"Loading {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sd = ckpt["model_state_dict"]

    for subdir in ["layernorm", "attention", "mlp", "embedding", "unembedding"]:
        os.makedirs(f"{OUT_DIR}/{subdir}", exist_ok=True)

    # --- Embeddings ---
    print("Embeddings...")
    wte = sd["tok_emb.weight"]   # (347, 256)
    wpe = sd["pos_emb.weight"]   # (785, 256)
    if wpe.dim() == 3:
        wpe = wpe.squeeze(0)
    assert wte.shape == (VOCAB_SIZE, EMBED_SIZE), f"tok_emb shape mismatch: {wte.shape}"
    assert wpe.shape == (CONTEXT, EMBED_SIZE),    f"pos_emb shape mismatch: {wpe.shape}"
    write_embedding(f"{OUT_DIR}/embedding/wte.bin", wte)
    write_embedding(f"{OUT_DIR}/embedding/wpe.bin", wpe)

    # --- LayerNorms ---
    print("LayerNorms...")
    for layer in range(LAYERS):
        for norm_idx, norm_name in enumerate(["norm1", "norm2"]):
            ln_index = 2 * layer + norm_idx + 1
            write_layernorm(
                f"{OUT_DIR}/layernorm/ln_{ln_index}.bin",
                sd[f"trf_blocks.{layer}.{norm_name}.scale"],
                sd[f"trf_blocks.{layer}.{norm_name}.shift"],
            )
    write_layernorm(
        f"{OUT_DIR}/layernorm/ln_{2 * LAYERS + 1}.bin",
        sd["final_norm.scale"],
        sd["final_norm.shift"],
    )

    # --- Attention ---
    print("Attention...")
    for layer in range(LAYERS):
        Wq = sd[f"trf_blocks.{layer}.att.W_query.weight"]    # (256, 256)
        Wk = sd[f"trf_blocks.{layer}.att.W_key.weight"]      # (256, 256)
        Wv = sd[f"trf_blocks.{layer}.att.W_value.weight"]    # (256, 256)
        Wo = sd[f"trf_blocks.{layer}.att.out_proj.weight"]   # (256, 256)
        bo = sd[f"trf_blocks.{layer}.att.out_proj.bias"]     # (256,)
        for head in range(HEADS):
            s, e = head * HEAD_SIZE, (head + 1) * HEAD_SIZE
            write_weight_matrix(f"{OUT_DIR}/attention/att_{layer}_h{head}_query.bin", Wq[s:e])
            write_weight_matrix(f"{OUT_DIR}/attention/att_{layer}_h{head}_key.bin",   Wk[s:e])
            write_weight_matrix(f"{OUT_DIR}/attention/att_{layer}_h{head}_value.bin", Wv[s:e])
        write_weight_matrix(f"{OUT_DIR}/attention/att_{layer}_proj.bin", Wo)
        write_bias(f"{OUT_DIR}/attention/att_{layer}_proj.bias", bo)

    # --- MLP ---
    print("MLP...")
    for layer in range(LAYERS):
        Wu = sd[f"trf_blocks.{layer}.ff.layers.0.weight"]   # (1024, 256)
        bu = sd[f"trf_blocks.{layer}.ff.layers.0.bias"]     # (1024,)
        Wd = sd[f"trf_blocks.{layer}.ff.layers.2.weight"]   # (256, 1024)
        bd = sd[f"trf_blocks.{layer}.ff.layers.2.bias"]     # (256,)
        write_weight_matrix(f"{OUT_DIR}/mlp/mlp_{layer}_up.bin",   Wu)
        write_bias(         f"{OUT_DIR}/mlp/mlp_{layer}_up.bias",  bu)
        write_weight_matrix(f"{OUT_DIR}/mlp/mlp_{layer}_down.bin", Wd)
        write_bias(         f"{OUT_DIR}/mlp/mlp_{layer}_down.bias",bd)

    # --- Unembedding ---
    print("Unembedding...")
    lm_head = sd["output_layer.weight"]   # (347, 256)
    assert lm_head.shape == (VOCAB_SIZE, EMBED_SIZE), f"output_layer shape mismatch: {lm_head.shape}"
    write_weight_matrix(f"{OUT_DIR}/unembedding/lm_head.bin", lm_head)

    print(f"Done. Files written to {OUT_DIR}/")
    print(f"Note: max representable weight magnitude is ±{MAX_WEIGHT:.4f}.")
    print("If clipping warnings appeared, consider scaling weights before export.")


if __name__ == "__main__":
    main()

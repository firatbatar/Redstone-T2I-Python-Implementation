# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Transformer-based model that generates images (binary) from text labels, intended for eventual reimplementation in Minecraft Redstone. Intentionally minimal and simple.

## Running

```bash
# From the project root
source .venv/bin/activate
python -m models                                  # train
python -m models.infer <word> [checkpoint_path]   # inference
```

The `.venv` is located inside the project root and contains all dependencies.

## Architecture

All source files live in `models/`:

- **`quickdraw_manager.py`** — `QuickdrawManager`: loads `.npz` files from `models/quickdraw/`, caches category sizes in `_data_shape.json`. `sample_images(n, seed)` returns proportionally sampled `(label, encoded_int)` tuples without replacement across calls, using `unseen_indices` to track what's been seen. Deficit from integer truncation is distributed randomly across categories; each category's count is clamped to its remaining unseen images. `encode_img_data` downscales each 28×28 source image to `IMG_SIZE` (=16) with Lanczos resampling, thresholds at >127 to rebinarize, then packs the binary pixel string into a single integer; `decode_img_data` unpacks it back to a flat `IMG_SIZE²` (=256) pixel array.
- **`tokenizer.py`** — `MinecraftTokenizer(vocab, patch_size)`: `encode(img_data: tuple[str, int])` decodes the packed image to a 16×16 grid, splits it into non-overlapping `patch_size`×`patch_size` patches, and packs each patch's bits into a single value, producing the token sequence `[word_id, patch_0, ..., patch_{N-1}]` (N = `(IMG_SIZE/patch_size)²`). Patch values are offset by `pixel_start_id = len(vocab)` to avoid collision with word tokens. `decode` reverses the patch portion back to a flat pixel list.
- **`dataset_loader.py`** — `MinecraftDataset` / `MinecraftDataloader`: sliding-window dataset over token sequences for next-token prediction. `MinecraftDataloader` is a convenience function wrapping the dataset in a PyTorch `DataLoader` and returns it directly.
- **`transformerblock.py`** — `TransformerBlock`, `MultiHeadAttention`, `LayerNorm`, `GELU`, `FeedForward`. Standard decoder-only transformer components with causal mask.
- **`model.py`** — `MinecraftGPT`: GPT-style model using the above blocks. `_main()` is the entry point: loads vocab/config, initializes the model, samples training/validation data, runs the training loop, and saves a per-epoch checkpoint. `calc_loss_batch` uses standard cross-entropy loss. Two generation functions exist: `generate_image` (greedy argmax, unused at runtime) and `generate` (top-k + temperature sampling, used by `generate_and_print_image`). `generate_and_print_image` generates a grid of images over the given `temperatures`/`top_ks` (defaults `(1.0,)` / `(4,)`), renders it with matplotlib (inverted: pixel=1 → black), saves a PNG to `generated/`, and tries to open it via `xdg-open` / IPython display.
- **`infer.py`** — `infer(word, checkpoint_path)`: loads config and a saved checkpoint, then generates a sample image for the given word via `generate_and_print_image`. Entry point for `python -m models.infer`.
- **`viz.py`** — `python -m models.viz <category> [index]`: renders a real QuickDraw image for a class (random index if unspecified) to a PNG in `generated/` for visual comparison against model output.
- **`__init__.py`** / **`__main__.py`** — package entry points; `python -m models` calls `_main()`.
- **`config.json`** — model hyperparameters (see Config below), loaded by both `model.py` and `infer.py`.
- **`vocab.txt`** — One class label per line (100 entries, Quick Draw dataset classes). Used to build the vocabulary at startup.

## Config (`models/config.json`)

Hyperparameters live in `models/config.json`, loaded at startup by both `model.py` and `infer.py`.

```json
{
    "vocab_size": 116,      // 100 class words + 16 patch values (2^(patch_size²) = 2^4)
    "context_length": 65,   // 1 word token + 64 patch tokens (8x8 patches over a 16x16 image)
    "emb_dim": 128,
    "n_heads": 8,
    "n_layers": 4,
    "drop_rate": 0.1,
    "qkv_bias": false,
    "patch_size": 2
}
```

`vocab_size` and `context_length` are derived from `patch_size` and the vocab size: each patch holds `patch_size²` bits (so `2^(patch_size²)` possible values, appended after the word tokens), and the image is `IMG_SIZE/patch_size` patches per dimension. Keep these consistent when changing `patch_size` or the number of words.

## Quantization / Redstone Target

The end goal is running inference entirely in Minecraft Redstone, which has no floating-point hardware — only binary logic. This requires rewriting the forward pass to use integer/fixed-point arithmetic with byte-quantized weights. This work lives on the `real-quantize` branch (the earlier Python emulator work has been removed from this branch). The intended quantization strategy, modeled on CraftGPT:

- **Weights**: 1-byte quantized (uint8), encoded with a variable-bit-shift scheme — not standard int8
- **Arithmetic**: 24-bit fixed-point throughout; no floating-point in the forward pass
- **Attention softmax**: Custom float16 emulation or precomputed lookup tables
- **Embeddings**: 3-byte format (wider range needed for input/output layers)
- **LayerNorm**: 3-byte scale/shift values
- **No standard `torch.quantization`** — the forward pass must be fully rewritten to use integer arithmetic compatible with Redstone circuit constraints

## Training

- Train/validation data sampled from QuickDraw via `QuickdrawManager` (requests up to 5,000,000 / 500,000 images, clamped to what's available across categories)
- Batch size 128, AdamW optimizer (lr=0.003, weight_decay=0.1), 3 epochs
- Cosine-annealing LR schedule (`CosineAnnealingLR`, `eta_min=1e-5`) over all steps; gradients clipped to `max_norm=1.0`
- Sliding window: `max_length = context_length - 1 = 64` (one window per image)
- Evaluates train/val loss every 100 steps (`eval_freq=100`, `eval_iter=20`)
- After each epoch, generates a sample image for `"apple"` via `generate_and_print_image` (saved as a PNG in `generated/`)
- Checkpoint saved per epoch as `model_and_optimizer_{epoch}.pth` (model + optimizer state); inference defaults to `model_and_optimizer.pth`
- Device selection: prefers CUDA, then MPS (PyTorch ≥ 2.9 only), then CPU

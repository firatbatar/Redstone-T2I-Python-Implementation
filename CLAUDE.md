# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Transformer-based model that generates images (binary) from text labels, intended for eventual reimplementation in Minecraft Redstone. Intentionally minimal and simple.

## Running

```bash
cd models
source .venv/bin/activate
python -m models                                  # train
python -m models.infer <word> [checkpoint_path]   # inference
```

The `.venv` is located inside `models/` and contains all dependencies (torch, etc.).

## Architecture

All source files live in `models/`:

- **`quickdraw_manager.py`** — `QuickdrawManager`: loads `.npz` files from `models/quickdraw/`, caches category sizes in `_data_shape.json`. `sample_images(n, seed)` returns proportionally sampled `(label, encoded_int)` tuples without replacement across calls, using `unseen_indices` to track what's been seen. Deficit from integer truncation is distributed randomly across categories; each category's count is clamped to its remaining unseen images. Images are packed into a single integer via `encode_img_data` (binary pixel string → int) and unpacked with `decode_img_data`.
- **`tokenizer.py`** — `MinecraftTokenizer`: `encode(img_data: tuple[str, int])` decodes the packed image integer back to a pixel array via `QuickdrawManager.decode_img_data`, then produces a flat token sequence `[word_id, pixel_id_0, ..., pixel_id_783]`. Pixel values (0/1) are offset by `pixel_start_id = 345` to avoid collision with word tokens. `decode_pixels` reverses the pixel portion.
- **`dataset_loader.py`** — `MinecraftDataset` / `MinecraftDataloader`: sliding-window dataset over token sequences for next-token prediction. `MinecraftDataloader` is a convenience function wrapping the dataset in a PyTorch `DataLoader` and returns it directly.
- **`transformerblock.py`** — `TransformerBlock`, `MultiHeadAttention`, `LayerNorm`, `GELU`, `FeedForward`. Standard decoder-only transformer components with causal mask.
- **`model.py`** — `MinecraftGPT`: GPT-style model using the above blocks. `_main()` is the entry point: loads vocab, initializes the model, samples training/validation data, runs the training loop, and saves a checkpoint. `calc_loss_batch` uses standard cross-entropy loss (no class weighting — note: pixel-1 token at index 346 is underrepresented due to class imbalance). `generate_and_print_image` prints a greedy-decoded image as ASCII (`#`/`.`) after each epoch.
- **`infer.py`** — `infer(word, checkpoint_path)`: loads a saved checkpoint and generates/prints a sample image for the given word. Entry point for `python -m models.infer`.
- **`__init__.py`** / **`__main__.py`** — package entry points; `python -m models` calls `_main()`.
- **`vocab.txt`** — One class label per line (345 entries, Quick Draw dataset classes). Used to build the vocabulary at startup.

## Config (`model.py` and `infer.py`)

The `cfg` dict is duplicated in both files and must be kept in sync when changing hyperparameters.

```python
cfg = {
    "vocab_size": 347,      # 345 class words + 2 pixel token values (0 and 1)
    "context_length": 785,  # 1 word token + 784 pixel tokens (28x28)
    "emb_dim": 256,
    "n_heads": 8,
    "n_layers": 6,
    "drop_rate": 0.1,
    "qkv_bias": False
}
```

## Training

- 100,000 train images, 5,000 validation images sampled from QuickDraw via `QuickdrawManager`
- Batch size 64, AdamW optimizer (lr=0.00175, weight_decay=0.1), 1 epoch
- Sliding window: `max_length = stride = context_length - 1 = 784` (one window per image)
- Evaluates train/val loss every 100 steps (`eval_freq=100`, `eval_iter=20`)
- After each epoch, generates and prints a sample image to stdout using `generate_and_print_image`
- Checkpoint saved to `model_and_optimizer.pth` after training completes (model + optimizer state)
- Device selection: prefers CUDA, then MPS (PyTorch ≥ 2.9 only), then CPU

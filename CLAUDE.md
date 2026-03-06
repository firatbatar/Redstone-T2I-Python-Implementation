# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Transformer-based model that generates images (initially binary/grayscale) from text labels, intended for eventual reimplementation in Minecraft Redstone. Intentionally minimal and simple.

## Running

```bash
cd models
source .venv/bin/activate
python model.py
```

The `.venv` is located inside `models/` and contains all dependencies (torch, etc.).

## Architecture

All source files live in `models/`:

- **`tokenizer.py`** — `MinecraftTokenizer`: encodes a class word + pixel array into a flat token sequence: `[word_id, sep_id, pixel_id_0, ..., pixel_id_N]`. Pixel values are binary (0 = white, 1 = black) and offset by `pixel_start_id` to avoid collision with word tokens. `model.py` quantizes raw 0/255 values to 0/1 before encoding.
- **`dataset_loader.py`** — `MinecraftDataset` / `MinecraftDataloader`: sliding-window dataset over token sequences. Currently incomplete (dataloader doesn't return the DataLoader object).
- **`transformerblock.py`** — `TransformerBlock`, `MultiHeadAttention`, `LayerNorm`, `GELU`, `FeedForward`. Standard decoder-only transformer components with causal mask.
- **`model.py`** — `MinecraftGPT`: GPT-style model using the above blocks. Entry point for running/testing. Contains an inline pixel array at the top used as a test sample.
- **`vocab.txt`** — One class label per line (~345 entries, Quick Draw dataset classes). Used to build the vocabulary at startup.

## Config (`model.py`)

```python
cfg = {
    "vocab_size": 348,      # 345 class words + separator + 2 pixel sentinels
    "context_length": 256,
    "emb_dim": 256,
    "n_heads": 8,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
```

## Known Issues / In-Progress

- `dataset_loader.py`: `MinecraftDataloader` never returns the created DataLoader; also `MinecraftDataset` calls `tokenizer.encode(txt)` with wrong signature (requires `word` + `pixel_array`).
- No training loop implemented yet.
- No model checkpointing.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python implementation of a transformer-based Text-to-Image (T2I) model called **RedstoneTransformer**. The project generates 64x64 binary images from text prompts using an encoder-decoder transformer architecture. Currently in early-stage research/development, originally developed as a Google Colab notebook.

## Running

```bash
python large_model.py
```

No build system, test suite, or linter is configured. Dependencies must be installed manually:

```bash
pip install torch pandas numpy matplotlib scikit-learn pillow tqdm
```

GPU is used automatically if available (`torch.cuda.is_available()`).

## Architecture

All code lives in `large_model.py` (single-file project).

**RedstoneTransformer** (encoder-decoder):
- **Text encoder**: Embeds text tokens → transformer encoder → context vectors
- **Image decoder**: Autoregressively generates 4096 pixels (64x64) conditioned on text context
- Default config: d_model=128, nhead=4, 2 encoder layers, 2 decoder layers, num_bins=3 (pixel values 0/1/2)

**Data pipeline**: `process_image()` quantizes images to binary → `build_vocab()` builds word-level vocabulary → `tokenize_and_pad()` produces token sequences (max 32 tokens) → `ImageTextDataset` wraps for PyTorch DataLoader.

**Training** (`train_model()`): Cross-entropy loss with causal masking, Adam optimizer (lr=0.001), BOS token prepended to pixel sequences.

**Inference** (`generate_image()`): Pixel-by-pixel autoregressive sampling with probability thresholding (greedy if prob > 0.9, else multinomial).

## Important Notes

- Data paths are hardcoded to Google Colab (`/content/drive/MyDrive/ENS491/...`) — must be updated for local use.
- No `requirements.txt` exists yet.
- No model checkpointing/saving is implemented.
- No validation split in training loop.
- The tokenizer is simple word-level with no OOV handling.

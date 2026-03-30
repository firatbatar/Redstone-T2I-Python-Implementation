# Redstone T2I — Transformer Text-to-Image (Python)

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/firatbatar/Redstone-T2I-Python-Implementation.git
   cd Redstone-T2I-Python-Implementation
   ```

2. **Create a virtual environment inside project root**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux / macOS
   # .venv\Scripts\activate    # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the QuickDraw dataset** into `models/quickdraw/` as `.npz` files (one per class).

## Training

From the project root directory, activate the virtual environment and run:

```bash
source .venv/bin/activate
python -m models
```

This trains the model on 100,000 QuickDraw images for 3 epochs and saves a checkpoint to `model_and_optimizer.pth`.

## Inference

Generate an image for a given word:

```bash
python -m models.infer <word>
python -m models.infer <word> <checkpoint_path>
```

Example:

```bash
python -m models.infer apple
```

Prints a 28×28 ASCII image (`#`/`.`) to stdout.

## Project Structure

```
.
├── models/
│   ├── __init__.py          # Package entry point, exposes main()
│   ├── __main__.py          # Enables `python -m models`
│   ├── infer.py             # Draw an image of your choice!
│   ├── viz.py               # Compare it with an image from the original dataset
│   ├── model.py             # MinecraftGPT model + training code
│   ├── quickdraw_manager.py # Quick Draw Dataset manager (enables sampling)
│   ├── tokenizer.py         # MinecraftTokenizer (word + pixel encoding)
│   ├── dataset_loader.py    # Dataset and DataLoader utilities
│   ├── transformerblock.py  # Transformer block components
│   └── vocab.txt            # Class labels vocabulary (345 entries)
├── requirements.txt
├── CLAUDE.md
└── README.md
```

## License

See [LICENSE](LICENSE) for details.

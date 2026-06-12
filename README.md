# Redstone T2I — Transformer Text-to-Image (Python)
This project attempts to implement a decoder-only text-to-image language model within Minecraft. The first step in achieving this task is to implement and pretrain a PyTorch base model. This implementation can be found in './models' and consists of ~830,000 parameters. Configuration details can also be found in this directory. The second step is to emulate the Minecraft environment using Python to effectively validate and verify circuit design. This work is still in progress, and can be found in the 'emulator' branch. The last step is to implement everything in Minecraft. Although the final LLM in Minecraft is yet to be implemented, showcases of some of our designs can be found in the following:
- Multi-class Logistic Regression: https://www.youtube.com/watch?v=xu7tTfp_Wi8
- Binary Arithmetic: https://www.youtube.com/watch?v=BhFFJV-35bY

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

4. **Download the QuickDraw dataset**

   Download the bitmap `.npz` files (one per class) and place them in `models/quickdraw/`.
   Each file must be named `<class>.npz` to match a label in [`models/vocab.txt`](models/vocab.txt)
   (e.g. `airplane.npz`, `alarm clock.npz`), for all 100 classes.

   Source: https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap;tab=objects?prefix=&forceOnObjectsSortingFiltering=false


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


## Visualization
Render a real image from the QuickDraw dataset to compare against the model's output.
Saves a PNG to `generated/` and opens it:

```bash
python -m models.viz <category>          # random image from the class
python -m models.viz <category> <index>  # specific image by index
```

Example:

```bash
python -m models.viz airplane
python -m models.viz airplane 42
```


## Configuration
Model hyperparameters (vocab size, context length, embedding dim, layers, heads, patch size)
live in [`models/config.json`](models/config.json) and are loaded by both `model.py` and `infer.py`.


## Quantization
The end goal is running inference entirely in Minecraft Redstone, which has no floating-point
hardware — only binary logic built from redstone components. The model must therefore be rewritten
to use integer/fixed-point arithmetic, with weights quantized to a compact byte format. Standard
`torch.quantization` is not used; the forward pass is reimplemented from scratch so every operation
maps onto circuits the Redstone target can actually execute. This work lives on the `real-quantize`
branch.


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

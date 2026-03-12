# Redstone T2I — Transformer Text-to-Image (Python)

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/firatbatar/Redstone-T2I-Python-Implementation.git
   cd Redstone-T2I-Python-Implementation
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux / macOS
   # .venv\Scripts\activate    # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Running the Model Test

From the **project root** directory, run:

```bash
python -m models
```

This executes the `_main()` function in `models/model.py`, which:

1. Loads the vocabulary from `models/vocab.txt`
2. Initializes the `MinecraftTokenizer` and `MinecraftGPT` model
3. Encodes a sample pixel array with the label `"apple"`
4. Runs a simple greedy generation and prints the output

## Training

> **Note:** The training module is a placeholder and is not yet fully implemented.

From the **project root** directory, run:

```bash
python -m training
```

## Project Structure

```
.
├── models/
│   ├── __init__.py          # Package entry point, exposes main()
│   ├── __main__.py          # Enables `python -m models`
│   ├── model.py             # MinecraftGPT model + test code
│   ├── tokenizer.py         # MinecraftTokenizer (word + pixel encoding)
│   ├── dataset_loader.py    # Dataset and DataLoader utilities
│   ├── transformerblock.py  # Transformer block components
│   └── vocab.txt            # Class labels vocabulary (345 entries)
├── training/
│   ├── __init__.py          # Package entry point, exposes main()
│   └── __main__.py          # Enables `python -m training`
├── requirements.txt
└── README.md
```

## License

See [LICENSE](LICENSE) for details.

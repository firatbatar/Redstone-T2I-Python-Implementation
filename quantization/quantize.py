import sys
import json
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.model import MinecraftGPT

cfg = json.loads((PROJECT_ROOT / "models" / "config.json").read_text())
checkpoint_path = PROJECT_ROOT / "quantization" / "model_and_optimizer_2.pth"

model = MinecraftGPT(cfg)
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model_state_dict"])
model.eval()

import copy
import torch.nn as nn
from torchao.quantization import quantize_, Int8WeightOnlyConfig

quantized_model = copy.deepcopy(model)
# Linear layers: int8
quantize_(quantized_model, Int8WeightOnlyConfig(), filter_fn=lambda m, _: isinstance(m, nn.Linear))
# Embedding layers: int8 (tok_emb, pos_emb)
quantize_(quantized_model, Int8WeightOnlyConfig(), filter_fn=lambda m, _: isinstance(m, nn.Embedding))
# LayerNorm: int8 via custom quantize() — scale/shift stored as int8, dequantized at runtime
from models.transformerblock import LayerNorm
for module in quantized_model.modules():
    if isinstance(module, LayerNorm):
        module.quantize()

torch.save(quantized_model.state_dict(), PROJECT_ROOT / "model_int8.pth")
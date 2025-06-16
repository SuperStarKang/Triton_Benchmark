"""
FLCE 부분을 대체한 Llama Model
: Llama 3.2 1B Model + Liger FLCE
"""

import torch
from ops import LlamaForCausalLMLiger

model_id = "meta-llama/Llama-3.2-1B"

flce_model = LlamaForCausalLMLiger.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
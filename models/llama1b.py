"""
원본 Llama Model
: Llama 3.2 1B Model
"""

from transformers import LlamaForCausalLM, LlamaConfig
import torch

model_id = "meta-llama/Llama-3.2-1B"


llama3_model = LlamaForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)


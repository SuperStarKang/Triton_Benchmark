import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaConfig
from torch.nn import CrossEntropyLoss
from ops import LlamaForCausalLMLiger

class StandardLlamaHead(nn.Module):
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B",
        bias: bool = False,
        ignore_index = -100,
        use_optimized: bool = False
    ):
        super().__init__()
        hf = LlamaForCausalLMLiger.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ) if use_optimized else LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.config: LlamaConfig = hf.config
        self.lm_head: nn.Linear = hf.lm_head
        self.loss_fn = CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.LongTensor = None,
    ):
        logits = self.lm_head(hidden_states)

        if labels is None:
            return None, logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # flatten 작업
        B, S1, V = shift_logits.shape
        shift_logits = shift_logits.view(-1, V)
        shift_labels = shift_labels.view(-1)

        shift_labels = shift_labels.to(shift_logits.device)

        loss = self.loss_fn(shift_logits, shift_labels)
        return loss, logits
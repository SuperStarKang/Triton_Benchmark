"""
Fused Linear Cross Entropy 최적화 코드
"""

import torch
import triton
import torch.nn as nn

from liger_kernel.ops.cross_entropy import liger_cross_entropy_kernel
from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd
from liger_kernel.ops.utils import element_mul_kernel
from liger_kernel.ops.utils import is_hip
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.amp import custom_fwd, custom_bwd


# The optimal maximum block size depends on your hardware, your kernel, and your dtype
MAX_FUSED_SIZE = 65536 // 2


#############################
########## Forward ##########
#############################

def fused_linear_cross_entropy_forward(
    _input,
    weight,
    target,
    ce_weight=None,
    bias=None,
    ignore_index=-100,
    lse_square_scale=0.0,
    label_smoothing=0.0,
    reduction="mean",
    softcap=None,
    return_z_loss=False,
):
    assert isinstance(return_z_loss, bool), f"return_z_loss must be True or False. Got: {return_z_loss}"
    device = _input.device

    if _input.dim() == 3:
        B, T, H = _input.shape
        _input = _input.view(-1, H)  # (B*T, H)로 reshape
        BT = B * T
    else:
        BT, H = _input.shape

    # target도 동일하게 처리
    if target.dim() == 2:
        target = target.view(-1)  # (B*T,)로 reshape

    V = weight.shape[0]
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    inc_factor = triton.cdiv(V, H)  # ceil(V/H)
    chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))  # ceil(BT/inc)
    num_chunks = triton.cdiv(BT, chunk_size)  # ceil(BT/chunksize)

    #grad 바로계산
    grad_weight = torch.zeros_like(weight, device=device) if weight.requires_grad else None
    grad_input = torch.zeros_like(_input, device=device)
    grad_bias = torch.zeros_like(bias, device=device) if bias is not None else None
    # we use fp32 for loss accumulator
    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)
    z_loss_1d = torch.zeros(BT, dtype=_input.dtype, device=_input.device) if return_z_loss else None

    target_mask = target != ignore_index
    total_n_non_ignore = target_mask.sum().item()
    total_sum_non_ignore_ce_weight = total_n_non_ignore
    ce_weight_sum = 0.0

    # CE_weight로 class별 가중치를 설정하는 부분
    if ce_weight is not None:
        assert ce_weight.shape[0] == V, f"If given, weight has to be a Tensor of size V. Got: {ce_weight.shape}"
        assert torch.is_floating_point(ce_weight), (
            f"If given, weight has to be a Tensor of floating point dtype. Got: {ce_weight.dtype}"
        )
        total_sum_non_ignore_ce_weight = (
            torch.gather(ce_weight, dim=0, index=target.masked_select(target_mask)).sum().item()
        )
        ce_weight_sum = ce_weight.sum().item()
        if ce_weight.stride(-1) != 1:
            ce_weight = ce_weight.contiguous()

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        _input_chunk = _input[start_idx:end_idx]

        # matmul
        logits_chunk = _input_chunk @ weight.t()
        if bias is not None:
            logits_chunk = logits_chunk + bias

        target_chunk = target[start_idx:end_idx]
        n_rows = logits_chunk.shape[0]

        # loss slicing
        loss_1d_slice = loss_1d[start_idx:end_idx]
        z_loss_1d_slice = z_loss_1d[start_idx:end_idx] if return_z_loss else None

        # input, target에 대해 row-major 순서로 메모리에 저장
        logits_chunk = logits_chunk.contiguous()
        target_chunk = target_chunk.contiguous()

        # CE kernel 사용 // logit -> X, target -> Y, weight -> weight, loss -> loss_1d_slice
        # in-place gradient calculating
        liger_cross_entropy_kernel[(n_rows,)](
            X_ptr=logits_chunk,
            X_stride=logits_chunk.stride(-2),
            Y_ptr=target_chunk,
            Y_stride=target_chunk.stride(-1),  # always 1
            weight_ptr=ce_weight,
            loss_ptr=loss_1d_slice,
            z_loss_ptr=z_loss_1d_slice,
            loss_stride=loss_1d_slice.stride(-1),  # always 1
            n_cols=V,
            n_non_ignore=total_n_non_ignore,
            sum_non_ignore_weight=total_sum_non_ignore_ce_weight,
            weight_sum=ce_weight_sum,
            ignore_index=ignore_index,
            lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing,
            reduction=reduction,
            softcap=softcap,
            RETURN_Z_LOSS=return_z_loss,
            HAS_WEIGHT=True if ce_weight is not None else False,
            HAS_SOFTCAPPING=True if softcap is not None else False,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 if not is_hip() else 16,
        )

        loss_1d[start_idx:end_idx] = loss_1d_slice
        if return_z_loss:
            z_loss_1d[start_idx:end_idx] = z_loss_1d_slice

        # grad_input
        # CE kernel에서 in-place로 logits_chunk 버퍼 위에 grad_logits_chunk가 저장되어 있음
        grad_logits_chunk = logits_chunk
        grad_input[start_idx:end_idx] = grad_logits_chunk @ weight

        if grad_weight is not None:
            torch.addmm(
                input=grad_weight,
                mat1=logits_chunk.t().to(
                    _input_chunk.dtype
                ), 
                mat2=_input_chunk,
                out=grad_weight,
                alpha=1.0,
                beta=1.0,
            )

        if bias is not None:
            torch.add(
                input=grad_bias,
                other=logits_chunk.sum(dim=0),
                out=grad_bias,
                alpha=1.0,
            )

    else:
        loss = torch.sum(loss_1d)
        z_loss = torch.sum(z_loss_1d) if return_z_loss else None
    return loss, z_loss, grad_input, grad_weight, grad_bias



#############################
######### Backward ##########
#############################

def fused_linear_cross_entropy_backward(grad_output, grad_input, grad_weight, grad_bias):
    if not torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        BT, H = grad_input.shape
        n_rows = BT
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))

        element_mul_kernel[(n_rows,)](
            grad_input,
            grad_input.stride(-2),
            grad_output,
            H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 if not is_hip() else 16,
        )

        if grad_weight is not None:
            V, H = grad_weight.shape
            n_rows = V

            element_mul_kernel[(n_rows,)](
                grad_weight,
                grad_weight.stride(-2),
                grad_output,
                H,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=32 if not is_hip() else 16,
            )

        if grad_bias is not None:
            V = grad_bias.shape[0]
            n_rows = V

            element_mul_kernel[(n_rows,)](
                grad_bias,
                grad_bias.stride(-1),
                grad_output,
                1,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=32 if not is_hip() else 16,
            )
    return grad_input, grad_weight, grad_bias

#############################
###### Module Replace #######
#############################


class LigerFusedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx, _input, weight, bias, target, ce_weight, ignore_index, lse_square_scale, label_smoothing, reduction, softcap, return_z_loss):
        loss, z_loss, grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_forward(
            _input=_input, weight=weight, target=target, bias=bias, ce_weight=ce_weight,
            ignore_index=ignore_index, lse_square_scale=lse_square_scale, label_smoothing=label_smoothing,
            reduction=reduction, softcap=softcap, return_z_loss=return_z_loss
        )
        tensors_to_save = [
            grad_input.detach(),
            grad_weight.detach() if grad_weight is not None else None,
            grad_bias.detach() if grad_bias is not None else None,
        ]

        ctx.save_for_backward(*tensors_to_save)
        ctx.return_z_loss = return_z_loss
        ctx.orig_shape = _input.shape
        return loss, z_loss

    @staticmethod
    @custom_fwd(device_type='cuda')
    def backward(ctx, grad_output, grad_output2):
        saved_grads = list(ctx.saved_tensors)
        grad_input_for_loss, grad_weight_for_loss, grad_bias_for_loss = saved_grads.pop(0), saved_grads.pop(0), saved_grads.pop(0)
        grad_input_for_loss = grad_input_for_loss.view(ctx.orig_shape)

        final_grad_input = grad_input_for_loss * grad_output
        final_grad_weight = grad_weight_for_loss * grad_output if grad_weight_for_loss is not None else None
        final_grad_bias = grad_bias_for_loss * grad_output if grad_bias_for_loss is not None else None

        if ctx.return_z_loss and grad_output2 is not None and grad_output2 != 0:
            grad_input_for_z, grad_weight_for_z = saved_grads.pop(0), saved_grads.pop(0)
            if grad_input_for_z is not None: final_grad_input.add_(grad_input_for_z * grad_output2)
            if grad_weight_for_z is not None and final_grad_weight is not None: final_grad_weight.add_(grad_weight_for_z * grad_output2)

        return final_grad_input, final_grad_weight, final_grad_bias, None, None, None, None, None, None, None, None


class LigerFusedLinearCrossEntropy(nn.Module):
    def __init__(self, config: LlamaConfig, bias: bool = False):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.empty(config.vocab_size, config.hidden_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(config.vocab_size))
        else:
            self.register_parameter('bias', None)

    def forward(self, hidden_states, labels=None, **kwargs):
        if labels is None:
            logits = F.linear(hidden_states, self.weight, self.bias)
            return None, logits

        fused_kwargs = {
            'ce_weight': None, 'ignore_index': -100, 'lse_square_scale': 0.0,
            'label_smoothing': 0.0, 'reduction': "mean", 'softcap': None,
            'return_z_loss': False
        }
        fused_kwargs.update(kwargs) # 외부에서 받은 인자로 덮어쓰기

        loss, z_loss = LigerFusedLinearCrossEntropyFunction.apply(
            hidden_states, self.weight, self.bias, labels,
            *fused_kwargs.values()
        )

        total_loss = loss + z_loss if z_loss is not None else loss
        logits = None

        return total_loss, logits


class LlamaForCausalLMLiger(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.lm_head = LigerFusedLinearCrossEntropy(config, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: torch.LongTensor = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:

        outputs = self.model(input_ids=input_ids, **kwargs)
        hidden_states = outputs[0]

        loss, logits = self.lm_head(hidden_states, labels=labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
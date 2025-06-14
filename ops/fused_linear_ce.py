"""
Fused Linear Cross Entropy 연산
Liger Kernel 방식을 직접 구현하여 최종 Linear 레이어와 Cross Entropy Loss를 융합
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple
import math

# ========================================
# Custom Triton Kernels (Liger 방식 직접 구현)
# ========================================

@triton.jit
def _cross_entropy_forward_kernel(
    logits_ptr,
    targets_ptr,
    loss_ptr,
    logits_stride,
    n_cols,
    ignore_index,
    label_smoothing: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Cross Entropy Forward Triton 커널"""
    row_idx = tl.program_id(0)
    
    # 현재 행의 시작 포인터
    logits_row_ptr = logits_ptr + row_idx * logits_stride
    target_idx = tl.load(targets_ptr + row_idx)
    
    # ignore_index 처리
    if target_idx == ignore_index:
        tl.store(loss_ptr + row_idx, 0.0)
        return
    
    # 열 인덱스
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # logits 로드
    logits = tl.load(logits_row_ptr + col_offsets, mask=mask, other=-float('inf'))
    
    # 수치적 안정성을 위한 최대값 빼기
    max_logits = tl.max(logits, axis=0)
    logits_shifted = logits - max_logits
    
    # softmax 계산
    exp_logits = tl.exp(logits_shifted)
    sum_exp = tl.sum(exp_logits, axis=0)
    
    # target에 해당하는 logit
    target_logit = tl.load(logits_row_ptr + target_idx)
    target_logit_shifted = target_logit - max_logits
    
    # Cross entropy loss = -log(softmax(target))
    loss = -(target_logit_shifted - tl.log(sum_exp))
    
    # Label smoothing 적용
    if label_smoothing > 0:
        smooth_loss = -tl.log(sum_exp) + tl.sum(logits_shifted * exp_logits) / sum_exp
        loss = (1 - label_smoothing) * loss + label_smoothing * smooth_loss / n_cols
    
    tl.store(loss_ptr + row_idx, loss)

@triton.jit  
def _cross_entropy_backward_kernel(
    grad_logits_ptr,
    logits_ptr, 
    targets_ptr,
    grad_logits_stride,
    logits_stride,
    n_cols,
    ignore_index,
    label_smoothing: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Cross Entropy Backward Triton 커널"""
    row_idx = tl.program_id(0)
    
    # 포인터 계산
    logits_row_ptr = logits_ptr + row_idx * logits_stride
    grad_row_ptr = grad_logits_ptr + row_idx * grad_logits_stride
    target_idx = tl.load(targets_ptr + row_idx)
    
    # ignore_index 처리
    if target_idx == ignore_index:
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        tl.store(grad_row_ptr + col_offsets, 0.0, mask=mask)
        return
    
    # 열 인덱스
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # logits 로드 및 softmax 계산
    logits = tl.load(logits_row_ptr + col_offsets, mask=mask, other=-float('inf'))
    max_logits = tl.max(logits, axis=0)
    logits_shifted = logits - max_logits
    exp_logits = tl.exp(logits_shifted)
    sum_exp = tl.sum(exp_logits, axis=0)
    softmax = exp_logits / sum_exp
    
    # gradient 계산: softmax - one_hot
    grad = softmax
    
    # target 위치의 gradient 조정
    target_mask = col_offsets == target_idx
    grad = tl.where(target_mask, grad - 1.0, grad)
    
    # Label smoothing 적용
    if label_smoothing > 0:
        uniform_prob = label_smoothing / n_cols
        grad = (1 - label_smoothing) * grad + uniform_prob
    
    tl.store(grad_row_ptr + col_offsets, grad, mask=mask)

# ========================================
# Python 래퍼 함수들 (Liger 방식)
# ========================================

def fused_linear_cross_entropy_forward(
    _input: torch.Tensor,
    weight: torch.Tensor, 
    target: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: str = "mean"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused Linear Cross Entropy Forward (Liger 방식 직접 구현)
    
    거대한 logits 텐서를 생성하지 않고 chunk 단위로 처리하여 메모리 절약
    """
    device = _input.device
    BT, H = _input.shape  # batch_size * seq_len, hidden_dim
    V = weight.shape[0]   # vocab_size
    
    # 메모리 효율을 위한 청크 크기 계산
    # 메모리 증가 = BT x V, 목표: BT x H와 비슷한 수준 유지
    inc_factor = triton.cdiv(V, H)  # (V + H - 1) // H
    chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))
    num_chunks = triton.cdiv(BT, chunk_size)
    
    # 결과 텐서들 초기화
    grad_input = torch.zeros_like(_input)
    grad_weight = torch.zeros_like(weight) if weight.requires_grad else None
    grad_bias = torch.zeros_like(bias) if bias is not None else None
    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)
    
    # 청크별 처리로 메모리 사용량 제한
    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, BT)
        
        # 현재 청크의 입력 데이터
        input_chunk = _input[start_idx:end_idx]  # [chunk_size, H]
        target_chunk = target[start_idx:end_idx]  # [chunk_size]
        
        # Linear 변환: input_chunk @ weight.T
        logits_chunk = input_chunk @ weight.t()  # [chunk_size, V]
        if bias is not None:
            logits_chunk = logits_chunk + bias
        
        # 현재 청크의 손실
        loss_chunk = loss_1d[start_idx:end_idx]
        
        # Triton 커널로 Cross Entropy 계산
        n_rows = logits_chunk.shape[0]
        BLOCK_SIZE = triton.next_power_of_2(V)
        
        _cross_entropy_forward_kernel[(n_rows,)](
            logits_ptr=logits_chunk,
            targets_ptr=target_chunk,
            loss_ptr=loss_chunk,
            logits_stride=logits_chunk.stride(0),
            n_cols=V,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Gradient 계산을 위해 logits_chunk를 재사용
        _cross_entropy_backward_kernel[(n_rows,)](
            grad_logits_ptr=logits_chunk,  # logits_chunk를 gradient로 덮어씀
            logits_ptr=logits_chunk,
            targets_ptr=target_chunk, 
            grad_logits_stride=logits_chunk.stride(0),
            logits_stride=logits_chunk.stride(0),
            n_cols=V,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # 이제 logits_chunk는 grad_logits
        grad_logits_chunk = logits_chunk
        
        # Input gradient: grad_logits @ weight
        grad_input[start_idx:end_idx] = grad_logits_chunk @ weight
        
        # Weight gradient: grad_logits.T @ input_chunk
        if grad_weight is not None:
            torch.addmm(grad_weight, grad_logits_chunk.t(), input_chunk, alpha=1.0, beta=1.0)
        
        # Bias gradient
        if bias is not None:
            grad_bias += grad_logits_chunk.sum(dim=0)
        
        # Loss 업데이트
        loss_1d[start_idx:end_idx] = loss_chunk
    
    # Reduction 적용
    if reduction == "mean":
        # ignore_index를 제외한 요소들의 평균
        valid_mask = target != ignore_index
        if valid_mask.sum() > 0:
            loss = loss_1d[valid_mask].mean()
        else:
            loss = torch.tensor(0.0, device=device)
    elif reduction == "sum":
        loss = loss_1d.sum()
    else:  # reduction == "none"
        loss = loss_1d
    
    return loss, grad_input, grad_weight, grad_bias

# ========================================
# PyTorch Function 클래스
# ========================================

class CustomFusedLinearCrossEntropyFunction(torch.autograd.Function):
    """직접 구현한 Fused Linear Cross Entropy Function"""
    
    @staticmethod
    def forward(
        ctx,
        _input: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = "mean"
    ):
        # Fused 계산 수행
        loss, grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_forward(
            _input, weight, target, bias, ignore_index, label_smoothing, reduction
        )
        
        # Backward를 위해 gradient 저장
        ctx.save_for_backward(
            grad_input.detach() if grad_input is not None else None,
            grad_weight.detach() if grad_weight is not None else None, 
            grad_bias.detach() if grad_bias is not None else None,
        )
        
        return loss
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight, grad_bias = ctx.saved_tensors
        
        # grad_output이 1.0이 아닌 경우 스케일링
        if not torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
            if grad_input is not None:
                grad_input = grad_input * grad_output
            if grad_weight is not None:
                grad_weight = grad_weight * grad_output
            if grad_bias is not None:
                grad_bias = grad_bias * grad_output
        
        return grad_input, grad_weight, None, grad_bias, None, None, None

# ========================================
# 모듈 클래스들
# ========================================

class NaiveLinearCrossEntropy(nn.Module):
    """표준 Linear + Cross Entropy 구현"""
    
    def __init__(self, input_dim: int, num_classes: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes, bias=bias)
        
    def forward(self, x, target, ignore_index=-100, label_smoothing=0.0, reduction="mean"):
        logits = self.linear(x)
        loss = F.cross_entropy(
            logits, target,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction=reduction
        )
        return loss, None  # z_loss는 None

class OptimizedLinearCrossEntropy(nn.Module):
    """직접 구현한 Fused Linear Cross Entropy"""
    
    def __init__(self, input_dim: int, num_classes: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, input_dim))
        self.bias = nn.Parameter(torch.randn(num_classes)) if bias else None
        
        # 가중치 초기화 (표준 방식)
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        
    def forward(self, x, target, ignore_index=-100, label_smoothing=0.0, reduction="mean"):
        loss = CustomFusedLinearCrossEntropyFunction.apply(
            x, self.weight, target, self.bias, ignore_index, label_smoothing, reduction
        )
        return loss, None  # z_loss는 None

def create_linear_cross_entropy(input_dim: int, num_classes: int, bias: bool = True, use_optimized: bool = False):
    """Linear Cross Entropy 생성 팩토리 함수"""
    if use_optimized:
        return OptimizedLinearCrossEntropy(input_dim, num_classes, bias)
    else:
        return NaiveLinearCrossEntropy(input_dim, num_classes, bias)

__all__ = [
    'NaiveLinearCrossEntropy',
    'OptimizedLinearCrossEntropy',
    'CustomFusedLinearCrossEntropyFunction',
    'create_linear_cross_entropy'
]
"""
GeGLU 연산 구현
- NaiveGeGLU: 표준 PyTorch 구현
- OptimizedGeGLU: Liger Kernel Triton 최적화 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton.language as tl
import triton

from ops.utils import calculate_settings, ensure_contiguous, tanh

# ========================================
# Triton 커널 구현
# ========================================

@triton.jit
def _geglu_tanh_forward_kernel(a, b, c, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """GeGLU forward Triton 커널"""
    program_id = tl.program_id(0).to(tl.int64)

    a += program_id * stride
    b += program_id * stride
    c += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b + col_offsets, mask=mask, other=0)

    # GELU tanh 근사: 0.5 * a * (1 + tanh(sqrt(2/pi) * (a + 0.044715 * a^3)))
    sqrt_2_over_pi = 0.7978845608028654
    a_cubed = a_row * a_row * a_row
    tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
    tanh_result = tanh(tanh_arg)
    geglu_a = 0.5 * a_row * (1 + tanh_result)
    c_row = geglu_a * b_row
    tl.store(c + col_offsets, c_row, mask=mask)

@triton.jit
def _geglu_tanh_backward_kernel(dc, a, b, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """GeGLU backward Triton 커널"""
    program_id = tl.program_id(0).to(tl.int64)

    dc += program_id * stride
    a += program_id * stride
    b += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dc_row = tl.load(dc + col_offsets, mask=mask, other=0)
    a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b + col_offsets, mask=mask, other=0)

    # 메모리 절약을 위한 재계산
    sqrt_2_over_pi = 0.7978845608028654
    a_cubed = a_row * a_row * a_row
    tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
    tanh_result = tanh(tanh_arg)
    geglu_a = 0.5 * a_row * (1 + tanh_result)

    db_row = dc_row * geglu_a

    # a에 대한 그래디언트 계산
    term1 = 0.5 * (1 + tanh_result)
    tanh_sq = tanh_result * tanh_result
    term2 = 0.5 * a_row * (1 - tanh_sq) * (sqrt_2_over_pi * (1 + 3 * 0.044715 * a_row * a_row))
    da_row = dc_row * b_row * (term1 + term2)

    tl.store(a + col_offsets, da_row, mask=mask)
    tl.store(b + col_offsets, db_row, mask=mask)

# ========================================
# Python 래퍼 함수들
# ========================================

def geglu_forward(a, b):
    """GeGLU forward pass"""
    ori_shape = a.shape
    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _geglu_tanh_forward_kernel[(n_rows,)](
        a, b, c, c.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return a, b, c.view(*ori_shape)

def geglu_backward(a, b, dc):
    """GeGLU backward pass"""
    ori_shape = dc.shape
    n_cols = ori_shape[-1]
    dc = dc.view(-1, n_cols)
    n_rows = dc.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _geglu_tanh_backward_kernel[(n_rows,)](
        dc, a, b, dc.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return a.view(*ori_shape), b.view(*ori_shape)

# ========================================
# PyTorch Function 클래스
# ========================================

class LigerGELUMulFunction(torch.autograd.Function):
    """Liger Kernel GeGLU Function"""
    
    @staticmethod
    @ensure_contiguous
    def forward(ctx, a, b):
        a, b, c = geglu_forward(a, b)
        ctx.save_for_backward(a, b)
        return c

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        a, b = geglu_backward(a, b, dc)
        return a, b

# ========================================
# 모듈 클래스들
# ========================================

class NaiveGeGLU(nn.Module):
    """표준 PyTorch GeGLU 구현"""
    
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim_in, dim_out, bias=False)
        self.up_proj = nn.Linear(dim_in, dim_out, bias=False)
        
    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return F.gelu(gate) * up

class OptimizedGeGLU(nn.Module):
    """Liger Kernel 최적화된 GeGLU 구현"""
    
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=False)
        
    def forward(self, x):
        gate_up = self.proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return LigerGELUMulFunction.apply(gate, up)

def create_geglu(dim_in: int, dim_out: int, use_optimized: bool = False):
    """GeGLU 생성 팩토리 함수"""
    if use_optimized:
        return OptimizedGeGLU(dim_in, dim_out)
    else:
        return NaiveGeGLU(dim_in, dim_out)

__all__ = [
    'NaiveGeGLU',
    'OptimizedGeGLU', 
    'LigerGELUMulFunction',
    'create_geglu'
]
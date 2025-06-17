"""
Layer Normalization 연산 구현
- NaiveLayerNorm: 표준 PyTorch 구현
- OptimizedLayerNorm: Triton 최적화 구현
"""

import math
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from ops.utils import calculate_settings, ensure_contiguous

# Triton 버전별 rsqrt import 처리
def _import_rsqrt():
    """Triton 버전에 따른 rsqrt 함수 import"""
    try:
        import importlib.metadata
        triton_version = importlib.metadata.version("triton")
        from packaging import version as pkg_version
        
        if pkg_version.parse(triton_version) >= pkg_version.parse("3.0.0"):
            try:
                from triton.language.extra.libdevice import rsqrt
            except ModuleNotFoundError:
                try:
                    from triton.language.extra.cuda.libdevice import rsqrt
                except ModuleNotFoundError:
                    from triton.language.math import rsqrt
        else:
            from triton.language.math import rsqrt
        return rsqrt
    except:
        # 기본값으로 math 모듈에서 import
        from triton.language.math import rsqrt
        return rsqrt

rsqrt = _import_rsqrt()

# ========================================
# Triton 커널 구현
# ========================================

@triton.jit
def _layer_norm_forward_kernel(
    Y_ptr,  # pointer to output, shape (n_rows, n_cols)
    Y_row_stride,  # stride of each row in output
    X_ptr,  # pointer to input, shape (n_rows, n_cols)
    X_row_stride,  # stride of each row in input
    W_ptr,  # pointer to weights, shape (n_cols,)
    W_row_stride,  # stride of each row in weights
    B_ptr,  # pointer to bias, shape (n_cols,)
    B_row_stride,  # stride of each row in bias
    Mean_ptr,  # pointer to mean, shape (n_rows,)
    Mean_row_stride,  # stride of each row in mean
    RSTD_ptr,  # pointer to rstd, shape (n_rows,)
    RSTD_row_stride,  # stride of each row in rstd
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Layer normalization forward kernel.
    References:
    https://arxiv.org/abs/1607.06450
    https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    Mean_ptr += row_idx * Mean_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)
    B_row = tl.load(B_ptr + col_offsets, mask=mask, other=0)

    mean = tl.sum(X_row, axis=0) / n_cols
    Xmm = tl.where(mask, X_row - mean, 0)
    var = tl.sum(Xmm * Xmm, axis=0) / n_cols
    rstd = rsqrt(var + eps)

    tl.store(Mean_ptr, mean)
    tl.store(RSTD_ptr, rstd)

    Y_row = Xmm * rstd * W_row + B_row

    tl.store(Y_ptr + col_offsets, Y_row, mask=mask)


@triton.jit
def _layer_norm_backward_kernel(
    X_ptr,  # pointer to input, shape (n_rows, n_cols)
    W_ptr,  # pointer to weights, shape (n_cols,)
    Mean_ptr,  # pointer to mean, shape (n_rows,)
    RSTD_ptr,  # pointer to rstd, shape (n_rows,)
    DX_ptr,  # pointer to input grad, shape (n_rows, n_cols)
    DW_ptr,  # pointer to weights grad, shape (n_cols,)
    DB_ptr,  # pointer to bias grad, shape (n_cols,)
    DY_ptr,  # pointer to output grad, shape (n_rows, n_cols)
    stride_x,  # stride of each row in input
    stride_dx,  # stride of each row in input grad
    stride_dw,  # stride of each row in weights grad
    stride_db,  # stride of each row in bias grad
    stride_dy,  # stride of each row in output grad
    n_rows,
    n_cols,
    rows_per_program: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    dtype: tl.constexpr,
):
    """
    Layer normalization backward kernel.
    References:
    https://arxiv.org/abs/1607.06450
    https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md
    https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/triton/layer_norm.py
    """
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    row_end = min((row_block_id + 1) * rows_per_program, n_rows)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    dw_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    db_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    X_ptr += row_start * stride_x
    Mean_ptr += row_start
    RSTD_ptr += row_start
    DX_ptr += row_start * stride_dx
    DY_ptr += row_start * stride_dy

    for _ in range(row_start, row_end):
        x = tl.load(X_ptr + cols, mask=mask, other=0.0)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0)
        dy = tl.load(DY_ptr + cols, mask=mask, other=0.0)
        mean = tl.load(Mean_ptr)
        rstd = tl.load(RSTD_ptr)

        x_hat = (x - mean) * rstd
        wdy = w * dy
        c1 = tl.sum(x_hat * wdy, axis=0) / n_cols
        c2 = tl.sum(wdy, axis=0) / n_cols
        dx = (wdy - (x_hat * c1 + c2)) * rstd
        tl.store(DX_ptr + cols, dx.to(dtype), mask=mask)

        dw_row += dy * x_hat
        db_row += dy

        X_ptr += stride_x
        Mean_ptr += 1
        RSTD_ptr += 1
        DX_ptr += stride_dx
        DY_ptr += stride_dy

    tl.store(DW_ptr + row_block_id * stride_dw + cols, dw_row.to(dtype), mask=mask)
    tl.store(DB_ptr + row_block_id * stride_db + cols, db_row.to(dtype), mask=mask)


# ========================================
# Python 래퍼 함수들
# ========================================

def layer_norm_forward(X, W, B, eps):
    """Layer normalization forward pass"""
    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    n_rows, n_cols = X.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    Mean = torch.empty(n_rows, dtype=X.dtype, device=X.device)
    RSTD = torch.empty(n_rows, dtype=X.dtype, device=X.device)
    
    if X.shape[1] != W.shape[0]:
        raise ValueError(
            f"Incompatible dimensions: input feature size (X.shape[1]={X.shape[1]}) "
            f"must match weight size (W.shape[0]={W.shape[0]})"
        )

    # Device-specific optimization
    kernel_args = {}
    if X.device.type == "xpu":
        kernel_args["grf_mode"] = "large"

    _layer_norm_forward_kernel[(n_rows,)](
        Y,
        Y.stride(0),
        X,
        X.stride(0),
        W,
        W.stride(0),
        B,
        B.stride(0),
        Mean,
        Mean.stride(0),
        RSTD,
        RSTD.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        **kernel_args,
    )
    return Y.view(*shape), X, Mean, RSTD, BLOCK_SIZE, num_warps


def layer_norm_backward(dY, X, W, B, Mean, RSTD):
    """Layer normalization backward pass"""
    shape = dY.shape
    dim = shape[-1]
    dY = dY.view(-1, dim)
    n_rows, n_cols = dY.shape

    # Get device-specific multiprocessor count
    sm_count = 1
    if X.device.type == "cuda":
        sm_count = torch.cuda.get_device_properties(X.device).multi_processor_count
    elif X.device.type == "xpu":
        sm_count = torch.xpu.get_device_properties(X.device).gpu_eu_count

    DX = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    _DW = torch.empty((sm_count, n_cols), dtype=W.dtype, device=W.device)
    _DB = torch.empty((sm_count, n_cols), dtype=W.dtype, device=W.device)

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    if n_cols > BLOCK_SIZE:
        raise RuntimeError(
            f"Feature dimension {n_cols} exceeds maximum supported size of {BLOCK_SIZE}. "
            f"Consider using a smaller feature dimension."
        )

    rows_per_program = math.ceil(n_rows / sm_count)
    grid = (sm_count,)
    
    # Map PyTorch dtype to Triton dtype
    triton_dtype = (
        tl.float32 if X.dtype == torch.float32
        else tl.bfloat16 if X.dtype == torch.bfloat16
        else tl.float16 if X.dtype == torch.float16
        else tl.float32  # fallback to float32 for other types
    )

    # Device-specific optimization
    kernel_args = {}
    if X.device.type == "xpu":
        kernel_args.update({"grf_mode": "large", "num_warps": 32, "num_stages": 4})

    _layer_norm_backward_kernel[grid](
        X,
        W,
        Mean,
        RSTD,
        DX,
        _DW,
        _DB,
        dY,
        X.stride(0),
        DX.stride(0),
        _DW.stride(0),
        _DB.stride(0),
        dY.stride(0),
        n_rows,
        n_cols,
        rows_per_program,
        BLOCK_SIZE=BLOCK_SIZE,
        dtype=triton_dtype,
        **kernel_args,
    )

    DW = _DW.sum(dim=0).to(W.dtype)
    DB = _DB.sum(dim=0).to(W.dtype)

    DX = DX.view(*shape)
    return DX, DW, DB


# ========================================
# PyTorch Function 클래스
# ========================================

class LigerLayerNormFunction(torch.autograd.Function):
    """Liger Kernel Layer Normalization Function"""
    
    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, W, B, eps):
        Y, X, Mean, RSTD, BLOCK_SIZE, num_warps = layer_norm_forward(X, W, B, eps)
        ctx.save_for_backward(X, W, B, Mean, RSTD)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        X, W, B, Mean, RSTD = ctx.saved_tensors
        DX, DW, DB = layer_norm_backward(dY, X, W, B, Mean, RSTD)
        return DX, DW, DB, None


# ========================================
# 모듈 클래스들
# ========================================

class NaiveLayerNorm(nn.Module):
    """표준 PyTorch Layer Normalization 구현"""
    
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
    
    def forward(self, input):
        """Forward pass using PyTorch's implementation"""
        return F.layer_norm(input, (self.normalized_shape,), self.weight, self.bias, self.eps)
    
    def extra_repr(self):
        return f'{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


class OptimizedLayerNorm(nn.Module):
    """Triton 최적화된 Layer Normalization 구현"""
    
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
    
    def forward(self, input):
        """Forward pass using Triton optimized implementation"""
        if self.elementwise_affine:
            return LigerLayerNormFunction.apply(input, self.weight, self.bias, self.eps)
        else:
            # Fallback to PyTorch's implementation when no learnable parameters
            return F.layer_norm(input, (self.normalized_shape,), None, None, self.eps)
    
    def extra_repr(self):
        return f'{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


def create_layer_norm(normalized_shape, eps=1e-5, elementwise_affine=True, 
                     use_optimized=False, device=None, dtype=None):
    """Layer Normalization 생성 팩토리 함수"""
    if use_optimized:
        return OptimizedLayerNorm(normalized_shape, eps, elementwise_affine, device, dtype)
    else:
        return NaiveLayerNorm(normalized_shape, eps, elementwise_affine, device, dtype)


# 함수형 인터페이스
def layer_norm(input, weight, bias, eps=1e-5):
    """
    Functional interface for layer normalization using Triton optimization.
    
    Args:
        input: Input tensor
        weight: Weight tensor
        bias: Bias tensor  
        eps: Small value for numerical stability
        
    Returns:
        Normalized tensor
    """
    return LigerLayerNormFunction.apply(input, weight, bias, eps)


__all__ = [
    'NaiveLayerNorm',
    'OptimizedLayerNorm',
    'LigerLayerNormFunction',
    'create_layer_norm',
    'layer_norm'
]
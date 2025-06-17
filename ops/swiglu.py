"""
SwiGLU 연산 구현
- NaiveSwiGLU: 표준 PyTorch 구현
- OptimizedSwiGLU: Triton 최적화 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton.language as tl
import triton

from ops.utils import calculate_settings, ensure_contiguous

# ========================================
# Triton 커널 구현
# ========================================

@triton.jit
def silu(x):
	"""SiLU (Swish) 활성화 함수: x * sigmoid(x)"""
	return x * tl.sigmoid(x)


@triton.jit
def _swiglu_forward_kernel(a_ptr, b_ptr, c_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
	"""SwiGLU forward Triton 커널"""
	program_id = tl.program_id(0).to(tl.int64)

	# 시작 인덱스 위치 설정
	a_ptr += program_id * stride
	b_ptr += program_id * stride
	c_ptr += program_id * stride

	col_offsets = tl.arange(0, BLOCK_SIZE)
	mask = col_offsets < n_cols

	# sigmoid는 float32 타입이 필요
	a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
	b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
		
	# SwiGLU: SiLU(a) * b = (a * sigmoid(a)) * b
	c_row = silu(a_row) * b_row
	tl.store(c_ptr + col_offsets, c_row, mask=mask)


@triton.jit
def _swiglu_backward_kernel(dc_ptr, a_ptr, b_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
	"""SwiGLU backward Triton 커널"""
	program_id = tl.program_id(0).to(tl.int64)

	# 시작 인덱스 위치 설정
	dc_ptr += program_id * stride
	a_ptr += program_id * stride
	b_ptr += program_id * stride

	col_offsets = tl.arange(0, BLOCK_SIZE)
	mask = col_offsets < n_cols

	dc_row = tl.load(dc_ptr + col_offsets, mask=mask, other=0)
	# sigmoid는 float32 타입이 필요
	a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
	b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)

	# 메모리 절약을 위한 재계산
	sig_a = tl.sigmoid(a_row)
	silu_a = a_row * sig_a
		
	# 그래디언트 계산
	db_row = dc_row * silu_a
	# SiLU의 도함수: d/dx[x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
	#									= sigmoid(x) * (1 + x * (1 - sigmoid(x)))
	#									= silu(x) * (1 - sigmoid(x)) + sigmoid(x)
	da_row = dc_row * (silu_a * (1 - sig_a) + sig_a) * b_row

	tl.store(a_ptr + col_offsets, da_row, mask=mask)
	tl.store(b_ptr + col_offsets, db_row, mask=mask)


# ========================================
# Python 래퍼 함수들
# ========================================

def swiglu_forward(a, b):
	"""SwiGLU forward pass"""
	ori_shape = a.shape
	n_cols = ori_shape[-1]
	a = a.view(-1, n_cols)
	b = b.view(-1, n_cols)
	c = torch.empty_like(a)
	n_rows = a.shape[0]

	BLOCK_SIZE, num_warps = calculate_settings(n_cols)

	_swiglu_forward_kernel[(n_rows,)](
		a, b, c, c.stride(-2),
		n_cols=n_cols,
		BLOCK_SIZE=BLOCK_SIZE,
		num_warps=num_warps,
	)
	return a, b, c.view(*ori_shape)


def swiglu_backward(a, b, dc):
	"""SwiGLU backward pass"""
	ori_shape = dc.shape
	n_cols = ori_shape[-1]
	dc = dc.view(-1, n_cols)
	n_rows = dc.shape[0]

	BLOCK_SIZE, num_warps = calculate_settings(n_cols)

	_swiglu_backward_kernel[(n_rows,)](
		dc, a, b, dc.stride(-2),
		n_cols=n_cols,
		BLOCK_SIZE=BLOCK_SIZE,
		num_warps=num_warps,
	)
	return a.view(*ori_shape), b.view(*ori_shape)


# ========================================
# PyTorch Function 클래스
# ========================================

class LigerSiLUMulFunction(torch.autograd.Function):
	"""Liger Kernel SwiGLU Function"""
		
	@staticmethod
	@ensure_contiguous
	def forward(ctx, a, b):
		a, b, c = swiglu_forward(a, b)
		ctx.save_for_backward(a, b)
		return c

	@staticmethod
	@ensure_contiguous
	def backward(ctx, dc):
		a, b = ctx.saved_tensors
		a, b = swiglu_backward(a, b, dc)
		return a, b


# ========================================
# 모듈 클래스들
# ========================================

class NaiveSwiGLU(nn.Module):
	"""표준 PyTorch SwiGLU 구현"""
		
	def __init__(self, dim_in: int, dim_out: int):
		super().__init__()
		self.gate_proj = nn.Linear(dim_in, dim_out, bias=False)
		self.up_proj = nn.Linear(dim_in, dim_out, bias=False)
		
	def forward(self, x):
		gate = self.gate_proj(x)
		up = self.up_proj(x)
		# SwiGLU: SiLU(gate) * up = (gate * sigmoid(gate)) * up
		return F.silu(gate) * up


class OptimizedSwiGLU(nn.Module):
	"""Triton 최적화된 SwiGLU 구현"""
		
	def __init__(self, dim_in: int, dim_out: int):
		super().__init__()
		self.proj = nn.Linear(dim_in, dim_out * 2, bias=False)
		
	def forward(self, x):
		gate_up = self.proj(x)
		gate, up = gate_up.chunk(2, dim=-1)
		return LigerSiLUMulFunction.apply(gate, up)


def create_swiglu(dim_in: int, dim_out: int, use_optimized: bool = False):
	"""SwiGLU 생성 팩토리 함수"""
	if use_optimized:
		return OptimizedSwiGLU(dim_in, dim_out)
	else:
		return NaiveSwiGLU(dim_in, dim_out)


__all__ = [
	'NaiveSwiGLU',
	'OptimizedSwiGLU', 
	'LigerSiLUMulFunction',
	'create_swiglu'
]
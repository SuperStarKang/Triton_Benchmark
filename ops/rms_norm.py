"""
RMS Norm 연산 구현
- NaiveRMSNorm: 표준 PyTorch 구현
- OptimizedRMSNorm: Triton 최적화 구현
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

from ops.utils import calculate_settings, ensure_contiguous

# ========================================
# Triton 커널 구현
# ========================================

@triton.jit
def _rms_norm_forward_kernel(
	Y_ptr,
	X_ptr,
	W_ptr,
	RSTD_ptr,
	stride,
	n_cols,
	eps,
	offset,
	BLOCK_SIZE: tl.constexpr,
):
	"""RMS Norm forward Triton 커널"""
	program_id = tl.program_id(0).to(tl.int64)
		
	# 포인터 오프셋 계산
	Y_ptr += program_id * stride
	X_ptr += program_id * stride
	RSTD_ptr += program_id
		
	col_offsets = tl.arange(0, BLOCK_SIZE)
	mask = col_offsets < n_cols
		
	# 입력 로드
	X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
	W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
		
	# RMS 계산: sqrt(mean(x^2))
	mean_square = tl.sum(X_row * X_row, axis=0) / n_cols
	# Reciprocal standard deviation: 1 / sqrt(mean_square + eps)
	rstd = 1.0 / tl.sqrt(mean_square + eps)
		
	# RSTD 저장 (캐싱용)
	tl.store(RSTD_ptr, rstd)
		
	# 정규화 및 스케일링: (x / rms) * (offset + w)
	normalized = X_row * rstd
	Y_row = normalized * (offset + W_row)
		
	tl.store(Y_ptr + col_offsets, Y_row, mask=mask)


@triton.jit  
def _rms_norm_backward_kernel(
	dY_ptr,
	dX_ptr, 
	X_ptr,
	W_ptr,
	RSTD_ptr,
	dW_ptr,
	stride,
	n_cols,
	offset,
	BLOCK_SIZE: tl.constexpr,
):
	"""RMS Norm backward Triton 커널"""
	program_id = tl.program_id(0).to(tl.int64)
		
	# 포인터 오프셋 계산
	dY_ptr += program_id * stride
	dX_ptr += program_id * stride
	X_ptr += program_id * stride
	RSTD_ptr += program_id
		
	col_offsets = tl.arange(0, BLOCK_SIZE)
	mask = col_offsets < n_cols
		
	# 입력 로드
	dY_row = tl.load(dY_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
	X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
	W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
	rstd = tl.load(RSTD_ptr)
		
	# Weight gradient: dW = sum(dY * (X * rstd))
	normalized_X = X_row * rstd
	dW_row = dY_row * normalized_X
		
	# Input gradient 계산
	# dx = (1/rms) * [dy * (w + offset) - (1/N) * (1/rms^2) * sum(dy * (w + offset) * x) * x]
	weighted_dy = dY_row * (W_row + offset)
		
	# 첫 번째 항: dy * (w + offset) * rstd
	term1 = weighted_dy * rstd
		
	# 두 번째 항: -(1/N) * rstd^3 * sum(weighted_dy * x) * x
	sum_term = tl.sum(weighted_dy * X_row, axis=0)
	term2 = -(1.0 / n_cols) * rstd * rstd * rstd * sum_term * X_row
		
	dX_row = term1 + term2
		
	# 결과 저장
	tl.store(dX_ptr + col_offsets, dX_row, mask=mask)
	tl.store(dW_ptr + program_id * n_cols + col_offsets, dW_row, mask=mask)


# ========================================
# Python 래퍼 함수들
# ========================================

def rms_norm_forward(X, W, eps, offset):
	"""RMS Norm forward pass"""
	ori_shape = X.shape
	n_cols = ori_shape[-1]
	X = X.view(-1, n_cols)
	n_rows = X.shape[0]
		
	# 출력 텐서 생성
	Y = torch.empty_like(X)
	RSTD = torch.empty(n_rows, dtype=torch.float32, device=X.device)
		
	BLOCK_SIZE, num_warps = calculate_settings(n_cols)
		
	_rms_norm_forward_kernel[(n_rows,)](
		Y, X, W, RSTD,
		stride=X.stride(-2),
		n_cols=n_cols,
		eps=eps,
		offset=offset,
		BLOCK_SIZE=BLOCK_SIZE,
		num_warps=num_warps,
	)
		
	return Y.view(*ori_shape), X, RSTD


def rms_norm_backward(dY, X, W, RSTD, eps, offset):
	"""RMS Norm backward pass"""
	ori_shape = dY.shape
	n_cols = ori_shape[-1]
	dY = dY.view(-1, n_cols)
	n_rows = dY.shape[0]
		
	# 출력 텐서 생성
	dX = torch.empty_like(dY)
	dW = torch.empty(n_rows, n_cols, dtype=torch.float32, device=W.device)
		
	BLOCK_SIZE, num_warps = calculate_settings(n_cols)
		
	_rms_norm_backward_kernel[(n_rows,)](
		dY, dX, X, W, RSTD, dW,
		stride=dY.stride(-2),
		n_cols=n_cols,
		offset=offset,
		BLOCK_SIZE=BLOCK_SIZE,
		num_warps=num_warps,
	)
		
	# Weight gradient 합산
	dW_final = dW.sum(dim=0).to(W.dtype)
		
	return dX.view(*ori_shape), dW_final


# ========================================
# PyTorch Function 클래스
# ========================================

class LigerRMSNormFunction(torch.autograd.Function):
	"""Liger Kernel RMS Norm Function"""
		
	@staticmethod
	@ensure_contiguous
	def forward(ctx, X, W, eps=1e-6, offset=0.0):
		Y, X, RSTD = rms_norm_forward(X, W, eps, offset)
		ctx.eps = eps
		ctx.offset = offset
		ctx.save_for_backward(X, W, RSTD)
		return Y

	@staticmethod
	@ensure_contiguous
	def backward(ctx, dY):
		X, W, RSTD = ctx.saved_tensors
		dX, dW = rms_norm_backward(dY, X, W, RSTD, ctx.eps, ctx.offset)
		return dX, dW, None, None


# ========================================
# 모듈 클래스들
# ========================================

class NaiveRMSNorm(nn.Module):
	"""표준 PyTorch RMS Norm 구현"""
		
	def __init__(self, hidden_size: int, eps: float = 1e-6, offset: float = 0.0):
		super().__init__()
		self.weight = nn.Parameter(torch.ones(hidden_size))
		self.eps = eps
		self.offset = offset
		
	def forward(self, x):
		# RMS 계산
		variance = x.pow(2).mean(dim=-1, keepdim=True)
		x_normalized = x * torch.rsqrt(variance + self.eps)
		
		# 스케일링
		return x_normalized * (self.offset + self.weight)


class OptimizedRMSNorm(nn.Module):
	"""Triton 최적화된 RMS Norm 구현"""
		
	def __init__(self, hidden_size: int, eps: float = 1e-6, offset: float = 0.0):
		super().__init__()
		self.weight = nn.Parameter(torch.ones(hidden_size))
		self.eps = eps
		self.offset = offset
		
	def forward(self, x):
		return LigerRMSNormFunction.apply(x, self.weight, self.eps, self.offset)


def create_rms_norm(hidden_size: int, eps: float = 1e-6, offset: float = 0.0, use_optimized: bool = False):
	"""RMS Norm 생성 팩토리 함수"""
	if use_optimized:
		return OptimizedRMSNorm(hidden_size, eps, offset)
	else:
		return NaiveRMSNorm(hidden_size, eps, offset)


__all__ = [
	'NaiveRMSNorm',
	'OptimizedRMSNorm',
	'LigerRMSNormFunction', 
	'create_rms_norm'
]
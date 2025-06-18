"""
간단한 언어 모델 아키텍처
SwiGLU와 Fused Linear Cross Entropy를 사용하는 Transformer 기반 모델
"""

import torch
import torch.nn as nn
from ops.swiglu import create_swiglu
from ops.fused_linear_ce import create_linear_cross_entropy


class SimpleTransformerBlock(nn.Module):
	"""
		SwiGLU를 사용하는 간단한 Transformer 블록
		Input x
		│
		├─ LayerNorm (LN1)
		├─ MultiheadAttention (Self-Attn) → + Residual (x₁ = x + Attn(x))
		│
		├─ LayerNorm (LN2)
		├─ SwiGLU-MLP → Projection (필요시) → + Residual (x₂ = x₁ + MLP(x₁))
		│
		└─ Output
	"""

	def __init__(self, hidden_size: int, intermediate_size: int, use_optimized: bool = True):
		super().__init__()
		self.hidden_size = hidden_size
		self.intermediate_size = intermediate_size

		# Layer Normalization
		self.input_layernorm = nn.LayerNorm(hidden_size)
		self.post_attention_layernorm = nn.LayerNorm(hidden_size)

		# Self-attention (간단화된 버전)
		self.self_attn = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)

		# SwiGLU MLP
		self.mlp = create_swiglu(hidden_size, intermediate_size, use_optimized=use_optimized)
		
	def forward(self, x):
		# Self-attention with residual connection
		residual = x
		x = self.input_layernorm(x)
		attn_output, _ = self.self_attn(x, x, x)
		x = residual + attn_output
		
		# MLP with residual connection
		residual = x
		x = self.post_attention_layernorm(x)
		mlp_output = self.mlp(x)
		# SwiGLU 출력 차원이 intermediate_size이므로 hidden_size로 projection 필요
		if mlp_output.shape[-1] != residual.shape[-1]:
			# 차원 맞추기 위한 projection layer 추가
			if not hasattr(self, 'down_proj'):
				self.down_proj = nn.Linear(mlp_output.shape[-1], residual.shape[-1], bias=False).to(mlp_output.device)
			mlp_output = self.down_proj(mlp_output)
		x = residual + mlp_output
		
		return x


class SimpleLanguageModel(nn.Module):
	"""
		SwiGLU + Fused Linear Cross Entropy를 사용하는 간단한 언어 모델
		Input → Token Embedding → [TransformerBlock * N] → LayerNorm → LM Head → (Loss or Logits)
	"""

	def __init__(
		self,
		vocab_size: int = 32000,
		hidden_size: int = 768,
		intermediate_size: int = 2048,
		num_layers: int = 2,
		use_optimized: bool = False
	):
		super().__init__()
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.use_optimized = use_optimized
		
		# Token Embedding
		self.token_embedding = nn.Embedding(vocab_size, hidden_size)
		
		# Transformer Layers
		self.layers = nn.ModuleList([
			SimpleTransformerBlock(hidden_size, intermediate_size, use_optimized)
			for _ in range(num_layers)
		])
		
		# Final Layer Norm
		self.final_layernorm = nn.LayerNorm(hidden_size)
		
		# Language Model Head (Fused Linear Cross Entropy)
		self.lm_head = create_linear_cross_entropy(
			hidden_size, vocab_size, bias=False, use_optimized=use_optimized
		)
		
		# Weight initialization
		self._init_weights()
		
	def _init_weights(self):
		"""가중치 초기화"""
		for module in self.modules():
			if isinstance(module, nn.Embedding):
				nn.init.normal_(module.weight, mean=0.0, std=0.02)
			elif isinstance(module, nn.Linear):
				nn.init.normal_(module.weight, mean=0.0, std=0.02)
				if module.bias is not None:
					nn.init.zeros_(module.bias)
			elif isinstance(module, nn.LayerNorm):
				nn.init.ones_(module.weight)
				nn.init.zeros_(module.bias)
		
	def forward(self, input_ids, targets=None):
		# Token embedding
		x = self.token_embedding(input_ids)
		
		# Transformer layers
		for layer in self.layers:
			x = layer(x)
		
		# Final layer norm
		x = self.final_layernorm(x)
		
		# Language model head
		if targets is not None:
			# 학습 모드: loss 계산
			# x를 flatten: [batch_size, seq_len, hidden_size] -> [batch_size * seq_len, hidden_size]
			x_flat = x.view(-1, self.hidden_size)
			targets_flat = targets.view(-1)
			
			loss, _ = self.lm_head(x_flat, targets_flat)
			return loss
		else:
			# 추론 모드: logits 반환
			# 마지막 토큰의 logits만 계산 (메모리 절약)
			x_last = x[:, -1, :]  # [batch_size, hidden_size]
			
			# 간단한 Linear layer로 logits 계산 (추론용)
			if hasattr(self.lm_head, 'linear'):
				logits = self.lm_head.linear(x_last)
			else:
				# OptimizedLinearCrossEntropy의 경우
				logits = x_last @ self.lm_head.weight.t()
				if self.lm_head.bias is not None:
					logits = logits + self.lm_head.bias
			
			return logits
		
	def get_num_parameters(self):
		"""모델 파라미터 수 반환"""
		return sum(p.numel() for p in self.parameters())
		
	def get_model_size_mb(self):
		"""모델 크기 반환 (MB)"""
		return self.get_num_parameters() * 4 / (1024**2)  # float32 기준


def create_simple_language_model(
	vocab_size: int = 32000,
	hidden_size: int = 768,
	intermediate_size: int = 2048,
	num_layers: int = 2,
	use_optimized: bool = False
):
	"""간단한 언어 모델 생성 팩토리 함수"""
	return SimpleLanguageModel(
		vocab_size=vocab_size,
		hidden_size=hidden_size,
		intermediate_size=intermediate_size,
		num_layers=num_layers,
		use_optimized=use_optimized
	)


__all__ = [
	'SimpleTransformerBlock',
	'SimpleLanguageModel',
	'create_simple_language_model'
]
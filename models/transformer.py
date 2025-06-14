"""
최적화된 연산들을 사용하는 Transformer 모델 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from ops import create_geglu, create_linear_cross_entropy


class MultiHeadAttention(nn.Module):
    """멀티헤드 어텐션 레이어"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        # QKV 프로젝션
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, d_model = x.shape
        
        # QKV 계산
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 어텐션 스코어 계산
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 마스킹 적용
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 소프트맥스 및 드롭아웃
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 어텐션 적용
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Feed Forward Network with GeGLU activation"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, use_optimized: bool = False):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # GeGLU 활성화 함수 사용
        self.activation = create_geglu(d_model, d_ff, use_optimized=use_optimized)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(d_ff, d_model, bias=False)
        
    def forward(self, x: torch.Tensor):
        # GeGLU 활성화
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output_proj(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer 블록"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, use_optimized: bool = False):
        super().__init__()
        
        # 레이어 정규화
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # 어텐션과 피드포워드
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout, use_optimized)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Pre-norm 방식
        # 어텐션 블록
        residual = x
        x = self.ln1(x)
        x = self.attention(x, mask)
        x = x + residual
        
        # 피드포워드 블록
        residual = x
        x = self.ln2(x)
        x = self.feed_forward(x)
        x = x + residual
        
        return x


class OptimizedTransformer(nn.Module):
    """최적화된 연산들을 사용하는 Transformer 모델"""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        use_optimized: bool = False
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.use_optimized = use_optimized
        
        # 임베딩 레이어
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer 블록들
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, use_optimized)
            for _ in range(n_layers)
        ])
        
        # 최종 레이어 정규화
        self.ln_final = nn.LayerNorm(d_model)
        
        # 출력 헤드 (Fused Linear Cross Entropy 사용)
        self.output_head = create_linear_cross_entropy(
            d_model, vocab_size, bias=False, use_optimized=use_optimized
        )
        
        # 가중치 초기화
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """가중치 초기화"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None, **kwargs):
        """
        Forward pass
        
        Args:
            input_ids: [batch_size, seq_len]
            targets: [batch_size, seq_len] (for training)
            
        Returns:
            loss or logits depending on targets
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 위치 인덱스 생성
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # 임베딩
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(pos_ids)
        x = token_emb + pos_emb
        
        # Causal mask 생성 (decoder-only)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
        
        # Transformer 블록들 통과
        for block in self.blocks:
            x = block(x, mask)
        
        # 최종 정규화
        x = self.ln_final(x)
        
        # 출력 계산
        if targets is not None:
            # 학습 모드: loss 계산
            # x를 (batch_size * seq_len, d_model)로 reshape
            x_flat = x.view(-1, self.d_model)
            targets_flat = targets.view(-1)
            
            loss, _ = self.output_head(
                x_flat, targets_flat,
                ignore_index=-100,
                label_smoothing=0.0,
                reduction="mean"
            )
            return loss
        else:
            # 추론 모드: logits 반환
            # 단순한 linear projection (Cross Entropy 없이)
            if hasattr(self.output_head, 'weight'):
                # OptimizedLinearCrossEntropy의 경우
                logits = F.linear(x, self.output_head.weight, self.output_head.bias)
            else:
                # NaiveLinearCrossEntropy의 경우  
                logits = self.output_head.linear(x)
            return logits
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 100, temperature: float = 1.0):
        """간단한 자동회귀 생성"""
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_length):
                # 현재까지의 시퀀스로 다음 토큰 예측
                logits = self.forward(input_ids)
                
                # 마지막 위치의 logits만 사용
                next_token_logits = logits[:, -1, :] / temperature
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), 1)
                
                # 새로운 토큰 추가
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
        return input_ids


class NaiveTransformer(OptimizedTransformer):
    """표준 구현을 사용하는 Transformer (비교용)"""
    
    def __init__(self, *args, **kwargs):
        # use_optimized=False로 고정
        kwargs['use_optimized'] = False
        super().__init__(*args, **kwargs)


def create_transformer(
    vocab_size: int,
    d_model: int = 512,
    n_heads: int = 8, 
    n_layers: int = 6,
    d_ff: int = 2048,
    max_seq_len: int = 1024,
    dropout: float = 0.1,
    use_optimized: bool = False
):
    """Transformer 생성 팩토리 함수"""
    return OptimizedTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout,
        use_optimized=use_optimized
    )


__all__ = [
    'MultiHeadAttention',
    'FeedForward', 
    'TransformerBlock',
    'OptimizedTransformer',
    'NaiveTransformer',
    'create_transformer'
]
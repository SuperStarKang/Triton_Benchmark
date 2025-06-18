"""
간단한 언어 모델 아키텍처 (RMSNorm 적용)
SwiGLU, RMSNorm, Fused Linear Cross Entropy를 사용하는 Transformer 기반 모델
"""

import torch
import torch.nn as nn
from ops.swiglu import create_swiglu
from ops.fused_linear_ce import create_linear_cross_entropy
from ops.rms_norm import create_rms_norm


class SimpleTransformerBlock(nn.Module):
    """
    RMSNorm과 SwiGLU를 사용하는 간단한 Transformer 블록
    Input x
    │
    ├─ RMSNorm (RN1)
    ├─ MultiheadAttention (Self-Attn) → + Residual (x₁ = x + Attn(x))
    │
    ├─ RMSNorm (RN2)
    ├─ SwiGLU-MLP → Projection (필요시) → + Residual (x₂ = x₁ + MLP(x₁))
    │
    └─ Output
    """

    def __init__(
        self, 
        hidden_size: int, 
        intermediate_size: int, 
        use_optimized: bool = True,
        rms_norm_eps: float = 1e-6,
        rms_norm_offset: float = 0.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # RMS Normalization (LayerNorm 대신)
        self.input_rmsnorm = create_rms_norm(
            hidden_size, 
            eps=rms_norm_eps, 
            offset=rms_norm_offset, 
            use_optimized=use_optimized
        )
        self.post_attention_rmsnorm = create_rms_norm(
            hidden_size, 
            eps=rms_norm_eps, 
            offset=rms_norm_offset, 
            use_optimized=use_optimized
        )

        # Self-attention (간단화된 버전)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)

        # SwiGLU MLP
        self.mlp = create_swiglu(hidden_size, intermediate_size, use_optimized=use_optimized)
        
        # Down projection for dimension matching
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
    def forward(self, x):
        # Self-attention with residual connection
        residual = x
        x = self.input_rmsnorm(x)
        attn_output, _ = self.self_attn(x, x, x)
        x = residual + attn_output
        
        # MLP with residual connection
        residual = x
        x = self.post_attention_rmsnorm(x)
        mlp_output = self.mlp(x)
        
        # SwiGLU 출력을 hidden_size로 projection
        mlp_output = self.down_proj(mlp_output)
        x = residual + mlp_output
        
        return x


class SimpleLanguageModel(nn.Module):
    """
    RMSNorm + SwiGLU + Fused Linear Cross Entropy를 사용하는 간단한 언어 모델
    Input → Token Embedding → [TransformerBlock * N] → RMSNorm → LM Head → (Loss or Logits)
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 768,
        intermediate_size: int = 2048,
        num_layers: int = 2,
        use_optimized: bool = False,
        rms_norm_eps: float = 1e-6,
        rms_norm_offset: float = 0.0,  # Llama: 0.0, Gemma: 1.0
        model_type: str = "llama"  # "llama" or "gemma"
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.use_optimized = use_optimized
        self.model_type = model_type
        
        # 모델 타입에 따른 RMSNorm 설정
        if model_type.lower() == "gemma":
            rms_norm_offset = 1.0
        elif model_type.lower() == "llama":
            rms_norm_offset = 0.0
        
        # Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Transformer Layers
        self.layers = nn.ModuleList([
            SimpleTransformerBlock(
                hidden_size, 
                intermediate_size, 
                use_optimized,
                rms_norm_eps,
                rms_norm_offset
            )
            for _ in range(num_layers)
        ])
        
        # Final RMS Norm (LayerNorm 대신)
        self.final_rmsnorm = create_rms_norm(
            hidden_size, 
            eps=rms_norm_eps, 
            offset=rms_norm_offset, 
            use_optimized=use_optimized
        )
        
        # Language Model Head (Fused Linear Cross Entropy)
        self.lm_head = create_linear_cross_entropy(
            hidden_size, vocab_size, bias=False, use_optimized=use_optimized
        )
        
        # Weight initialization
        self._init_weights()
        
    def _init_weights(self):
        """가중치 초기화 (RMSNorm 고려)"""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif hasattr(module, 'weight') and module.__class__.__name__.endswith('RMSNorm'):
                # RMSNorm weight 초기화
                nn.init.ones_(module.weight)
        
    def forward(self, input_ids, targets=None):
        # Token embedding
        x = self.token_embedding(input_ids)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Final RMS norm
        x = self.final_rmsnorm(x)
        
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
                if hasattr(self.lm_head, 'bias') and self.lm_head.bias is not None:
                    logits = logits + self.lm_head.bias
            
            return logits
    
    def get_num_parameters(self):
        """모델 파라미터 수 반환"""
        return sum(p.numel() for p in self.parameters())
        
    def get_model_size_mb(self):
        """모델 크기 반환 (MB)"""
        return self.get_num_parameters() * 4 / (1024**2)  # float32 기준
    
    def get_model_info(self):
        """모델 정보 반환"""
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": len(self.layers),
            "num_parameters": self.get_num_parameters(),
            "model_size_mb": self.get_model_size_mb(),
            "model_type": self.model_type,
            "use_optimized": self.use_optimized,
            "normalization": "RMSNorm",
            "activation": "SwiGLU",
            "cross_entropy": "Fused Linear CE" if self.use_optimized else "Standard CE"
        }


def create_simple_language_model(
    vocab_size: int = 32000,
    hidden_size: int = 768,
    intermediate_size: int = 2048,
    num_layers: int = 2,
    use_optimized: bool = False,
    model_type: str = "llama",  # "llama" or "gemma"
    rms_norm_eps: float = 1e-6
):
    """간단한 언어 모델 생성 팩토리 함수 (RMSNorm 적용)"""
    return SimpleLanguageModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        use_optimized=use_optimized,
        model_type=model_type,
        rms_norm_eps=rms_norm_eps
    )


def create_llama_style_model(
    vocab_size: int = 32000,
    hidden_size: int = 768,
    intermediate_size: int = 2048,
    num_layers: int = 2,
    use_optimized: bool = False
):
    """Llama 스타일 모델 생성 (RMSNorm offset=0.0)"""
    return create_simple_language_model(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        use_optimized=use_optimized,
        model_type="llama"
    )


def create_gemma_style_model(
    vocab_size: int = 32000,
    hidden_size: int = 768,
    intermediate_size: int = 2048,
    num_layers: int = 2,
    use_optimized: bool = False
):
    """Gemma 스타일 모델 생성 (RMSNorm offset=1.0)"""
    return create_simple_language_model(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        use_optimized=use_optimized,
        model_type="gemma"
    )


# 간단한 사용 예시
def demo_model_comparison():
    """모델 비교 데모"""
    # 테스트 데이터
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    targets = torch.randint(0, 1000, (batch_size, seq_len))
    
    print("=== 모델 비교 ===")
    
    # Llama 스타일 (표준)
    llama_model = create_llama_style_model(vocab_size=1000, hidden_size=128, use_optimized=False)
    print(f"Llama (Standard): {llama_model.get_model_info()}")
    
    # Llama 스타일 (최적화)
    llama_opt_model = create_llama_style_model(vocab_size=1000, hidden_size=128, use_optimized=True)
    print(f"Llama (Optimized): {llama_opt_model.get_model_info()}")
    
    # Gemma 스타일 (최적화)
    gemma_model = create_gemma_style_model(vocab_size=1000, hidden_size=128, use_optimized=True)
    print(f"Gemma (Optimized): {gemma_model.get_model_info()}")
    
    # Forward pass 테스트
    with torch.no_grad():
        llama_loss = llama_model(input_ids, targets)
        llama_opt_loss = llama_opt_model(input_ids, targets)
        gemma_loss = gemma_model(input_ids, targets)
        
        print(f"\nLoss 비교:")
        print(f"Llama (Standard): {llama_loss.item():.4f}")
        print(f"Llama (Optimized): {llama_opt_loss.item():.4f}")
        print(f"Gemma (Optimized): {gemma_loss.item():.4f}")


__all__ = [
    'SimpleTransformerBlock',
    'SimpleLanguageModel',
    'create_simple_language_model',
    'create_llama_style_model',
    'create_gemma_style_model',
    'demo_model_comparison'
]
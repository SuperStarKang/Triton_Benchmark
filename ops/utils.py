"""
공통 유틸리티 함수들
- Triton 설정 최적화
- 메모리 연속성 보장 
- 수치 연산 함수들
"""

import torch
import triton
import triton.language as tl
from functools import wraps
import math


def calculate_settings(n_cols: int):
    """
    Triton 커널을 위한 최적 설정 계산
    
    Args:
        n_cols: 열의 개수 (처리할 요소 수)
    
    Returns:
        tuple: (BLOCK_SIZE, num_warps)
    """
    # 2의 거듭제곱으로 블록 크기 설정
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # 블록 크기에 따른 워프 수 최적화
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    elif BLOCK_SIZE >= 1024:
        num_warps = 4
    elif BLOCK_SIZE >= 512:
        num_warps = 2
    else:
        num_warps = 1
    
    return BLOCK_SIZE, num_warps


def ensure_contiguous(func):
    """
    텐서의 메모리 연속성을 보장하는 데코레이터
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 텐서 인자들을 연속 메모리로 변환
        new_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                new_args.append(arg.contiguous())
            else:
                new_args.append(arg)
        
        new_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                new_kwargs[key] = value.contiguous()
            else:
                new_kwargs[key] = value
                
        return func(*new_args, **new_kwargs)
    return wrapper


@triton.jit
def tanh(x):
    """
    고정밀도 tanh 함수 구현
    Taylor 급수와 지수 함수를 조합하여 정확도 향상
    """
    # 절댓값이 작을 때는 Taylor 급수 사용
    abs_x = tl.abs(x)
    small_mask = abs_x < 1.0
    
    # Taylor 급수: tanh(x) ≈ x - x³/3 + 2x⁵/15 - ...
    x2 = x * x
    x3 = x2 * x
    x5 = x3 * x2
    taylor_result = x - x3 / 3.0 + 2.0 * x5 / 15.0
    
    # 절댓값이 클 때는 지수 함수 사용
    # tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
    exp_2x = tl.exp(2.0 * x)
    exp_result = (exp_2x - 1.0) / (exp_2x + 1.0)
    
    # 조건에 따라 선택
    return tl.where(small_mask, taylor_result, exp_result)


# 수치 상수들
SQRT_2_OVER_PI = 0.7978845608028654  # sqrt(2/π)
GELU_TANH_COEFF = 0.044715  # GELU tanh 근사 계수


def get_cuda_capability():
    """현재 GPU의 CUDA 계산 능력 반환"""
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return major, minor
    return None, None


def optimize_memory_config(tensor_size_mb: float):
    """
    텐서 크기에 따른 메모리 최적화 설정
    
    Args:
        tensor_size_mb: 텐서 크기 (MB)
        
    Returns:
        dict: 최적화 설정
    """
    config = {
        'use_chunking': False,
        'chunk_size': None,
        'use_checkpointing': False
    }
    
    # 1GB 이상일 때 청킹 사용
    if tensor_size_mb > 1024:
        config['use_chunking'] = True
        # 청크 크기를 512MB로 제한
        config['chunk_size'] = int(512 * 1024 * 1024 / 4)  # float32 기준
        
    # 2GB 이상일 때 그래디언트 체크포인팅 권장
    if tensor_size_mb > 2048:
        config['use_checkpointing'] = True
        
    return config


def estimate_tensor_memory(shape, dtype=torch.float32):
    """
    텐서의 메모리 사용량 추정 (MB 단위)
    
    Args:
        shape: 텐서 모양
        dtype: 데이터 타입
        
    Returns:
        float: 메모리 사용량 (MB)
    """
    element_count = math.prod(shape)
    
    # 데이터 타입별 바이트 수
    dtype_bytes = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int32: 4,
        torch.int64: 8,
        torch.bool: 1
    }
    
    bytes_per_element = dtype_bytes.get(dtype, 4)
    total_bytes = element_count * bytes_per_element
    
    return total_bytes / (1024 * 1024)  # MB 단위로 변환


class TritonKernelProfiler:
    """Triton 커널 성능 프로파일링 유틸리티"""
    
    def __init__(self):
        self.profiles = {}
    
    def profile_kernel(self, kernel_name: str, kernel_func, *args, **kwargs):
        """
        커널 실행 시간 측정
        
        Args:
            kernel_name: 커널 이름
            kernel_func: 실행할 커널 함수
            *args, **kwargs: 커널 인자들
        """
        # GPU 동기화
        torch.cuda.synchronize()
        
        # 시작 이벤트
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        result = kernel_func(*args, **kwargs)
        end_event.record()
        
        torch.cuda.synchronize()
        
        # 실행 시간 계산
        elapsed_time = start_event.elapsed_time(end_event)  # ms
        
        if kernel_name not in self.profiles:
            self.profiles[kernel_name] = []
        self.profiles[kernel_name].append(elapsed_time)
        
        return result
    
    def get_average_time(self, kernel_name: str):
        """평균 실행 시간 반환 (ms)"""
        if kernel_name in self.profiles:
            return sum(self.profiles[kernel_name]) / len(self.profiles[kernel_name])
        return 0.0
    
    def reset(self):
        """프로파일 데이터 초기화"""
        self.profiles.clear()


# 전역 프로파일러 인스턴스
kernel_profiler = TritonKernelProfiler()


__all__ = [
    'calculate_settings',
    'ensure_contiguous', 
    'tanh',
    'SQRT_2_OVER_PI',
    'GELU_TANH_COEFF',
    'get_cuda_capability',
    'optimize_memory_config',
    'estimate_tensor_memory',
    'TritonKernelProfiler',
    'kernel_profiler'
]
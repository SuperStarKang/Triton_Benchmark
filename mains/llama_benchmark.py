"""
간단한 모델 메모리 벤치마킹 메인 함수 (수정 버전)
"""

import torch

# 메모리 벤치마킹 도구 import
import sys
import os

from benchmarks.memory_profiler import MemoryBenchmark
from benchmarks.model_visualizer import BenchmarkVisualizer

# 모델 import
from models.llama_gemma_model import create_llama_style_model


def create_test_data(batch_size, seq_len, vocab_size, device):
    """테스트 데이터 생성"""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return input_ids, targets

def create_model_wrapper(model, input_ids, targets):
    """
    매번 새로운 데이터로 모델을 실행하는 wrapper
    PyTorch 내부 graph reuse 최적화 방지
    """
    def model_func():
        # 매번 새로운 입력 데이터 생성 (올바른 device에 생성)
        device = input_ids.device
        vocab_size = input_ids.max().item() + 1
        
        new_input_ids = torch.randint(0, vocab_size, input_ids.shape, device=device)
        new_targets = torch.randint(0, targets.max().item() + 1, targets.shape, device=device)
        
        # 모델 파라미터 gradient 초기화
        model.zero_grad()
        
        return model(new_input_ids, new_targets)
        
    return model_func


def main():
    """메인 함수"""
    
    # 설정
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 벤치마크 파라미터
    configs = {
        "batch_size": 128,
        "seq_len": 128,
        "vocab_size": 32000,
        "hidden_size": 768,
        "intermediate_size": 768,
        "num_layers": 8
    }
    
    print(f"Llama Language Model Memory Benchmark")
    print(f"Device: {device}")
    print(f"Input shape: ({configs['batch_size']}, {configs['seq_len']})")
    print(f"Model config: hidden={configs['hidden_size']}, intermediate={configs['intermediate_size']}, layers={configs['num_layers']}")
    print("=" * 50)
    
    if device == "cpu":
        print("Warning: Running on CPU. Triton optimizations may not work.")
        return
    
    # 메모리 벤치마크 인스턴스 생성
    memory_benchmark = MemoryBenchmark(device)
    
    # 테스트 데이터 생성
    input_ids, targets = create_test_data(
        configs['batch_size'], 
        configs['seq_len'], 
        configs['vocab_size'], 
        device
    )

    # 1. Naive 모델 벤치마킹
    print("\n=== NAIVE MODEL BENCHMARKING ===")
    naive_model = create_llama_style_model(
        vocab_size=configs['vocab_size'],
        hidden_size=configs['hidden_size'],
        intermediate_size=configs['intermediate_size'],
        num_layers=configs['num_layers'],
        use_optimized=False
    ).to(device)  # 명시적으로 device로 이동
    
    print(f"Model parameters: {naive_model.get_num_parameters():,}")
    print(f"Model size: ~{naive_model.get_model_size_mb():.1f} MB")

    # 모델 wrapper 생성 (매번 새로운 데이터 사용)
    naive_model_func = create_model_wrapper(naive_model, input_ids, targets)

    # operation_memory로 직접 벤치마킹 (graph 재사용 문제 해결)
    naive_stats = memory_benchmark.profile_operation_memory(
        naive_model_func,
        warmup_runs=2,
        profile_runs=5
    )

    print(f"Naive Peak Memory: {naive_stats.get('avg_peak_memory_gb', 0):.4f} GB")
    print(f"Naive Avg Memory: {naive_stats.get('avg_avg_memory_gb', 0):.4f} GB")
    
    # 메모리 정리
    del naive_model
    torch.cuda.empty_cache()
    
    # 2. Optimized 모델 벤치마킹
    print("\n=== OPTIMIZED MODEL BENCHMARKING ===")
    optimized_model = create_llama_style_model(
        vocab_size=configs['vocab_size'],
        hidden_size=configs['hidden_size'],
        intermediate_size=configs['intermediate_size'],
        num_layers=configs['num_layers'],
        use_optimized=True
    ).to(device)  # 명시적으로 device로 이동
    
    # 모델 wrapper 생성 (매번 새로운 데이터 사용)
    optimized_model_func = create_model_wrapper(optimized_model, input_ids, targets)
    
    # operation_memory로 직접 벤치마킹
    optimized_stats = memory_benchmark.profile_operation_memory(
        optimized_model_func,
        warmup_runs=2,
        profile_runs=5
    )
    
    print(f"Optimized Peak Memory: {optimized_stats.get('avg_peak_memory_gb', 0):.4f} GB")
    print(f"Optimized Avg Memory: {optimized_stats.get('avg_avg_memory_gb', 0):.4f} GB")
    
    # 메모리 정리
    del optimized_model
    torch.cuda.empty_cache()
    
    # 3. 비교 결과 계산
    def calculate_improvement(naive_stats, optimized_stats):
        comparison = {
            "naive": naive_stats,
            "optimized": optimized_stats,
            "improvement": {}
        }
        
        # 주요 메트릭들에 대한 개선율 계산
        key_metrics = ["avg_peak_memory_gb", "avg_avg_memory_gb", "avg_memory_increase_gb"]
        for metric in key_metrics:
            if metric in naive_stats and metric in optimized_stats:
                naive_val = naive_stats[metric]
                opt_val = optimized_stats[metric]
                
                if naive_val > 0:
                    improvement_pct = ((naive_val - opt_val) / naive_val) * 100
                    comparison["improvement"][metric] = improvement_pct
        
        return comparison
    
    model_comparison = calculate_improvement(naive_stats, optimized_stats)
    
    # 4. 결과 저장
    memory_benchmark.results["LlamaLanguageModel"] = model_comparison
    
    # 결과 출력
    print("\n=== COMPARISON RESULTS ===")
    print(f"Peak Memory Improvement: {model_comparison['improvement'].get('avg_peak_memory_gb', 0):.2f}%")
    print(f"Average Memory Improvement: {model_comparison['improvement'].get('avg_avg_memory_gb', 0):.2f}%")
    
    # 5. 벤치마크 결과 시각화
    print("\n=== GENERATING VISUALIZATIONS ===")
    visualizer = BenchmarkVisualizer("./benchmark_results")

    # 메모리 세부 분석 그래프
    visualizer.plot_memory_breakdown(
        {"LlamaLanguageModel": model_comparison},
        save_name="Llama_model_memory_breakdown"
    )
    
    print("\n=== BENCHMARK COMPLETED ===")
    print("All visualizations and results have been saved to ./benchmark_results/")


if __name__ == "__main__":
    main()
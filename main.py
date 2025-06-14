"""
기존 벤치마크 클래스를 활용한 순차적 벤치마크 메인 스크립트
"""

import argparse
import sys
import os
import torch

# 프로젝트 루트 디렉토리를 Python path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ops import (
    NaiveGeGLU, OptimizedGeGLU,
    NaiveLinearCrossEntropy, OptimizedLinearCrossEntropy
)
from models import create_transformer
from benchmarks import BenchmarkVisualizer
from benchmarks.sequential_benchmark_runner import SequentialBenchmark


def setup_device():
    """GPU 설정 및 확인"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark requires GPU.")
    
    device = "cuda:0"
    torch.cuda.set_device(device)
    
    # GPU 정보 출력
    gpu_name = torch.cuda.get_device_name(device)
    gpu_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    
    print(f"Using GPU: {gpu_name}")
    print(f"GPU Memory: {gpu_memory:.1f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    
    return device, gpu_memory


def get_benchmark_config(gpu_memory: float, mode: str = "quick"):
    """GPU 메모리에 따른 벤치마크 설정"""
    if gpu_memory < 4:
        config = {
            'vocab_size': 8000,
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 2,
            'd_ff': 512,
            'batch_size': 2,
            'seq_len': 128,
            'max_linear_size': 512
        }
    elif gpu_memory < 6:
        config = {
            'vocab_size': 16000,
            'd_model': 256,
            'n_heads': 4,
            'n_layers': 3,
            'd_ff': 1024,
            'batch_size': 4,
            'seq_len': 256,
            'max_linear_size': 1024
        }
    else:
        config = {
            'vocab_size': 32000,
            'd_model': 512,
            'n_heads': 8,
            'n_layers': 4,
            'd_ff': 2048,
            'batch_size': 8,
            'seq_len': 512,
            'max_linear_size': 2048
        }
    
    # comprehensive 모드에서는 크기 증가
    if mode == "comprehensive":
        config['batch_size'] = min(config['batch_size'] * 2, 16)
        config['seq_len'] = min(config['seq_len'] * 2, 1024)
    
    return config


def run_geglu_benchmark(benchmark: SequentialBenchmark, config: dict, device: str):
    """GeGLU 순차적 벤치마크"""
    print("\n" + "="*60)
    print("🔧 BENCHMARKING GEGLU OPERATION")
    print("="*60)
    
    hidden_dim = config['d_model']
    ff_dim = config['d_ff']
    data_size = config['batch_size'] * config['seq_len']
    
    # 테스트 데이터 생성
    test_data = torch.randn(data_size, hidden_dim, device=device, requires_grad=True)
    
    def naive_creator():
        return NaiveGeGLU(hidden_dim, ff_dim)
    
    def optimized_creator():
        return OptimizedGeGLU(hidden_dim, ff_dim)
    
    benchmark.benchmark_operation_sequential(
        "GeGLU",
        naive_creator,
        optimized_creator,
        test_data,
        warmup_runs=5,
        profile_runs=20
    )


def run_linear_ce_benchmark(benchmark: SequentialBenchmark, config: dict, device: str):
    """Linear Cross Entropy 순차적 벤치마크"""
    print("\n" + "="*60)
    print("🔧 BENCHMARKING LINEAR CROSS ENTROPY")
    print("="*60)
    
    hidden_dim = config['d_model']
    vocab_size = config['vocab_size']
    data_size = min(config['max_linear_size'], config['batch_size'] * config['seq_len'])
    
    # 메모리 추정
    estimated_memory = data_size * vocab_size * 4 / (1024**3)  # GB
    print(f"Estimated LinearCE memory: {estimated_memory:.2f} GB")
    
    # 메모리 제한 확인
    gpu_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    if estimated_memory > gpu_memory * 0.6:
        print("⚠️  Skipping LinearCE due to memory constraints")
        return
    
    # 테스트 데이터 생성
    x = torch.randn(data_size, hidden_dim, device=device, requires_grad=True)
    targets = torch.randint(0, vocab_size, (data_size,), device=device)
    test_data = (x, targets)
    
    def naive_creator():
        return NaiveLinearCrossEntropy(hidden_dim, vocab_size, bias=False)
    
    def optimized_creator():
        return OptimizedLinearCrossEntropy(hidden_dim, vocab_size, bias=False)
    
    # 커스텀 실행 함수 (loss만 반환하도록)
    class LinearCEWrapper:
        def __init__(self, model):
            self.model = model
        
        def __call__(self, test_data):
            x, targets = test_data
            loss, _ = self.model(x, targets)
            return loss
    
    def naive_creator_wrapped():
        model = NaiveLinearCrossEntropy(hidden_dim, vocab_size, bias=False)
        return LinearCEWrapper(model)
    
    def optimized_creator_wrapped():
        model = OptimizedLinearCrossEntropy(hidden_dim, vocab_size, bias=False)
        return LinearCEWrapper(model)
    
    benchmark.benchmark_operation_sequential(
        "LinearCE",
        naive_creator_wrapped,
        optimized_creator_wrapped,
        test_data,
        warmup_runs=3,
        profile_runs=10
    )


def run_model_benchmark(benchmark: SequentialBenchmark, config: dict, device: str):
    """모델 전체 순차적 벤치마크"""
    print("\n" + "="*60)
    print("🏗️ BENCHMARKING TRANSFORMER MODEL")
    print("="*60)
    
    # 모델 설정
    model_config = {
        'vocab_size': config['vocab_size'],
        'd_model': config['d_model'],
        'n_heads': config['n_heads'],
        'n_layers': config['n_layers'],
        'd_ff': config['d_ff'],
        'max_seq_len': 1024,
        'dropout': 0.0
    }
    
    # 입력 데이터 (모델용으로 작게)
    batch_size = min(config['batch_size'], 4)
    seq_len = min(config['seq_len'], 256)
    
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len), device=device)
    targets = torch.randint(0, config['vocab_size'], (batch_size, seq_len), device=device)
    
    def naive_model_creator():
        return create_transformer(**model_config, use_optimized=False)
    
    def optimized_model_creator():
        return create_transformer(**model_config, use_optimized=True)
    
    benchmark.benchmark_model_sequential(
        "Transformer",
        naive_model_creator,
        optimized_model_creator,
        input_ids,
        targets,
        warmup_runs=3,
        profile_runs=10
    )


def create_visualizations(results: dict, output_dir: str):
    """결과 시각화"""
    memory_results = results.get('memory', {})
    speed_results = results.get('speed', {})
    
    if not memory_results and not speed_results:
        print("❌ No data to visualize")
        return
    
    print("\n" + "="*60)
    print("📊 CREATING VISUALIZATIONS")
    print("="*60)
    
    try:
        visualizer = BenchmarkVisualizer(output_dir)
        
        # 연산별 결과 (모델 결과 제외)
        operation_memory = {k: v for k, v in memory_results.items() if not k.startswith('model_')}
        operation_speed = {k: v for k, v in speed_results.items() if not k.startswith('model_')}
        
        if operation_memory:
            print("📊 Creating memory comparison...")
            visualizer.plot_memory_comparison(operation_memory, "sequential_memory_comparison")
        
        if operation_speed:
            print("⚡ Creating speed comparison...")
            visualizer.plot_speed_comparison(operation_speed, "sequential_speed_comparison")
        
        # CSV 저장
        print("💾 Saving results to CSV...")
        visualizer.save_results_to_csv(memory_results, speed_results, "sequential_benchmark_results.csv")
        
        print(f"✅ Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")


def print_summary(results: dict):
    """결과 요약 출력"""
    memory_results = results.get('memory', {})
    speed_results = results.get('speed', {})
    
    print("\n" + "="*60)
    print("🎉 BENCHMARK SUMMARY")
    print("="*60)
    
    memory_improvements = []
    speed_improvements = []
    
    # 메모리 개선율 수집
    for result in memory_results.values():
        if "improvement" in result:
            for imp in result["improvement"].values():
                if isinstance(imp, (int, float)):
                    memory_improvements.append(imp)
    
    # 속도 개선율 수집
    for result in speed_results.values():
        if "speedup" in result:
            for speedup_info in result["speedup"].values():
                if isinstance(speedup_info, dict) and "improvement_pct" in speedup_info:
                    speed_improvements.append(speedup_info["improvement_pct"])
    
    if memory_improvements:
        print(f"💾 Memory Improvements:")
        print(f"  Average: {sum(memory_improvements) / len(memory_improvements):.1f}%")
        print(f"  Best:    {max(memory_improvements):.1f}%")
        print(f"  Worst:   {min(memory_improvements):.1f}%")
    
    if speed_improvements:
        print(f"\n⚡ Speed Improvements:")
        print(f"  Average: {sum(speed_improvements) / len(speed_improvements):.1f}%")
        print(f"  Best:    {max(speed_improvements):.1f}%")
        print(f"  Worst:   {min(speed_improvements):.1f}%")
    
    total_benchmarks = len(memory_results) + len(speed_results)
    print(f"\n📊 Total benchmarks completed: {total_benchmarks}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Sequential Triton Optimization Benchmark")
    parser.add_argument("--mode", choices=["quick", "comprehensive"], default="quick",
                       help="Benchmark mode")
    parser.add_argument("--output-dir", default="benchmark_results",
                       help="Output directory")
    parser.add_argument("--operations", nargs="+", 
                       choices=["geglu", "linear_ce", "model"], 
                       default=["geglu", "linear_ce"],
                       help="Operations to benchmark")
    
    args = parser.parse_args()
    
    print("🚀 순차적 Triton 최적화 벤치마크")
    print("="*60)
    
    try:
        # 디바이스 설정
        device, gpu_memory = setup_device()
        
        # 설정 가져오기
        config = get_benchmark_config(gpu_memory, args.mode)
        
        print(f"📋 Mode: {args.mode}")
        print(f"🔧 GPU Memory: {gpu_memory:.1f} GB")
        print(f"📦 Batch size: {config['batch_size']}")
        print(f"📏 Sequence length: {config['seq_len']}")
        print(f"📚 Vocab size: {config['vocab_size']}")
        print(f"🧠 Model dimension: {config['d_model']}")
        print(f"⚙️  Operations: {args.operations}")
        
        # 순차적 벤치마크 실행기
        benchmark = SequentialBenchmark(device)
        
        # 선택된 연산들 벤치마크
        if "geglu" in args.operations:
            try:
                run_geglu_benchmark(benchmark, config, device)
            except Exception as e:
                print(f"❌ GeGLU benchmark failed: {e}")
        
        if "linear_ce" in args.operations:
            try:
                run_linear_ce_benchmark(benchmark, config, device)
            except Exception as e:
                print(f"❌ LinearCE benchmark failed: {e}")
        
        if "model" in args.operations:
            try:
                run_model_benchmark(benchmark, config, device)
            except Exception as e:
                print(f"❌ Model benchmark failed: {e}")
        
        # 결과 가져오기
        results = benchmark.get_results()
        
        if results['memory'] or results['speed']:
            # 시각화 생성
            create_visualizations(results, args.output_dir)
            
            # 요약 출력
            print_summary(results)
            
            print("\n🎉 Benchmark completed successfully!")
        else:
            print("❌ No successful benchmark runs")
            return 1
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
"""
간단한 GeGLU 메모리 벤치마킹 메인 함수
"""

import torch

# 메모리 벤치마킹 도구 import
from benchmarks.memory_profiler import MemoryBenchmark
from benchmarks.speed_profiler import SpeedBenchmark
from benchmarks.operation_visualizer import BenchmarkVisualizer

# GeGLU 구현들 import
from ops.geglu import create_geglu


def main():
	"""메인 함수"""
		
	# 설정
	device = "cuda:0" if torch.cuda.is_available() else "cpu"

	# 벤치마크 파라미터
	configs = {
		"batch_size": 256,
		"seq_len": 128,
		"dim_in": 768,
		"dim_out": 2048
	}
		
	print(f"GeGLU Memory Benchmark")
	print(f"Device: {device}")
	print(f"Input shape: ({configs['batch_size']}, {configs['seq_len']}, {configs['dim_in']})")
	print(f"Output shape: ({configs['batch_size']}, {configs['seq_len']}, {configs['dim_out']})")
	print("=" * 50)
		
	if device == "cpu":
		print("Warning: Running on CPU. Triton optimizations may not work.")
		return
		
	# 메모리 벤치마크 인스턴스 생성
	memory_benchmark = MemoryBenchmark(device)
		
	# 테스트 데이터 생성
	input_tensor = torch.randn(configs['batch_size'], configs['seq_len'], configs['dim_in'], device=device, requires_grad=False)

	# GeGLU 모델들 생성
	naive_model = create_geglu(configs['dim_in'], configs['dim_out'], use_optimized=False).to(device)
	optimized_model = create_geglu(configs['dim_in'], configs['dim_out'], use_optimized=True).to(device)
		
	# 1. Forward Only 메모리 벤치마킹
	input_tensor.requires_grad = False  # Forward Only 벤치마킹을 위해 requires_grad=False로 설정
	print("\n1. Forward Only Memory Benchmarking")
	print("-" * 40)
		
	forward_comparison = memory_benchmark.compare_memory_usage(
		lambda: naive_model(input_tensor),
		lambda: optimized_model(input_tensor),
		warmup_runs=3,
		profile_runs=5
	)
		
	# 결과 출력
	memory_benchmark._print_operation_results("GeGLU_Forward", forward_comparison)

	# 2. Forward + Backward 메모리 벤치마킹
	input_tensor.requires_grad = True  # Forward + Backward 벤치마킹을 위해 requires_grad=True로 설정
	print("\n2. Forward + Backward Memory Benchmarking")
	print("-" * 40)

	forward_backward_comparison = memory_benchmark.compare_memory_usage(
		lambda: naive_model(input_tensor),
		lambda: optimized_model(input_tensor),
		warmup_runs=3,
		profile_runs=5
	)
		
	# 결과 출력
	memory_benchmark._print_operation_results("GeGLU_Forward_Backward", forward_backward_comparison)
		
	# 3. 결과를 벤치마크 인스턴스에 저장
	memory_benchmark.results["GeGLU_Forward"] = forward_comparison
	memory_benchmark.results["GeGLU_Forward_Backward"] = forward_backward_comparison
		
	# 4. 결과를 json 형태로 저장
	# memory_benchmark.save_results("./benchmark_results/geglu_memory_benchmark_results.json")

	# 5. 벤치마크 결과 시각화
	visualizer = BenchmarkVisualizer("./benchmark_results")
	
	# 메모리 비교 그래프 생성
	visualizer.plot_memory_comparison(
		memory_benchmark.results,
		save_name="geglu_memory_comparison"
	)

	# 6. 실행 속도 벤치마킹
	print("=" * 50)
	print("\nSpeed Benchmarking")
	speed_benchmark = SpeedBenchmark(device)
	
	speed_comparison = speed_benchmark.compare_speed(
		lambda: naive_model(input_tensor),
		lambda: optimized_model(input_tensor),
		warmup_runs=3,
		profile_runs=5
	)
	
	# 결과 출력
	speed_benchmark._print_operation_results("GeGLU_Speed", speed_comparison)
	
	# 결과를 벤치마크 인스턴스에 저장
	speed_benchmark.results["GeGLU_Speed"] = speed_comparison
	
	# 실행 속도 결과를 json 형태로 저장
	# speed_benchmark.save_results("./benchmark_results/geglu_speed_benchmark_results.json")

	# 실행 속도 결과 시각화
	visualizer.plot_speed_comparison(
		speed_benchmark.results,
		save_name="geglu_speed_comparison"
	)

if __name__ == "__main__":
	main()
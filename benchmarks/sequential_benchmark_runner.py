"""
기존 벤치마크 클래스를 활용한 순차적 벤치마크 실행기
"""

import torch
import gc
from typing import Dict, Any, Callable

from .memory_profiler import MemoryBenchmark
from .speed_profiler import SpeedBenchmark


class SequentialBenchmark:
    """기존 벤치마크 클래스를 활용하여 순차적으로 실행"""
    
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.results = {}
    
    def _clear_memory(self):
        """메모리 완전 정리"""
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
    
    def benchmark_operation_sequential(
        self,
        name: str,
        naive_creator: Callable,
        optimized_creator: Callable,
        test_data: Any,
        warmup_runs: int = 10,
        profile_runs: int = 50
    ):
        """
        순차적으로 naive와 optimized 연산을 벤치마크
        
        Args:
            name: 벤치마크 이름
            naive_creator: naive 모델을 생성하는 함수  
            optimized_creator: optimized 모델을 생성하는 함수
            test_data: 테스트 데이터
            warmup_runs: 워밍업 실행 횟수
            profile_runs: 프로파일링 실행 횟수
        """
        print(f"\n=== Sequential Benchmarking {name} ===")
        
        # 1단계: Naive 구현 벤치마크
        print("🔍 Step 1: Benchmarking naive implementation...")
        self._clear_memory()
        
        naive_model = naive_creator().to(self.device)
        
        def run_naive():
            if isinstance(test_data, torch.Tensor):
                return naive_model(test_data)
            elif isinstance(test_data, (list, tuple)):
                return naive_model(*test_data)
            elif isinstance(test_data, dict):
                return naive_model(**test_data)
            else:
                return naive_model(test_data)
        
        # Naive 벤치마크 실행 (기존 클래스 활용)
        memory_benchmark = MemoryBenchmark(self.device)
        speed_benchmark = SpeedBenchmark(self.device)
        
        # 동일한 함수로 비교 (기존 구조 활용)
        memory_benchmark.benchmark_operation(f"{name}_naive", run_naive, run_naive)
        speed_benchmark.benchmark_operation(
            f"{name}_naive", run_naive, run_naive,
            warmup_runs=warmup_runs, profile_runs=profile_runs
        )
        
        # Naive 결과 저장
        naive_memory = memory_benchmark.results[f"{name}_naive"]
        naive_speed = speed_benchmark.results[f"{name}_naive"]
        
        # Naive 모델 삭제 및 메모리 정리
        del naive_model, memory_benchmark, speed_benchmark
        self._clear_memory()
        
        # 2단계: Optimized 구현 벤치마크
        print("🚀 Step 2: Benchmarking optimized implementation...")
        
        optimized_model = optimized_creator().to(self.device)
        
        def run_optimized():
            if isinstance(test_data, torch.Tensor):
                return optimized_model(test_data)
            elif isinstance(test_data, (list, tuple)):
                return optimized_model(*test_data)
            elif isinstance(test_data, dict):
                return optimized_model(**test_data)
            else:
                return optimized_model(test_data)
        
        # Optimized 벤치마크 실행
        memory_benchmark = MemoryBenchmark(self.device)
        speed_benchmark = SpeedBenchmark(self.device)
        
        memory_benchmark.benchmark_operation(f"{name}_optimized", run_optimized, run_optimized)
        speed_benchmark.benchmark_operation(
            f"{name}_optimized", run_optimized, run_optimized,
            warmup_runs=warmup_runs, profile_runs=profile_runs
        )
        
        # Optimized 결과 저장
        optimized_memory = memory_benchmark.results[f"{name}_optimized"]
        optimized_speed = speed_benchmark.results[f"{name}_optimized"]
        
        # Optimized 모델 삭제 및 메모리 정리
        del optimized_model, memory_benchmark, speed_benchmark
        self._clear_memory()
        
        # 3단계: 결과 통합 및 비교
        print("📊 Step 3: Comparing results...")
        
        # 결과를 기존 형식으로 재구성
        memory_comparison = self._create_comparison_result(naive_memory, optimized_memory, "memory")
        speed_comparison = self._create_comparison_result(naive_speed, optimized_speed, "speed")
        
        # 결과 저장
        self.results[name] = {
            'memory': memory_comparison,
            'speed': speed_comparison
        }
        
        # 결과 출력
        self._print_comparison_results(name, memory_comparison, speed_comparison)
    
    def benchmark_model_sequential(
        self,
        name: str,
        naive_model_creator: Callable,
        optimized_model_creator: Callable,
        input_data: torch.Tensor,
        targets: torch.Tensor = None,
        warmup_runs: int = 5,
        profile_runs: int = 20
    ):
        """모델 순차적 벤치마크"""
        print(f"\n=== Sequential Model Benchmarking {name} ===")
        
        # 1단계: Naive 모델
        print("🔍 Step 1: Benchmarking naive model...")
        self._clear_memory()
        
        naive_model = naive_model_creator().to(self.device)
        
        memory_benchmark = MemoryBenchmark(self.device)
        speed_benchmark = SpeedBenchmark(self.device)
        
        # 동일 모델로 비교 (기존 구조 활용)
        memory_benchmark.benchmark_model("naive", naive_model, naive_model, input_data, targets)
        speed_benchmark.benchmark_model(
            "naive", naive_model, naive_model, input_data, targets,
            warmup_runs=warmup_runs, profile_runs=profile_runs
        )
        
        naive_memory = memory_benchmark.results["model_naive"]
        naive_speed = speed_benchmark.results["model_naive"]
        
        del naive_model, memory_benchmark, speed_benchmark
        self._clear_memory()
        
        # 2단계: Optimized 모델
        print("🚀 Step 2: Benchmarking optimized model...")
        
        optimized_model = optimized_model_creator().to(self.device)
        
        memory_benchmark = MemoryBenchmark(self.device)
        speed_benchmark = SpeedBenchmark(self.device)
        
        memory_benchmark.benchmark_model("optimized", optimized_model, optimized_model, input_data, targets)
        speed_benchmark.benchmark_model(
            "optimized", optimized_model, optimized_model, input_data, targets,
            warmup_runs=warmup_runs, profile_runs=profile_runs
        )
        
        optimized_memory = memory_benchmark.results["model_optimized"]
        optimized_speed = speed_benchmark.results["model_optimized"]
        
        del optimized_model, memory_benchmark, speed_benchmark
        self._clear_memory()
        
        # 결과 통합
        memory_comparison = self._create_model_comparison(naive_memory, optimized_memory, "memory")
        speed_comparison = self._create_model_comparison(naive_speed, optimized_speed, "speed")
        
        self.results[f"model_{name}"] = {
            'memory': memory_comparison,
            'speed': speed_comparison
        }
        
        self._print_model_comparison_results(name, memory_comparison, speed_comparison)
    
    def _create_comparison_result(self, naive_result: Dict, optimized_result: Dict, metric_type: str) -> Dict:
        """비교 결과 생성 - 기존 구조와 호환"""
        # 기존 벤치마크 결과에서 실제 데이터 추출
        naive_data = naive_result.get("naive", {})
        optimized_data = optimized_result.get("naive", {})  # 동일 함수 비교이므로 둘 다 "naive" 키
        
        if metric_type == "memory":
            comparison = {
                "naive": naive_data,
                "optimized": optimized_data,
                "improvement": {}
            }
            
            # 메모리 개선율 계산
            key_metrics = ["avg_peak_memory_gb", "avg_avg_memory_gb", "avg_memory_increase_gb"]
            
            for metric in key_metrics:
                if metric in naive_data and metric in optimized_data:
                    naive_val = naive_data[metric]
                    opt_val = optimized_data[metric]
                    
                    if naive_val > 0:
                        improvement_pct = ((naive_val - opt_val) / naive_val) * 100
                        comparison["improvement"][metric] = improvement_pct
        
        elif metric_type == "speed":
            comparison = {
                "naive": naive_data,
                "optimized": optimized_data,
                "speedup": {}
            }
            
            # 속도 향상 계산
            metrics = ["mean_time_ms", "median_time_ms", "min_time_ms"]
            
            for metric in metrics:
                if metric in naive_data and metric in optimized_data:
                    naive_val = naive_data[metric]
                    opt_val = optimized_data[metric]
                    
                    if opt_val > 0:
                        speedup = naive_val / opt_val
                        improvement_pct = ((naive_val - opt_val) / naive_val) * 100
                        comparison["speedup"][metric] = {
                            "speedup_ratio": speedup,
                            "improvement_pct": improvement_pct
                        }
        
        return comparison
    
    def _create_model_comparison(self, naive_result: Dict, optimized_result: Dict, metric_type: str) -> Dict:
        """모델 비교 결과 생성"""
        naive_data = naive_result.get("naive", {})
        optimized_data = optimized_result.get("naive", {})
        
        if metric_type == "memory":
            comparison = {
                "naive": naive_data,
                "optimized": optimized_data,
                "improvement": {}
            }
            
            # 모델 메모리 개선율 계산
            for phase in ["forward", "backward"]:
                for metric_type in ["avg_peak_memory_gb", "avg_avg_memory_gb"]:
                    naive_key = f"{phase}_{metric_type}"
                    opt_key = f"{phase}_{metric_type}"
                    
                    if naive_key in naive_data and opt_key in optimized_data:
                        naive_val = naive_data[naive_key]
                        opt_val = optimized_data[opt_key]
                        
                        if naive_val > 0:
                            improvement_pct = ((naive_val - opt_val) / naive_val) * 100
                            comparison["improvement"][naive_key] = improvement_pct
        
        elif metric_type == "speed":
            comparison = {
                "naive": naive_data,
                "optimized": optimized_data,
                "speedup": {}
            }
            
            # 모델 속도 향상 계산
            for phase in ["forward", "backward", "total"]:
                if phase in naive_data and phase in optimized_data:
                    naive_mean = naive_data[phase]["mean_time_ms"]
                    opt_mean = optimized_data[phase]["mean_time_ms"]
                    
                    if opt_mean > 0:
                        speedup = naive_mean / opt_mean
                        improvement_pct = ((naive_mean - opt_mean) / naive_mean) * 100
                        comparison["speedup"][phase] = {
                            "speedup_ratio": speedup,
                            "improvement_pct": improvement_pct
                        }
        
        return comparison
    
    def _print_comparison_results(self, name: str, memory_result: Dict, speed_result: Dict):
        """비교 결과 출력"""
        print(f"\n📊 Sequential Results for {name}:")
        print("-" * 60)

        # 메모리 결과
        print("💾 Memory Results:")
        if "naive" in memory_result and "optimized" in memory_result:
            naive_mem = memory_result["naive"]
            opt_mem = memory_result["optimized"]
            improvement = memory_result.get("improvement", {})
            
            metrics = [
                ("Peak Memory", "avg_peak_memory_gb"),
                ("Average Memory", "avg_avg_memory_gb"),
                ("Memory Increase", "avg_memory_increase_gb")
            ]
            
            for metric_name, metric_key in metrics:
                if metric_key in naive_mem and metric_key in opt_mem:
                    naive_val = naive_mem[metric_key]
                    opt_val = opt_mem[metric_key]
                    
                    print(f"  {metric_name}:")
                    print(f"    Naive:      {naive_val:.4f} GB")
                    print(f"    Optimized:  {opt_val:.4f} GB")
                    
                    if metric_key in improvement:
                        imp_pct = improvement[metric_key]
                        print(f"    Improvement: {imp_pct:.2f}%")
        
        # 속도 결과
        print("\n⚡ Speed Results:")
        if "naive" in speed_result and "optimized" in speed_result:
            naive_speed = speed_result["naive"]
            opt_speed = speed_result["optimized"]
            speedup = speed_result.get("speedup", {})
            
            speed_metrics = [
                ("Mean Time", "mean_time_ms"),
                ("Median Time", "median_time_ms"),
                ("Min Time", "min_time_ms")
            ]
            
            for metric_name, metric_key in speed_metrics:
                if metric_key in naive_speed and metric_key in opt_speed:
                    naive_val = naive_speed[metric_key]
                    opt_val = opt_speed[metric_key]
                    
                    print(f"  {metric_name}:")
                    print(f"    Naive:      {naive_val:.4f} ms")
                    print(f"    Optimized:  {opt_val:.4f} ms")
                    
                    if metric_key in speedup:
                        speedup_info = speedup[metric_key]
                        print(f"    Speedup:    {speedup_info['speedup_ratio']:.2f}x")
                        print(f"    Improvement: {speedup_info['improvement_pct']:.2f}%")
        
        print("-" * 60)
    
    def _print_model_comparison_results(self, name: str, memory_result: Dict, speed_result: Dict):
        """모델 비교 결과 출력"""
        print(f"\n🏗️ Sequential Model Results for {name}:")
        print("-" * 60)
        
        # 메모리 결과
        if "naive" in memory_result and "optimized" in memory_result:
            naive_mem = memory_result["naive"]
            opt_mem = memory_result["optimized"]
            improvement = memory_result.get("improvement", {})
            
            for phase in ["forward", "backward"]:
                print(f"\n{phase.capitalize()} Pass Memory:")
                
                for metric_name, metric_suffix in [("Peak Memory", "peak_memory_gb"), ("Average Memory", "avg_memory_gb")]:
                    naive_key = f"{phase}_avg_{metric_suffix}"
                    opt_key = f"{phase}_avg_{metric_suffix}"
                    
                    if naive_key in naive_mem and opt_key in opt_mem:
                        naive_val = naive_mem[naive_key]
                        opt_val = opt_mem[opt_key]
                        
                        print(f"  {metric_name}:")
                        print(f"    Naive:      {naive_val:.4f} GB")
                        print(f"    Optimized:  {opt_val:.4f} GB")
                        
                        if naive_key in improvement:
                            imp_pct = improvement[naive_key]
                            print(f"    Improvement: {imp_pct:.2f}%")
        
        # 속도 결과
        if "naive" in speed_result and "optimized" in speed_result:
            naive_speed = speed_result["naive"]
            opt_speed = speed_result["optimized"]
            speedup = speed_result.get("speedup", {})
            
            phases = ["forward", "backward", "total"]
            
            for phase in phases:
                if phase in naive_speed and phase in opt_speed:
                    naive_time = naive_speed[phase]["mean_time_ms"]
                    opt_time = opt_speed[phase]["mean_time_ms"]
                    
                    print(f"\n{phase.capitalize()} Pass Speed:")
                    print(f"  Naive:      {naive_time:.4f} ms")
                    print(f"  Optimized:  {opt_time:.4f} ms")
                    
                    if phase in speedup:
                        speedup_info = speedup[phase]
                        print(f"  Speedup:    {speedup_info['speedup_ratio']:.2f}x")
                        print(f"  Improvement: {speedup_info['improvement_pct']:.2f}%")
    
    def get_results(self) -> Dict[str, Dict]:
        """벤치마크 결과 반환 - 기존 구조와 호환"""
        memory_results = {}
        speed_results = {}
        
        for name, result in self.results.items():
            if 'memory' in result:
                memory_results[name] = result['memory']
            if 'speed' in result:
                speed_results[name] = result['speed']
        
        return {
            'memory': memory_results,
            'speed': speed_results
        }


__all__ = ['SequentialBenchmark']
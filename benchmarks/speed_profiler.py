"""
실행 시간 프로파일링 도구 (리팩토링된 버전)
"""

import torch
import time
import statistics
from typing import Dict, List, Tuple, Callable, Any
from contextlib import contextmanager


class CUDATimer:
    """CUDA 이벤트를 사용한 정확한 GPU 시간 측정"""
    
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        
    @contextmanager
    def timer(self):
        """CUDA 타이머 컨텍스트 매니저"""
        torch.cuda.synchronize(self.device)
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        
        try:
            yield
        finally:
            end_event.record()
            torch.cuda.synchronize(self.device)
            
        self.elapsed_time = start_event.elapsed_time(end_event)  # milliseconds
    
    def get_time_ms(self) -> float:
        """경과 시간 반환 (밀리초)"""
        return self.elapsed_time


class SpeedBenchmark:
    """속도 벤치마크 실행기 """
    
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.results = {}
        self.timer = CUDATimer(device)

    def profile_operation_speed(
        self,
        operation_func: Callable,
        *args,
        warmup_runs: int = 10,
        profile_runs: int = 100,
        **kwargs
    ) -> Dict[str, float]:
        """
        단일 연산의 실행 시간 프로파일링
        
        Args:
            operation_func: 프로파일링할 함수
            *args: 함수 인자들
            warmup_runs: 워밍업 실행 횟수
            profile_runs: 프로파일링 실행 횟수
            **kwargs: 함수 키워드 인자들
            
        Returns:
            실행 시간 통계 (밀리초)
        """
        times = []
        
        # 워밍업 실행
        for _ in range(warmup_runs):
            try:
                _ = operation_func(*args, **kwargs)
                torch.cuda.synchronize(self.device)
            except:
                pass
        
        # 실제 프로파일링
        for _ in range(profile_runs):
            try:
                with self.timer.timer():
                    result = operation_func(*args, **kwargs)
                    
                    # gradient 계산이 필요한 경우
                    if hasattr(result, 'backward') and result.requires_grad:
                        dummy_loss = result.sum()
                        dummy_loss.backward()
                
                times.append(self.timer.get_time_ms())
                
            except Exception as e:
                print(f"Error during profiling: {e}")
                continue
        
        if not times:
            return {}
        
        # 통계 계산
        return {
            "mean_time_ms": statistics.mean(times),
            "median_time_ms": statistics.median(times),
            "min_time_ms": min(times),
            "max_time_ms": max(times),
            "std_time_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
            "total_runs": len(times)
        }

    def profile_model_speed(
        self,
        model: torch.nn.Module,
        input_data: torch.Tensor,
        targets: torch.Tensor = None,
        warmup_runs: int = 5,
        profile_runs: int = 50,
        include_backward: bool = True
    ) -> Dict[str, Any]:
        """
        모델 전체의 실행 시간 프로파일링
        
        Args:
            model: 프로파일링할 모델
            input_data: 입력 데이터
            targets: 타겟 데이터
            warmup_runs: 워밍업 실행 횟수
            profile_runs: 프로파일링 실행 횟수
            include_backward: backward pass 포함 여부
            
        Returns:
            실행 시간 통계
        """
        # 모델과 데이터를 디바이스로 이동
        model = model.to(self.device)
        input_data = input_data.to(self.device)
        if targets is not None:
            targets = targets.to(self.device)
        
        forward_times = []
        backward_times = []
        total_times = []
        
        # 워밍업
        for _ in range(warmup_runs):
            try:
                if targets is not None:
                    model.train()
                    loss = model(input_data, targets)
                    if include_backward:
                        loss.backward()
                        model.zero_grad()
                else:
                    model.eval()
                    with torch.no_grad():
                        _ = model(input_data)
            except:
                pass
        
        # 실제 프로파일링
        for _ in range(profile_runs):
            try:
                if targets is not None:
                    # 학습 모드 프로파일링
                    model.train()
                    
                    # Forward pass 시간 측정
                    with self.timer.timer():
                        loss = model(input_data, targets)
                    forward_time = self.timer.get_time_ms()
                    forward_times.append(forward_time)
                    
                    if include_backward:
                        # Backward pass 시간 측정
                        with self.timer.timer():
                            loss.backward()
                        backward_time = self.timer.get_time_ms()
                        backward_times.append(backward_time)
                        
                        total_times.append(forward_time + backward_time)
                        model.zero_grad()
                    else:
                        total_times.append(forward_time)
                else:
                    # 추론 모드 프로파일링
                    model.eval()
                    with torch.no_grad():
                        with self.timer.timer():
                            _ = model(input_data)
                        forward_time = self.timer.get_time_ms()
                        forward_times.append(forward_time)
                        total_times.append(forward_time)
            
            except Exception as e:
                print(f"Error during model profiling: {e}")
                continue
        
        # 결과 구성
        results = {}
        
        if forward_times:
            results["forward"] = {
                "mean_time_ms": statistics.mean(forward_times),
                "median_time_ms": statistics.median(forward_times),
                "min_time_ms": min(forward_times),
                "max_time_ms": max(forward_times),
                "std_time_ms": statistics.stdev(forward_times) if len(forward_times) > 1 else 0.0
            }
        
        if backward_times:
            results["backward"] = {
                "mean_time_ms": statistics.mean(backward_times),
                "median_time_ms": statistics.median(backward_times),
                "min_time_ms": min(backward_times),
                "max_time_ms": max(backward_times),
                "std_time_ms": statistics.stdev(backward_times) if len(backward_times) > 1 else 0.0
            }
        
        if total_times:
            results["total"] = {
                "mean_time_ms": statistics.mean(total_times),
                "median_time_ms": statistics.median(total_times),
                "min_time_ms": min(total_times),
                "max_time_ms": max(total_times),
                "std_time_ms": statistics.stdev(total_times) if len(total_times) > 1 else 0.0
            }
            
            # Throughput 계산 (samples/second)
            batch_size = input_data.shape[0]
            avg_time_sec = statistics.mean(total_times) / 1000.0  # ms to sec
            results["throughput"] = {
                "samples_per_sec": batch_size / avg_time_sec,
                "avg_time_per_sample_ms": statistics.mean(total_times) / batch_size
            }
        
        return results

    def compare_speed(
        self,
        naive_func: Callable,
        optimized_func: Callable,
        *args,
        warmup_runs: int = 10,
        profile_runs: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """
        두 구현의 실행 시간 비교
        
        Args:
            naive_func: 표준 구현 함수
            optimized_func: 최적화 구현 함수
            *args: 함수 인자들
            warmup_runs: 워밍업 실행 횟수
            profile_runs: 프로파일링 실행 횟수
            **kwargs: 함수 키워드 인자들
            
        Returns:
            비교 결과
        """
        print("Profiling naive implementation...")
        naive_stats = self.profile_operation_speed(
            naive_func, *args, warmup_runs=warmup_runs, profile_runs=profile_runs, **kwargs
        )
        
        print("Profiling optimized implementation...")
        optimized_stats = self.profile_operation_speed(
            optimized_func, *args, warmup_runs=warmup_runs, profile_runs=profile_runs, **kwargs
        )
        
        # 속도 향상 계산
        comparison = {
            "naive": naive_stats,
            "optimized": optimized_stats,
            "speedup": {}
        }
        
        # 주요 메트릭에 대한 속도 향상 계산
        metrics = ["mean_time_ms", "median_time_ms", "min_time_ms"]
        
        for metric in metrics:
            if metric in naive_stats and metric in optimized_stats:
                naive_val = naive_stats[metric]
                opt_val = optimized_stats[metric]
                
                if opt_val > 0:
                    speedup = naive_val / opt_val
                    improvement_pct = ((naive_val - opt_val) / naive_val) * 100
                    comparison["speedup"][metric] = {
                        "speedup_ratio": speedup,
                        "improvement_pct": improvement_pct
                    }
        
        return comparison

    def compare_model_speed(
        self,
        naive_model: torch.nn.Module,
        optimized_model: torch.nn.Module,
        input_data: torch.Tensor,
        targets: torch.Tensor = None,
        warmup_runs: int = 5,
        profile_runs: int = 50,
        include_backward: bool = True
    ) -> Dict[str, Any]:
        """
        두 모델의 실행 시간 비교
        
        Args:
            naive_model: 표준 구현 모델
            optimized_model: 최적화 구현 모델
            input_data: 입력 데이터
            targets: 타겟 데이터
            warmup_runs: 워밍업 실행 횟수
            profile_runs: 프로파일링 실행 횟수
            include_backward: backward pass 포함 여부
            
        Returns:
            비교 결과
        """
        print("Profiling naive model...")
        naive_stats = self.profile_model_speed(
            naive_model, input_data, targets, warmup_runs, profile_runs, include_backward
        )
        
        print("Profiling optimized model...")
        optimized_stats = self.profile_model_speed(
            optimized_model, input_data, targets, warmup_runs, profile_runs, include_backward
        )
        
        # 비교 결과 계산
        comparison = {
            "naive": naive_stats,
            "optimized": optimized_stats,
            "speedup": {}
        }
        
        # 각 페이즈별 속도 향상 계산
        phases = ["forward", "backward", "total"] if include_backward else ["forward", "total"]
        for phase in phases:
            if phase in naive_stats and phase in optimized_stats:
                naive_mean = naive_stats[phase]["mean_time_ms"]
                opt_mean = optimized_stats[phase]["mean_time_ms"]
                
                if opt_mean > 0:
                    speedup = naive_mean / opt_mean
                    improvement_pct = ((naive_mean - opt_mean) / naive_mean) * 100
                    comparison["speedup"][phase] = {
                        "speedup_ratio": speedup,
                        "improvement_pct": improvement_pct
                    }
        
        # Throughput 비교
        if "throughput" in naive_stats and "throughput" in optimized_stats:
            naive_throughput = naive_stats["throughput"]["samples_per_sec"]
            opt_throughput = optimized_stats["throughput"]["samples_per_sec"]
            
            throughput_improvement = ((opt_throughput - naive_throughput) / naive_throughput) * 100
            comparison["speedup"]["throughput"] = {
                "throughput_improvement_pct": throughput_improvement,
                "throughput_ratio": opt_throughput / naive_throughput
            }
        
        return comparison

    def _print_operation_results(self, name: str, comparison: Dict):
        """연산 결과 출력"""
        naive = comparison["naive"]
        optimized = comparison["optimized"]
        speedup = comparison["speedup"]
        
        print(f"\nSpeed Results for {name}:")
        print("-" * 50)
        
        metrics = [
            ("Mean Time", "mean_time_ms"),
            ("Median Time", "median_time_ms"),
            ("Min Time", "min_time_ms")
        ]
        
        for metric_name, metric_key in metrics:
            if metric_key in naive and metric_key in optimized:
                naive_val = naive[metric_key]
                opt_val = optimized[metric_key]
                
                print(f"{metric_name}:")
                print(f"  Naive:      {naive_val:.4f} ms")
                print(f"  Optimized:  {opt_val:.4f} ms")
                
                if metric_key in speedup:
                    speedup_info = speedup[metric_key]
                    print(f"  Speedup:    {speedup_info['speedup_ratio']:.2f}x")
                    print(f"  Improvement: {speedup_info['improvement_pct']:.2f}%")
                print()

    def _print_model_results(self, name: str, comparison: Dict):
        """모델 결과 출력"""
        naive = comparison["naive"]
        optimized = comparison["optimized"]
        speedup = comparison["speedup"]
        
        print(f"\nModel Speed Results for {name}:")
        print("-" * 50)
        
        phases = ["forward", "backward", "total"]
        
        for phase in phases:
            if phase in naive and phase in optimized:
                naive_time = naive[phase]["mean_time_ms"]
                opt_time = optimized[phase]["mean_time_ms"]
                
                print(f"\n{phase.capitalize()} Pass:")
                print(f"  Naive:      {naive_time:.4f} ms")
                print(f"  Optimized:  {opt_time:.4f} ms")
                
                if phase in speedup:
                    speedup_info = speedup[phase]
                    print(f"  Speedup:    {speedup_info['speedup_ratio']:.2f}x")
                    print(f"  Improvement: {speedup_info['improvement_pct']:.2f}%")
        
        # Throughput 결과
        if "throughput" in speedup:
            throughput_info = speedup["throughput"]
            print(f"\nThroughput:")
            if "throughput" in naive and "throughput" in optimized:
                naive_throughput = naive["throughput"]["samples_per_sec"]
                opt_throughput = optimized["throughput"]["samples_per_sec"]
                print(f"  Naive:      {naive_throughput:.2f} samples/sec")
                print(f"  Optimized:  {opt_throughput:.2f} samples/sec")
            print(f"  Improvement: {throughput_info['throughput_improvement_pct']:.2f}%")

    def get_summary(self) -> Dict[str, Any]:
        """벤치마크 결과 요약 반환"""
        summary = {
            "operations": {},
            "models": {},
            "overall_speedup": {}
        }
        
        # 연산별 요약
        for name, result in self.results.items():
            if name.startswith("model_"):
                model_name = name[6:]  # "model_" 제거
                summary["models"][model_name] = result["speedup"]
            else:
                summary["operations"][name] = result["speedup"]
        
        # 전체 속도 향상 계산
        all_speedups = []
        for result in self.results.values():
            for speedup_info in result["speedup"].values():
                if isinstance(speedup_info, dict) and "speedup_ratio" in speedup_info:
                    all_speedups.append(speedup_info["speedup_ratio"])
        
        if all_speedups:
            summary["overall_speedup"] = {
                "average": sum(all_speedups) / len(all_speedups),
                "max": max(all_speedups),
                "min": min(all_speedups)
            }
        
        return summary

    def clear_results(self):
        """결과 초기화"""
        self.results.clear()

    def save_results(self, filepath: str):
        """결과를 JSON 파일로 저장"""
        import json
        
        def make_json_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)
        
        serializable_results = make_json_serializable(self.results)
        
        with open(filepath, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {filepath}")

__all__ = [
    'CUDATimer',
    'SpeedBenchmark',
]
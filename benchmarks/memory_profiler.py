"""
메모리 사용량 프로파일링 도구 (리팩토링된 버전)
"""

import torch
import gc
import time
from typing import Dict, List, Tuple, Callable, Any
from contextlib import contextmanager
import threading
from collections import defaultdict


class GPUMemoryMonitor:
    """GPU 메모리 사용량 실시간 모니터링"""
    
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.monitoring = False
        self.memory_history = []
        self.peak_memory = 0
        self.monitor_thread = None
        
    def start_monitoring(self, interval: float = 0.01):
        """메모리 모니터링 시작"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.memory_history.clear()
        self.peak_memory = 0
        
        def monitor():
            while self.monitoring:
                try:
                    current_memory = torch.cuda.memory_allocated(self.device) / (1024 ** 3)  # GB
                    self.memory_history.append(current_memory)
                    self.peak_memory = max(self.peak_memory, current_memory)
                    time.sleep(interval)
                except:
                    break
        
        self.monitor_thread = threading.Thread(target=monitor)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """메모리 모니터링 중지"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def get_stats(self) -> Dict[str, float]:
        """메모리 사용량 통계 반환"""
        if not self.memory_history:
            return {"peak_memory_gb": 0.0, "avg_memory_gb": 0.0, "min_memory_gb": 0.0}
        
        return {
            "peak_memory_gb": self.peak_memory,
            "avg_memory_gb": sum(self.memory_history) / len(self.memory_history),
            "min_memory_gb": min(self.memory_history)
        }


@contextmanager
def memory_profiler(device: str = "cuda:0"):
    """메모리 프로파일링 컨텍스트 매니저"""
    # 메모리 정리
    torch.cuda.empty_cache()
    gc.collect()
    
    # 시작 메모리
    start_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)
    
    # 모니터 시작
    monitor = GPUMemoryMonitor(device)
    monitor.start_monitoring()
    
    try:
        yield monitor
    finally:
        # 모니터 중지
        monitor.stop_monitoring()
        
        # 종료 메모리
        end_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)
        
        # 최종 통계에 시작/종료 메모리 추가
        stats = monitor.get_stats()
        stats.update({
            "start_memory_gb": start_memory,
            "end_memory_gb": end_memory,
            "memory_increase_gb": end_memory - start_memory
        })
        monitor.final_stats = stats


class MemoryBenchmark:
    """메모리 벤치마크 실행기 (리팩토링된 버전)"""
    
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.results = {}
    
    def profile_operation_memory(
        self,
        operation_func: Callable,
        *args,
        warmup_runs: int = 3,
        profile_runs: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        단일 연산의 메모리 사용량 프로파일링
        
        Args:
            operation_func: 프로파일링할 함수
            *args: 함수 인자들
            warmup_runs: 워밍업 실행 횟수
            profile_runs: 프로파일링 실행 횟수
            **kwargs: 함수 키워드 인자들
            
        Returns:
            메모리 사용량 통계
        """
        results = []
        
        # 워밍업
        for _ in range(warmup_runs):
            with torch.no_grad():
                try:
                    _ = operation_func(*args, **kwargs)
                except:
                    pass
            torch.cuda.empty_cache()
            gc.collect()
        
        # 실제 프로파일링
        for run_idx in range(profile_runs):
            with memory_profiler(self.device) as monitor:
                try:
                    result = operation_func(*args, **kwargs)
                    # gradient 계산이 필요한 경우
                    if hasattr(result, 'backward') and result.requires_grad:
                        dummy_loss = result.sum()
                        dummy_loss.backward()
                except Exception as e:
                    print(f"Error in run {run_idx}: {e}")
                    continue
            
            if hasattr(monitor, 'final_stats'):
                results.append(monitor.final_stats)
            
            # 메모리 정리
            torch.cuda.empty_cache()
            gc.collect()
        
        if not results:
            return {}
        
        # 평균 통계 계산
        avg_stats = {}
        for key in results[0].keys():
            values = [r[key] for r in results if key in r]
            if values:
                avg_stats[f"avg_{key}"] = sum(values) / len(values)
                avg_stats[f"max_{key}"] = max(values)
                avg_stats[f"min_{key}"] = min(values)
        
        return avg_stats

    def profile_model_memory(
        self,
        model: torch.nn.Module,
        input_data: torch.Tensor,
        targets: torch.Tensor = None,
        include_backward: bool = True,
        warmup_runs: int = 2,
        profile_runs: int = 3
    ) -> Dict[str, float]:
        """
        모델 전체의 메모리 사용량 프로파일링
        
        Args:
            model: 프로파일링할 모델
            input_data: 입력 데이터
            targets: 타겟 데이터 (loss 계산용)
            include_backward: backward pass 포함 여부
            warmup_runs: 워밍업 실행 횟수
            profile_runs: 프로파일링 실행 횟수
            
        Returns:
            메모리 사용량 통계
        """
        # 모델과 데이터를 디바이스로 이동
        model = model.to(self.device)
        input_data = input_data.to(self.device)
        if targets is not None:
            targets = targets.to(self.device)
        
        model.eval()
        
        def model_forward():
            if targets is not None:
                # 학습 모드로 설정
                model.train()
                output = model(input_data, targets)
                return output
            else:
                # 추론 모드
                with torch.no_grad():
                    output = model(input_data)
                return output
        
        def model_forward_backward():
            model.train()
            if targets is not None:
                loss = model(input_data, targets)
                loss.backward()
                return loss
            else:
                output = model(input_data)
                # 더미 손실로 backward 수행
                dummy_loss = output.sum()
                dummy_loss.backward()
                return dummy_loss
        
        # Forward pass 메모리 프로파일링
        forward_stats = self.profile_operation_memory(
            model_forward,
            warmup_runs=warmup_runs,
            profile_runs=profile_runs
        )
        
        # Forward + Backward pass 메모리 프로파일링 (옵션)
        if include_backward:
            backward_stats = self.profile_operation_memory(
                model_forward_backward,
                warmup_runs=warmup_runs,
                profile_runs=profile_runs
            )
            
            # 결과 통합
            combined_stats = {}
            for key, value in forward_stats.items():
                combined_stats[f"forward_{key}"] = value
            for key, value in backward_stats.items():
                combined_stats[f"backward_{key}"] = value
            
            return combined_stats
        
        return forward_stats

    def compare_memory_usage(
        self,
        naive_func: Callable,
        optimized_func: Callable,
        *args,
        warmup_runs: int = 3,
        profile_runs: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        두 구현의 메모리 사용량 비교
        
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
        naive_stats = self.profile_operation_memory(
            naive_func, *args, warmup_runs=warmup_runs, profile_runs=profile_runs, **kwargs
        )
        
        print("Profiling optimized implementation...")
        optimized_stats = self.profile_operation_memory(
            optimized_func, *args, warmup_runs=warmup_runs, profile_runs=profile_runs, **kwargs
        )
        
        # 개선율 계산
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

    def compare_model_memory(
        self,
        naive_model: torch.nn.Module,
        optimized_model: torch.nn.Module,
        input_data: torch.Tensor,
        targets: torch.Tensor = None,
        include_backward: bool = True,
        warmup_runs: int = 2,
        profile_runs: int = 3
    ) -> Dict[str, Any]:
        """
        두 모델의 메모리 사용량 비교
        
        Args:
            naive_model: 표준 구현 모델
            optimized_model: 최적화 구현 모델
            input_data: 입력 데이터
            targets: 타겟 데이터
            include_backward: backward pass 포함 여부
            warmup_runs: 워밍업 실행 횟수
            profile_runs: 프로파일링 실행 횟수
            
        Returns:
            비교 결과
        """
        print("Profiling naive model...")
        naive_stats = self.profile_model_memory(
            naive_model, input_data, targets, include_backward, warmup_runs, profile_runs
        )
        
        print("Profiling optimized model...")
        optimized_stats = self.profile_model_memory(
            optimized_model, input_data, targets, include_backward, warmup_runs, profile_runs
        )
        
        # 비교 결과 저장
        comparison = {
            "naive": naive_stats,
            "optimized": optimized_stats,
            "improvement": {}
        }
        
        # 개선율 계산
        phases = ["forward", "backward"] if include_backward else ["forward"]
        for phase in phases:
            for metric_type in ["avg_peak_memory_gb", "avg_avg_memory_gb"]:
                naive_key = f"{phase}_{metric_type}"
                opt_key = f"{phase}_{metric_type}"
                
                if naive_key in naive_stats and opt_key in optimized_stats:
                    naive_val = naive_stats[naive_key]
                    opt_val = optimized_stats[opt_key]
                    
                    if naive_val > 0:
                        improvement_pct = ((naive_val - opt_val) / naive_val) * 100
                        comparison["improvement"][naive_key] = improvement_pct
        
        return comparison

    def _print_operation_results(self, name: str, comparison: Dict):
        """연산 결과 출력"""
        naive = comparison["naive"]
        optimized = comparison["optimized"]
        improvement = comparison["improvement"]
        
        print(f"\nResults for {name}:")
        print("-" * 50)
        
        metrics = [
            ("Peak Memory", "avg_peak_memory_gb"),
            ("Average Memory", "avg_avg_memory_gb"),
            ("Memory Increase", "avg_memory_increase_gb")
        ]
        
        for metric_name, metric_key in metrics:
            if metric_key in naive and metric_key in optimized:
                naive_val = naive[metric_key]
                opt_val = optimized[metric_key]
                
                print(f"{metric_name}:")
                print(f"  Naive:      {naive_val:.4f} GB")
                print(f"  Optimized:  {opt_val:.4f} GB")
                
                if metric_key in improvement:
                    imp_pct = improvement[metric_key]
                    print(f"  Improvement: {imp_pct:.2f}%")
                print()

    def _print_model_results(self, name: str, comparison: Dict):
        """모델 결과 출력"""
        naive = comparison["naive"]
        optimized = comparison["optimized"]
        improvement = comparison["improvement"]
        
        print(f"\nModel Results for {name}:")
        print("-" * 50)
        
        phases = []
        if any(key.startswith("forward_") for key in naive.keys()):
            phases.append("forward")
        if any(key.startswith("backward_") for key in naive.keys()):
            phases.append("backward")
        
        for phase in phases:
            print(f"\n{phase.capitalize()} Pass:")
            
            for metric_name, metric_suffix in [("Peak Memory", "peak_memory_gb"), ("Average Memory", "avg_memory_gb")]:
                naive_key = f"{phase}_avg_{metric_suffix}"
                opt_key = f"{phase}_avg_{metric_suffix}"
                
                if naive_key in naive and opt_key in optimized:
                    naive_val = naive[naive_key]
                    opt_val = optimized[opt_key]
                    
                    print(f"  {metric_name}:")
                    print(f"    Naive:      {naive_val:.4f} GB")
                    print(f"    Optimized:  {opt_val:.4f} GB")
                    
                    if naive_key in improvement:
                        imp_pct = improvement[naive_key]
                        print(f"    Improvement: {imp_pct:.2f}%")

    def get_summary(self) -> Dict[str, Any]:
        """벤치마크 결과 요약 반환"""
        summary = {
            "operations": {},
            "models": {},
            "overall_improvement": {}
        }
        
        # 연산별 요약
        for name, result in self.results.items():
            if name.startswith("model_"):
                model_name = name[6:]  # "model_" 제거
                summary["models"][model_name] = result["improvement"]
            else:
                summary["operations"][name] = result["improvement"]
        
        # 전체 개선율 계산
        all_improvements = []
        for result in self.results.values():
            for imp_val in result["improvement"].values():
                if isinstance(imp_val, (int, float)):
                    all_improvements.append(imp_val)
        
        if all_improvements:
            summary["overall_improvement"] = {
                "average": sum(all_improvements) / len(all_improvements),
                "max": max(all_improvements),
                "min": min(all_improvements)
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
    'GPUMemoryMonitor',
    'memory_profiler', 
    'MemoryBenchmark',
]
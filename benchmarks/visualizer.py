"""
벤치마크 결과 시각화 도구
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import os


# 시각화 스타일 설정
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class BenchmarkVisualizer:
    """벤치마크 결과 시각화 클래스"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_memory_comparison(
        self,
        memory_results: Dict[str, Any],
        save_name: str = "memory_comparison"
    ):
        """메모리 사용량 비교 그래프"""
        # 데이터 준비
        operations = []
        naive_peak = []
        optimized_peak = []
        naive_avg = []
        optimized_avg = []
        
        for op_name, result in memory_results.items():
            if "naive" in result and "optimized" in result:
                operations.append(op_name)
                
                # Peak memory
                naive_peak.append(result["naive"].get("avg_peak_memory_gb", 0))
                optimized_peak.append(result["optimized"].get("avg_peak_memory_gb", 0))
                
                # Average memory
                naive_avg.append(result["naive"].get("avg_avg_memory_gb", 0))
                optimized_avg.append(result["optimized"].get("avg_avg_memory_gb", 0))
        
        if not operations:
            print("No memory data to plot")
            return
        
        # 서브플롯 생성
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        x = np.arange(len(operations))
        width = 0.35
        
        # Peak Memory 그래프
        bars1 = ax1.bar(x - width/2, naive_peak, width, label='Naive', alpha=0.8)
        bars2 = ax1.bar(x + width/2, optimized_peak, width, label='Optimized', alpha=0.8)
        
        ax1.set_xlabel('Operations')
        ax1.set_ylabel('Peak Memory (GB)')
        ax1.set_title('Peak Memory Usage Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(operations, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 개선율 표시
        for i, (naive, opt) in enumerate(zip(naive_peak, optimized_peak)):
            if naive > 0:
                improvement = ((naive - opt) / naive) * 100
                ax1.text(i, max(naive, opt) + 0.01, f'{improvement:.1f}%', 
                        ha='center', va='bottom', fontweight='bold', color='green')
        
        # Average Memory 그래프
        bars3 = ax2.bar(x - width/2, naive_avg, width, label='Naive', alpha=0.8)
        bars4 = ax2.bar(x + width/2, optimized_avg, width, label='Optimized', alpha=0.8)
        
        ax2.set_xlabel('Operations')
        ax2.set_ylabel('Average Memory (GB)')
        ax2.set_title('Average Memory Usage Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(operations, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 개선율 표시
        for i, (naive, opt) in enumerate(zip(naive_avg, optimized_avg)):
            if naive > 0:
                improvement = ((naive - opt) / naive) * 100
                ax2.text(i, max(naive, opt) + 0.01, f'{improvement:.1f}%', 
                        ha='center', va='bottom', fontweight='bold', color='green')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{save_name}.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_speed_comparison(
        self,
        speed_results: Dict[str, Any],
        save_name: str = "speed_comparison"
    ):
        """실행 시간 비교 그래프"""
        operations = []
        naive_times = []
        optimized_times = []
        speedup_ratios = []
        
        for op_name, result in speed_results.items():
            if "naive" in result and "optimized" in result:
                operations.append(op_name)
                
                naive_time = result["naive"].get("mean_time_ms", 0)
                opt_time = result["optimized"].get("mean_time_ms", 0)
                
                naive_times.append(naive_time)
                optimized_times.append(opt_time)
                
                # 속도 향상 비율 계산
                if opt_time > 0:
                    speedup_ratios.append(naive_time / opt_time)
                else:
                    speedup_ratios.append(1.0)
        
        if not operations:
            print("No speed data to plot")
            return
        
        # 서브플롯 생성
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        x = np.arange(len(operations))
        width = 0.35
        
        # 실행 시간 비교
        bars1 = ax1.bar(x - width/2, naive_times, width, label='Naive', alpha=0.8)
        bars2 = ax1.bar(x + width/2, optimized_times, width, label='Optimized', alpha=0.8)
        
        ax1.set_xlabel('Operations')
        ax1.set_ylabel('Execution Time (ms)')
        ax1.set_title('Execution Time Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(operations, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')  # 로그 스케일로 차이를 명확히 표시
        
        # 속도 향상 비율 그래프
        colors = ['green' if ratio > 1 else 'red' for ratio in speedup_ratios]
        bars3 = ax2.bar(x, speedup_ratios, color=colors, alpha=0.8)
        
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Operations')
        ax2.set_ylabel('Speedup Ratio (x)')
        ax2.set_title('Speedup Ratio (Higher is Better)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(operations, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # 속도 향상 비율 값 표시
        for i, ratio in enumerate(speedup_ratios):
            ax2.text(i, ratio + 0.05, f'{ratio:.2f}x', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{save_name}.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(
        self,
        model_memory_results: Dict[str, Any],
        model_speed_results: Dict[str, Any],
        save_name: str = "model_comparison"
    ):
        """모델 전체 비교 그래프"""
        # 데이터 준비
        models = []
        memory_improvements = []
        speed_improvements = []
        
        for model_name in model_memory_results.keys():
            if model_name in model_speed_results:
                models.append(model_name)
                
                # 메모리 개선율 (forward pass peak memory 기준)
                mem_result = model_memory_results[model_name]
                if "improvement" in mem_result:
                    forward_peak_key = "forward_avg_peak_memory_gb"
                    mem_improvement = mem_result["improvement"].get(forward_peak_key, 0)
                    memory_improvements.append(mem_improvement)
                else:
                    memory_improvements.append(0)
                
                # 속도 개선율 (total time 기준)
                speed_result = model_speed_results[model_name]
                if "speedup" in speed_result and "total" in speed_result["speedup"]:
                    speed_improvement = speed_result["speedup"]["total"].get("improvement_pct", 0)
                    speed_improvements.append(speed_improvement)
                else:
                    speed_improvements.append(0)
        
        if not models:
            print("No model comparison data to plot")
            return
        
        # 그래프 생성
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        x = np.arange(len(models))
        
        # 메모리 개선율 그래프
        colors1 = ['green' if imp > 0 else 'red' for imp in memory_improvements]
        bars1 = ax1.bar(x, memory_improvements, color=colors1, alpha=0.8)
        
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Memory Improvement (%)')
        ax1.set_title('Memory Usage Improvement')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # 값 표시
        for i, imp in enumerate(memory_improvements):
            ax1.text(i, imp + (1 if imp >= 0 else -1), f'{imp:.1f}%', 
                    ha='center', va='bottom' if imp >= 0 else 'top', fontweight='bold')
        
        # 속도 개선율 그래프
        colors2 = ['green' if imp > 0 else 'red' for imp in speed_improvements]
        bars2 = ax2.bar(x, speed_improvements, color=colors2, alpha=0.8)
        
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Speed Improvement (%)')
        ax2.set_title('Execution Speed Improvement')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # 값 표시
        for i, imp in enumerate(speed_improvements):
            ax2.text(i, imp + (1 if imp >= 0 else -1), f'{imp:.1f}%', 
                    ha='center', va='bottom' if imp >= 0 else 'top', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{save_name}.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_detailed_breakdown(
        self,
        model_speed_results: Dict[str, Any],
        save_name: str = "detailed_breakdown"
    ):
        """모델별 세부 분석 그래프"""
        if not model_speed_results:
            print("No detailed data to plot")
            return
        
        # 데이터 준비
        models = list(model_speed_results.keys())
        phases = ["forward", "backward", "total"]
        
        fig, axes = plt.subplots(len(models), 1, figsize=(12, 4 * len(models)))
        if len(models) == 1:
            axes = [axes]
        
        for idx, (model_name, result) in enumerate(model_speed_results.items()):
            ax = axes[idx]
            
            naive_times = []
            opt_times = []
            phase_labels = []
            
            for phase in phases:
                if phase in result.get("naive", {}) and phase in result.get("optimized", {}):
                    naive_time = result["naive"][phase].get("mean_time_ms", 0)
                    opt_time = result["optimized"][phase].get("mean_time_ms", 0)
                    
                    naive_times.append(naive_time)
                    opt_times.append(opt_time)
                    phase_labels.append(phase.capitalize())
            
            x = np.arange(len(phase_labels))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, naive_times, width, label='Naive', alpha=0.8)
            bars2 = ax.bar(x + width/2, opt_times, width, label='Optimized', alpha=0.8)
            
            ax.set_xlabel('Phase')
            ax.set_ylabel('Time (ms)')
            ax.set_title(f'{model_name} - Detailed Breakdown')
            ax.set_xticks(x)
            ax.set_xticklabels(phase_labels)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 개선율 표시
            for i, (naive, opt) in enumerate(zip(naive_times, opt_times)):
                if naive > 0:
                    improvement = ((naive - opt) / naive) * 100
                    ax.text(i, max(naive, opt) + max(naive_times) * 0.05, f'{improvement:.1f}%', 
                           ha='center', va='bottom', fontweight='bold', color='green')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{save_name}.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_summary_report(
        self,
        memory_summary: Dict[str, Any],
        speed_summary: Dict[str, Any],
        save_name: str = "summary_report"
    ):
        """종합 요약 리포트 생성"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 전체 메모리 개선율 히스토그램
        if "operations" in memory_summary:
            all_mem_improvements = []
            for op_improvements in memory_summary["operations"].values():
                for imp in op_improvements.values():
                    if isinstance(imp, (int, float)):
                        all_mem_improvements.append(imp)
            
            if all_mem_improvements:
                ax1.hist(all_mem_improvements, bins=20, alpha=0.7, color='blue', edgecolor='black')
                ax1.axvline(np.mean(all_mem_improvements), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(all_mem_improvements):.1f}%')
                ax1.set_xlabel('Memory Improvement (%)')
                ax1.set_ylabel('Frequency')
                ax1.set_title('Distribution of Memory Improvements')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
        
        # 2. 전체 속도 개선율 히스토그램
        if "operations" in speed_summary:
            all_speed_improvements = []
            for op_speedups in speed_summary["operations"].values():
                for speedup_info in op_speedups.values():
                    if isinstance(speedup_info, dict) and "improvement_pct" in speedup_info:
                        all_speed_improvements.append(speedup_info["improvement_pct"])
            
            if all_speed_improvements:
                ax2.hist(all_speed_improvements, bins=20, alpha=0.7, color='green', edgecolor='black')
                ax2.axvline(np.mean(all_speed_improvements), color='red', linestyle='--',
                           label=f'Mean: {np.mean(all_speed_improvements):.1f}%')
                ax2.set_xlabel('Speed Improvement (%)')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Distribution of Speed Improvements')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        # 3. 메모리 vs 속도 개선율 산점도
        mem_improvements = []
        speed_improvements = []
        labels = []
        
        # 연산별 데이터 수집
        if "operations" in memory_summary and "operations" in speed_summary:
            for op_name in memory_summary["operations"]:
                if op_name in speed_summary["operations"]:
                    # 메모리 개선율 평균
                    mem_vals = list(memory_summary["operations"][op_name].values())
                    mem_avg = np.mean([v for v in mem_vals if isinstance(v, (int, float))])
                    
                    # 속도 개선율 평균
                    speed_vals = []
                    for speedup_info in speed_summary["operations"][op_name].values():
                        if isinstance(speedup_info, dict) and "improvement_pct" in speedup_info:
                            speed_vals.append(speedup_info["improvement_pct"])
                    speed_avg = np.mean(speed_vals) if speed_vals else 0
                    
                    mem_improvements.append(mem_avg)
                    speed_improvements.append(speed_avg)
                    labels.append(op_name)
        
        if mem_improvements and speed_improvements:
            scatter = ax3.scatter(mem_improvements, speed_improvements, alpha=0.7, s=100)
            
            # 레이블 추가
            for i, label in enumerate(labels):
                ax3.annotate(label, (mem_improvements[i], speed_improvements[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax3.set_xlabel('Memory Improvement (%)')
            ax3.set_ylabel('Speed Improvement (%)')
            ax3.set_title('Memory vs Speed Improvement')
            ax3.grid(True, alpha=0.3)
            
            # 참조선 추가
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # 4. 종합 요약 테이블
        ax4.axis('tight')
        ax4.axis('off')
        
        # 요약 통계 테이블 생성
        summary_data = []
        
        if "overall_improvement" in memory_summary:
            mem_overall = memory_summary["overall_improvement"]
            summary_data.append([
                "Memory Improvement",
                f"{mem_overall.get('average', 0):.1f}%",
                f"{mem_overall.get('max', 0):.1f}%",
                f"{mem_overall.get('min', 0):.1f}%"
            ])
        
        if "overall_speedup" in speed_summary:
            speed_overall = speed_summary["overall_speedup"]
            summary_data.append([
                "Speed Improvement",
                f"{(speed_overall.get('average', 1) - 1) * 100:.1f}%",
                f"{(speed_overall.get('max', 1) - 1) * 100:.1f}%", 
                f"{(speed_overall.get('min', 1) - 1) * 100:.1f}%"
            ])
        
        if summary_data:
            table = ax4.table(cellText=summary_data,
                            colLabels=["Metric", "Average", "Maximum", "Minimum"],
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 2)
            ax4.set_title('Overall Performance Summary', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{save_name}.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results_to_csv(
        self,
        memory_results: Dict[str, Any],
        speed_results: Dict[str, Any],
        filename: str = "benchmark_results.csv"
    ):
        """결과를 CSV 파일로 저장"""
        data = []
        
        # 모든 연산/모델에 대해 데이터 수집
        all_names = set(memory_results.keys()) | set(speed_results.keys())
        
        for name in all_names:
            row = {"name": name}
            
            # 메모리 데이터
            if name in memory_results:
                mem_result = memory_results[name]
                if "naive" in mem_result:
                    row["naive_peak_memory_gb"] = mem_result["naive"].get("avg_peak_memory_gb", 0)
                    row["naive_avg_memory_gb"] = mem_result["naive"].get("avg_avg_memory_gb", 0)
                if "optimized" in mem_result:
                    row["optimized_peak_memory_gb"] = mem_result["optimized"].get("avg_peak_memory_gb", 0)
                    row["optimized_avg_memory_gb"] = mem_result["optimized"].get("avg_avg_memory_gb", 0)
            
            # 속도 데이터
            if name in speed_results:
                speed_result = speed_results[name]
                if "naive" in speed_result:
                    row["naive_time_ms"] = speed_result["naive"].get("mean_time_ms", 0)
                if "optimized" in speed_result:
                    row["optimized_time_ms"] = speed_result["optimized"].get("mean_time_ms", 0)
                
                # 속도 향상 비율
                if "speedup" in speed_result:
                    speedup_info = speed_result["speedup"]
                    if "mean_time_ms" in speedup_info:
                        row["speedup_ratio"] = speedup_info["mean_time_ms"].get("speedup_ratio", 1.0)
            
            data.append(row)
        
        # DataFrame 생성 및 저장
        df = pd.DataFrame(data)
        csv_path = os.path.join(self.output_dir, filename)
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")


__all__ = [
    'BenchmarkVisualizer'
]
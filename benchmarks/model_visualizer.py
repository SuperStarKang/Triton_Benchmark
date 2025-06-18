"""
벤치마크 결과 시각화 도구 (메모리 벤치마킹에 특화)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import os


# 시각화 스타일 설정
plt.style.use('default')  # seaborn-v0_8 대신 default 사용
sns.set_palette("husl")


class BenchmarkVisualizer:
    """벤치마크 결과 시각화 클래스 (메모리 벤치마킹 특화)"""
    
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
        improvements = []
        
        for op_name, result in memory_results.items():
            if isinstance(result, dict) and "naive" in result and "optimized" in result:
                operations.append(op_name)
                
                # Peak memory 추출
                naive_peak_val = result["naive"].get("avg_peak_memory_gb", 0)
                optimized_peak_val = result["optimized"].get("avg_peak_memory_gb", 0)
                naive_peak.append(naive_peak_val)
                optimized_peak.append(optimized_peak_val)
                
                # Average memory 추출
                naive_avg_val = result["naive"].get("avg_avg_memory_gb", 0)
                optimized_avg_val = result["optimized"].get("avg_avg_memory_gb", 0)
                naive_avg.append(naive_avg_val)
                optimized_avg.append(optimized_avg_val)
                
                # 개선율 계산
                if naive_peak_val > 0:
                    improvement = ((naive_peak_val - optimized_peak_val) / naive_peak_val) * 100
                    improvements.append(improvement)
                else:
                    improvements.append(0)
        
        if not operations:
            print("No memory data to plot")
            return
        
        # 짧은 operation 이름으로 변환
        short_ops = [op.replace("SimpleLanguageModel", "SLM")[:12] for op in operations]
        
        # Figure 생성 - 여백을 충분히 확보
        fig = plt.figure(figsize=(18, 10))
        
        # GridSpec을 사용하여 레이아웃 조정 (여백 확보)
        gs = fig.add_gridspec(2, 2, 
                             left=0.1, right=0.95, 
                             top=0.9, bottom=0.15,
                             hspace=0.4, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])  # 하단에 개선율 그래프
        
        x = np.arange(len(operations))
        width = 0.35
        
        # Peak Memory 그래프
        bars1 = ax1.bar(x - width/2, naive_peak, width, label='Naive', alpha=0.8, color='#ff7f0e')
        bars2 = ax1.bar(x + width/2, optimized_peak, width, label='Optimized', alpha=0.8, color='#2ca02c')
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Peak Memory (GB)')
        ax1.set_title('Peak Memory Usage Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(short_ops, rotation=0, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Peak Memory 값 표시
        for i, (naive, opt) in enumerate(zip(naive_peak, optimized_peak)):
            if naive > 0 and opt > 0:
                ax1.text(i - width/2, naive + max(naive_peak) * 0.02, f'{naive:.2f}', 
                        ha='center', va='bottom', fontsize=9)
                ax1.text(i + width/2, opt + max(naive_peak) * 0.02, f'{opt:.2f}', 
                        ha='center', va='bottom', fontsize=9)
        
        # Average Memory 그래프
        bars3 = ax2.bar(x - width/2, naive_avg, width, label='Naive', alpha=0.8, color='#ff7f0e')
        bars4 = ax2.bar(x + width/2, optimized_avg, width, label='Optimized', alpha=0.8, color='#2ca02c')
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Average Memory (GB)')
        ax2.set_title('Average Memory Usage Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(short_ops, rotation=0, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Average Memory 값 표시
        for i, (naive, opt) in enumerate(zip(naive_avg, optimized_avg)):
            if naive > 0 and opt > 0:
                ax2.text(i - width/2, naive + max(naive_avg) * 0.02, f'{naive:.2f}', 
                        ha='center', va='bottom', fontsize=9)
                ax2.text(i + width/2, opt + max(naive_avg) * 0.02, f'{opt:.2f}', 
                        ha='center', va='bottom', fontsize=9)
        
        # 개선율 그래프
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars5 = ax3.bar(x, improvements, color=colors, alpha=0.8)
        
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Memory Improvement (%)')
        ax3.set_title('Memory Usage Improvement (Peak Memory)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(short_ops, rotation=0, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # 개선율 값 표시
        for i, imp in enumerate(improvements):
            if imp != 0:
                y_offset = abs(imp) * 0.1 + 1
                ax3.text(i, imp + (y_offset if imp >= 0 else -y_offset), f'{imp:.1f}%', 
                        ha='center', va='bottom' if imp >= 0 else 'top', 
                        fontweight='bold', fontsize=10)
        
        plt.savefig(os.path.join(self.output_dir, f"{save_name}.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(
        self,
        model_memory_results: Dict[str, Any],
        model_speed_results: Optional[Dict[str, Any]] = None,
        save_name: str = "model_comparison"
    ):
        """모델 전체 비교 그래프 (메모리 중심)"""
        # 데이터 준비
        models = []
        memory_improvements = []
        peak_memory_naive = []
        peak_memory_opt = []
        avg_memory_naive = []
        avg_memory_opt = []
        
        for model_name, result in model_memory_results.items():
            if isinstance(result, dict):
                models.append(model_name)
                
                # 메모리 데이터 추출
                naive_data = result.get("naive", {})
                opt_data = result.get("optimized", {})
                
                # Peak memory 데이터 (forward pass 기준)
                naive_peak = naive_data.get("avg_peak_memory_gb", 0)
                opt_peak = opt_data.get("avg_peak_memory_gb", 0)
                peak_memory_naive.append(naive_peak)
                peak_memory_opt.append(opt_peak)
                
                # Average memory 데이터
                naive_avg = naive_data.get("avg_avg_memory_gb", 0)
                opt_avg = opt_data.get("avg_avg_memory_gb", 0)
                avg_memory_naive.append(naive_avg)
                avg_memory_opt.append(opt_avg)
                
                # 개선율 계산 (peak memory 기준)
                if naive_peak > 0:
                    improvement = ((naive_peak - opt_peak) / naive_peak) * 100
                    memory_improvements.append(improvement)
                else:
                    memory_improvements.append(0)
        
        if not models:
            print("No model comparison data to plot")
            return
        
        # 긴 모델 이름 줄임
        short_models = [model.replace("SimpleLanguageModel", "SLM")[:15] for model in models]
        
        # Figure 생성 - 충분한 여백 확보
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2,
                             left=0.1, right=0.95,
                             top=0.9, bottom=0.15,
                             hspace=0.4, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        
        x = np.arange(len(models))
        width = 0.35
        
        # Peak Memory 비교
        bars1 = ax1.bar(x - width/2, peak_memory_naive, width, label='Naive', alpha=0.8, color='#ff7f0e')
        bars2 = ax1.bar(x + width/2, peak_memory_opt, width, label='Optimized', alpha=0.8, color='#2ca02c')
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Peak Memory (GB)')
        ax1.set_title('Peak Memory Usage')
        ax1.set_xticks(x)
        ax1.set_xticklabels(short_models, rotation=0, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 값 표시
        for i, (naive, opt) in enumerate(zip(peak_memory_naive, peak_memory_opt)):
            if naive > 0:
                ax1.text(i - width/2, naive + max(peak_memory_naive) * 0.02, f'{naive:.2f}', 
                        ha='center', va='bottom', fontsize=9)
            if opt > 0:
                ax1.text(i + width/2, opt + max(peak_memory_naive) * 0.02, f'{opt:.2f}', 
                        ha='center', va='bottom', fontsize=9)
        
        # Average Memory 비교
        bars3 = ax2.bar(x - width/2, avg_memory_naive, width, label='Naive', alpha=0.8, color='#ff7f0e')
        bars4 = ax2.bar(x + width/2, avg_memory_opt, width, label='Optimized', alpha=0.8, color='#2ca02c')
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Average Memory (GB)')
        ax2.set_title('Average Memory Usage')
        ax2.set_xticks(x)
        ax2.set_xticklabels(short_models, rotation=0, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 값 표시
        for i, (naive, opt) in enumerate(zip(avg_memory_naive, avg_memory_opt)):
            if naive > 0:
                ax2.text(i - width/2, naive + max(avg_memory_naive) * 0.02, f'{naive:.2f}', 
                        ha='center', va='bottom', fontsize=9)
            if opt > 0:
                ax2.text(i + width/2, opt + max(avg_memory_naive) * 0.02, f'{opt:.2f}', 
                        ha='center', va='bottom', fontsize=9)
        
        # 메모리 개선율 그래프
        colors = ['green' if imp > 0 else 'red' for imp in memory_improvements]
        bars5 = ax3.bar(x, memory_improvements, color=colors, alpha=0.8)
        
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Memory Improvement (%)')
        ax3.set_title('Memory Usage Improvement (Peak Memory Based)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(short_models, rotation=0, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # 개선율 값 표시
        for i, imp in enumerate(memory_improvements):
            if imp != 0:
                y_offset = abs(imp) * 0.1 + 1
                ax3.text(i, imp + (y_offset if imp >= 0 else -y_offset), f'{imp:.1f}%', 
                        ha='center', va='bottom' if imp >= 0 else 'top', 
                        fontweight='bold', fontsize=12)
        
        plt.savefig(os.path.join(self.output_dir, f"{save_name}.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_memory_breakdown(
        self,
        memory_results: Dict[str, Any],
        save_name: str = "memory_breakdown"
    ):
        """메모리 사용량 세부 분석 그래프"""
        if not memory_results:
            print("No memory data to plot")
            return
        
        # 데이터 준비
        models = list(memory_results.keys())
        
        fig = plt.figure(figsize=(16, 6 * len(models)))
        
        for idx, (model_name, result) in enumerate(memory_results.items()):
            if not isinstance(result, dict) or "naive" not in result or "optimized" not in result:
                continue
            
            # 서브플롯 생성 - 충분한 여백 확보
            ax = plt.subplot(len(models), 1, idx + 1)
            
            naive_data = result["naive"]
            opt_data = result["optimized"]
            
            # 메트릭들 준비
            metrics = []
            naive_values = []
            opt_values = []
            
            metric_mapping = {
                "Peak Memory": "avg_peak_memory_gb",
                "Average Memory": "avg_avg_memory_gb",
                "Min Memory": "avg_min_memory_gb",
                "Memory Increase": "avg_memory_increase_gb"
            }
            
            for metric_name, metric_key in metric_mapping.items():
                if metric_key in naive_data and metric_key in opt_data:
                    metrics.append(metric_name)
                    naive_values.append(naive_data[metric_key])
                    opt_values.append(opt_data[metric_key])
            
            if not metrics:
                ax.text(0.5, 0.5, f'No data for {model_name}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{model_name} - No Data Available')
                continue
            
            x = np.arange(len(metrics))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, naive_values, width, label='Naive', alpha=0.8, color='#ff7f0e')
            bars2 = ax.bar(x + width/2, opt_values, width, label='Optimized', alpha=0.8, color='#2ca02c')
            
            ax.set_xlabel('Memory Metrics')
            ax.set_ylabel('Memory (GB)')
            ax.set_title(f'{model_name} - Memory Usage Breakdown')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics, rotation=0, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 값 표시
            max_val = max(naive_values + opt_values) if naive_values + opt_values else 1
            for i, (naive, opt) in enumerate(zip(naive_values, opt_values)):
                if naive > 0:
                    ax.text(i - width/2, naive + max_val * 0.02, f'{naive:.3f}', 
                           ha='center', va='bottom', fontsize=8)
                if opt > 0:
                    ax.text(i + width/2, opt + max_val * 0.02, f'{opt:.3f}', 
                           ha='center', va='bottom', fontsize=8)
                
                # 개선율 표시
                if naive > 0 and opt >= 0:
                    improvement = ((naive - opt) / naive) * 100
                    y_pos = max(naive, opt) + max_val * 0.08
                    ax.text(i, y_pos, f'{improvement:.1f}%', 
                           ha='center', va='bottom', fontweight='bold', 
                           color='green' if improvement > 0 else 'red', fontsize=9)
        
        plt.tight_layout(pad=3.0)  # 패딩 증가
        plt.savefig(os.path.join(self.output_dir, f"{save_name}.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_summary_report(
        self,
        memory_results: Dict[str, Any],
        save_name: str = "summary_report"
    ):
        """메모리 벤치마크 요약 리포트 생성"""
        if not memory_results:
            print("No data for summary report")
            return
        
        # Figure 생성 - 충분한 여백 확보
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2,
                             left=0.1, right=0.95,
                             top=0.9, bottom=0.15,
                             hspace=0.4, wspace=0.3)
        
        # 1. 전체 메모리 개선율 분포
        ax1 = fig.add_subplot(gs[0, 0])
        
        all_improvements = []
        for result in memory_results.values():
            if isinstance(result, dict) and "improvement" in result:
                for imp in result["improvement"].values():
                    if isinstance(imp, (int, float)) and not np.isnan(imp):
                        all_improvements.append(imp)
        
        if all_improvements:
            ax1.hist(all_improvements, bins=min(10, len(all_improvements)), 
                    alpha=0.7, color='blue', edgecolor='black')
            mean_val = np.mean(all_improvements)
            ax1.axvline(mean_val, color='red', linestyle='--', 
                       label=f'Mean: {mean_val:.1f}%')
            ax1.set_xlabel('Memory Improvement (%)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Distribution of Memory Improvements')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No improvement data available', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Memory Improvements - No Data')
        
        # 2. 메모리 사용량 비교
        ax2 = fig.add_subplot(gs[0, 1])
        
        naive_peaks = []
        opt_peaks = []
        model_names = []
        
        for model_name, result in memory_results.items():
            if isinstance(result, dict) and "naive" in result and "optimized" in result:
                naive_peak = result["naive"].get("avg_peak_memory_gb", 0)
                opt_peak = result["optimized"].get("avg_peak_memory_gb", 0)
                if naive_peak > 0 or opt_peak > 0:
                    naive_peaks.append(naive_peak)
                    opt_peaks.append(opt_peak)
                    model_names.append(model_name.replace("SimpleLanguageModel", "SLM")[:10])
        
        if naive_peaks and opt_peaks:
            x = np.arange(len(model_names))
            width = 0.35
            
            ax2.bar(x - width/2, naive_peaks, width, label='Naive', alpha=0.8, color='#ff7f0e')
            ax2.bar(x + width/2, opt_peaks, width, label='Optimized', alpha=0.8, color='#2ca02c')
            
            ax2.set_xlabel('Models')
            ax2.set_ylabel('Peak Memory (GB)')
            ax2.set_title('Peak Memory Usage Comparison')
            ax2.set_xticks(x)
            ax2.set_xticklabels(model_names, rotation=0, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No memory usage data available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Memory Usage - No Data')
        
        # 3. 요약 테이블
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis('tight')
        ax3.axis('off')
        
        # 요약 통계 계산
        summary_data = []
        
        if all_improvements:
            summary_data.append([
                "Memory Improvement",
                f"{np.mean(all_improvements):.1f}%",
                f"{np.max(all_improvements):.1f}%",
                f"{np.min(all_improvements):.1f}%"
            ])
        
        if naive_peaks and opt_peaks:
            total_naive = sum(naive_peaks)
            total_opt = sum(opt_peaks)
            total_improvement = ((total_naive - total_opt) / total_naive) * 100 if total_naive > 0 else 0
            
            summary_data.append([
                "Total Memory Usage",
                f"{total_naive:.2f} GB → {total_opt:.2f} GB",
                f"{total_improvement:.1f}% reduction",
                f"{len(model_names)} model(s)"
            ])
        
        if summary_data:
            table = ax3.table(cellText=summary_data,
                            colLabels=["Metric", "Value/Average", "Best/Max", "Additional Info"],
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 2)
            ax3.set_title('Memory Benchmark Summary', pad=20, fontsize=14)
        else:
            ax3.text(0.5, 0.5, 'No summary data available', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Benchmark Summary - No Data', fontsize=14)
        
        plt.savefig(os.path.join(self.output_dir, f"{save_name}.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results_to_csv(
        self,
        memory_results: Dict[str, Any],
        filename: str = "memory_benchmark_results.csv"
    ):
        """메모리 벤치마크 결과를 CSV 파일로 저장"""
        data = []
        
        for name, result in memory_results.items():
            if isinstance(result, dict):
                row = {"model_name": name}
                
                # Naive 데이터
                if "naive" in result:
                    for key, value in result["naive"].items():
                        row[f"naive_{key}"] = value
                
                # Optimized 데이터
                if "optimized" in result:
                    for key, value in result["optimized"].items():
                        row[f"optimized_{key}"] = value
                
                # Improvement 데이터
                if "improvement" in result:
                    for key, value in result["improvement"].items():
                        row[f"improvement_{key}"] = value
                
                data.append(row)
        
        # DataFrame 생성 및 저장
        if data:
            df = pd.DataFrame(data)
            csv_path = os.path.join(self.output_dir, filename)
            df.to_csv(csv_path, index=False)
            print(f"Results saved to {csv_path}")
        else:
            print("No data to save")


__all__ = [
    'BenchmarkVisualizer'
]
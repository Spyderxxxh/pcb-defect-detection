"""
Analyzer Agent - 分析Agent
负责识别性能瓶颈，定位问题根因
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.core.registry import MODEL_REGISTRY


class BottleneckType(Enum):
    """瓶颈类型"""
    COMPUTE_BOUND = "compute_bound"  # 计算瓶颈
    MEMORY_BOUND = "memory_bound"    # 内存瓶颈
    IO_BOUND = "io_bound"            # IO瓶颈
    GPU_UTILIZATION_LOW = "gpu_utilization_low"  # GPU利用率低
    LATENCY_SPIKE = "latency_spike"  # 延迟突增
    THROUGHPUT_DROP = "throughput_drop"  # 吞吐量下降
    STAGE_IMBALANCE = "stage_imbalance"  # Stage不平衡


@dataclass
class Bottleneck:
    """瓶颈信息"""
    type: BottleneckType
    stage: Optional[str]
    severity: float  # 0-1，严重程度
    description: str
    metrics: Dict[str, float]
    recommendation: str


@MODEL_REGISTRY.register()
class AnalyzerAgent:
    """
    分析Agent
    
    分析Collector采集的数据，识别性能瓶颈：
    - 计算瓶颈：GPU利用率高但吞吐量低
    - 内存瓶颈：显存占用高
    - IO瓶颈：GPU利用率低
    - Stage不平衡：某个Stage耗时过长
    
    Args:
        bottleneck_threshold: 瓶颈检测阈值
        analysis_depth: 分析深度 (pipeline, stage, operator)
    """
    
    def __init__(
        self,
        bottleneck_threshold: float = 0.8,
        analysis_depth: str = "stage"
    ):
        self.bottleneck_threshold = bottleneck_threshold
        self.analysis_depth = analysis_depth
        
        # 历史数据用于趋势分析
        self.history: List[Dict] = []
        self.max_history = 100
        
        # 阈值配置
        self.thresholds = {
            'gpu_utilization_high': 85.0,   # GPU利用率高
            'gpu_utilization_low': 30.0,    # GPU利用率低
            'memory_high': 8000.0,          # 显存占用高 (MB)
            'latency_spike_factor': 2.0,    # 延迟突增倍数
            'throughput_drop_factor': 0.7,  # 吞吐量下降比例
            'stage_imbalance_factor': 3.0   # Stage不平衡倍数
        }
    
    def analyze(self, metrics: Dict[str, Any], 
                stage_stats: Optional[Dict] = None) -> List[Bottleneck]:
        """
        分析性能数据，识别瓶颈
        
        Args:
            metrics: 性能指标统计
            stage_stats: Stage级统计
            
        Returns:
            瓶颈列表
        """
        bottlenecks = []
        
        # 保存历史
        self.history.append({
            'metrics': metrics,
            'stage_stats': stage_stats
        })
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # 分析各类瓶颈
        bottlenecks.extend(self._analyze_compute_bound(metrics))
        bottlenecks.extend(self._analyze_memory_bound(metrics))
        bottlenecks.extend(self._analyze_io_bound(metrics))
        bottlenecks.extend(self._analyze_latency_spike(metrics))
        bottlenecks.extend(self._analyze_throughput_drop(metrics))
        
        if stage_stats:
            bottlenecks.extend(self._analyze_stage_imbalance(stage_stats))
        
        # 按严重程度排序
        bottlenecks.sort(key=lambda x: x.severity, reverse=True)
        
        return bottlenecks
    
    def _analyze_compute_bound(self, metrics: Dict) -> List[Bottleneck]:
        """分析计算瓶颈"""
        bottlenecks = []
        
        gpu_util = metrics.get('gpu_utilization', {})
        throughput = metrics.get('throughput', {})
        
        if not gpu_util or not throughput:
            return bottlenecks
        
        avg_gpu = gpu_util.get('mean', 0)
        avg_throughput = throughput.get('mean', 0)
        
        # GPU利用率高但吞吐量低
        if avg_gpu > self.thresholds['gpu_utilization_high']:
            if len(self.history) > 1:
                prev_throughput = self.history[-2]['metrics'].get('throughput', {}).get('mean', avg_throughput)
                if avg_throughput < prev_throughput * self.thresholds['throughput_drop_factor']:
                    severity = min(1.0, avg_gpu / 100.0)
                    bottlenecks.append(Bottleneck(
                        type=BottleneckType.COMPUTE_BOUND,
                        stage=None,
                        severity=severity,
                        description=f"GPU利用率高({avg_gpu:.1f}%)但吞吐量下降",
                        metrics={'gpu_utilization': avg_gpu, 'throughput': avg_throughput},
                        recommendation="考虑模型量化、算子融合或TensorRT优化"
                    ))
        
        return bottlenecks
    
    def _analyze_memory_bound(self, metrics: Dict) -> List[Bottleneck]:
        """分析内存瓶颈"""
        bottlenecks = []
        
        gpu_memory = metrics.get('gpu_memory_mb', {})
        
        if not gpu_memory:
            return bottlenecks
        
        max_memory = gpu_memory.get('max', 0)
        avg_memory = gpu_memory.get('mean', 0)
        
        if max_memory > self.thresholds['memory_high']:
            severity = min(1.0, max_memory / 12000.0)  # 假设12GB为上限
            bottlenecks.append(Bottleneck(
                type=BottleneckType.MEMORY_BOUND,
                stage=None,
                severity=severity,
                description=f"显存占用过高(峰值{max_memory:.0f}MB, 平均{avg_memory:.0f}MB)",
                metrics={'max_memory_mb': max_memory, 'avg_memory_mb': avg_memory},
                recommendation="减小batch_size、使用混合精度训练或梯度检查点"
            ))
        
        return bottlenecks
    
    def _analyze_io_bound(self, metrics: Dict) -> List[Bottleneck]:
        """分析IO瓶颈"""
        bottlenecks = []
        
        gpu_util = metrics.get('gpu_utilization', {})
        
        if not gpu_util:
            return bottlenecks
        
        avg_gpu = gpu_util.get('mean', 0)
        
        # GPU利用率低可能是IO瓶颈
        if avg_gpu < self.thresholds['gpu_utilization_low']:
            severity = 1.0 - (avg_gpu / self.thresholds['gpu_utilization_low'])
            bottlenecks.append(Bottleneck(
                type=BottleneckType.IO_BOUND,
                stage=None,
                severity=severity,
                description=f"GPU利用率低({avg_gpu:.1f}%)，可能存在IO瓶颈",
                metrics={'gpu_utilization': avg_gpu},
                recommendation="增加num_workers、使用pin_memory或预加载数据"
            ))
        
        return bottlenecks
    
    def _analyze_latency_spike(self, metrics: Dict) -> List[Bottleneck]:
        """分析延迟突增"""
        bottlenecks = []
        
        latency = metrics.get('latency_ms', {})
        
        if not latency:
            return bottlenecks
        
        current_p95 = latency.get('p95', 0)
        
        if len(self.history) > 5:
            # 计算历史平均p95
            historical_p95 = []
            for h in self.history[-10:-1]:
                h_latency = h['metrics'].get('latency_ms', {})
                if h_latency:
                    historical_p95.append(h_latency.get('p95', current_p95))
            
            if historical_p95:
                avg_historical = np.mean(historical_p95)
                if current_p95 > avg_historical * self.thresholds['latency_spike_factor']:
                    severity = min(1.0, (current_p95 / avg_historical - 1) / 2)
                    bottlenecks.append(Bottleneck(
                        type=BottleneckType.LATENCY_SPIKE,
                        stage=None,
                        severity=severity,
                        description=f"P95延迟突增: {current_p95:.1f}ms (历史平均{avg_historical:.1f}ms)",
                        metrics={'current_p95': current_p95, 'historical_p95': avg_historical},
                        recommendation="检查是否有内存泄漏、GC频率或后台进程干扰"
                    ))
        
        return bottlenecks
    
    def _analyze_throughput_drop(self, metrics: Dict) -> List[Bottleneck]:
        """分析吞吐量下降"""
        bottlenecks = []
        
        throughput = metrics.get('throughput', {})
        
        if not throughput:
            return bottlenecks
        
        current = throughput.get('mean', 0)
        
        if len(self.history) > 5:
            historical = []
            for h in self.history[-10:-1]:
                h_throughput = h['metrics'].get('throughput', {})
                if h_throughput:
                    historical.append(h_throughput.get('mean', current))
            
            if historical:
                avg_historical = np.mean(historical)
                if current < avg_historical * self.thresholds['throughput_drop_factor']:
                    severity = 1.0 - (current / avg_historical)
                    bottlenecks.append(Bottleneck(
                        type=BottleneckType.THROUGHPUT_DROP,
                        stage=None,
                        severity=severity,
                        description=f"吞吐量下降: {current:.1f} (历史平均{avg_historical:.1f})",
                        metrics={'current': current, 'historical': avg_historical},
                        recommendation="检查系统资源、批处理大小或模型复杂度"
                    ))
        
        return bottlenecks
    
    def _analyze_stage_imbalance(self, stage_stats: Dict) -> List[Bottleneck]:
        """分析Stage不平衡"""
        bottlenecks = []
        
        if not stage_stats:
            return bottlenecks
        
        # 计算各Stage平均延迟
        stage_latencies = {}
        for stage_name, stats in stage_stats.items():
            stage_latencies[stage_name] = stats.get('avg_latency_ms', 0)
        
        if not stage_latencies:
            return bottlenecks
        
        # 找出最慢的Stage
        total_latency = sum(stage_latencies.values())
        if total_latency == 0:
            return bottlenecks
        
        for stage_name, latency in stage_latencies.items():
            ratio = latency / total_latency
            
            # 如果某个Stage占比过高
            if ratio > 0.5:  # 超过50%
                severity = min(1.0, ratio)
                bottlenecks.append(Bottleneck(
                    type=BottleneckType.STAGE_IMBALANCE,
                    stage=stage_name,
                    severity=severity,
                    description=f"Stage '{stage_name}' 耗时占比过高({ratio*100:.1f}%, {latency:.1f}ms)",
                    metrics={'stage_latency': latency, 'total_latency': total_latency, 'ratio': ratio},
                    recommendation=f"优化'{stage_name}'阶段，考虑并行化或算法改进"
                ))
        
        return bottlenecks
    
    def generate_report(self, bottlenecks: List[Bottleneck]) -> Dict[str, Any]:
        """生成分析报告"""
        if not bottlenecks:
            return {
                'status': 'healthy',
                'message': '未检测到明显性能瓶颈',
                'bottlenecks': [],
                'recommendations': []
            }
        
        # 按类型分组
        by_type = {}
        for b in bottlenecks:
            if b.type.value not in by_type:
                by_type[b.type.value] = []
            by_type[b.type.value].append(b)
        
        # 提取推荐
        recommendations = list(set([b.recommendation for b in bottlenecks]))
        
        return {
            'status': 'critical' if any(b.severity > 0.8 for b in bottlenecks) else 'warning',
            'message': f"检测到{len(bottlenecks)}个性能瓶颈",
            'bottlenecks': [
                {
                    'type': b.type.value,
                    'stage': b.stage,
                    'severity': b.severity,
                    'description': b.description,
                    'metrics': b.metrics
                }
                for b in bottlenecks
            ],
            'recommendations': recommendations,
            'by_type': {k: len(v) for k, v in by_type.items()}
        }


if __name__ == "__main__":
    # 测试
    agent = AnalyzerAgent()
    
    # 模拟数据
    metrics = {
        'gpu_utilization': {'mean': 90.0, 'max': 95.0},
        'gpu_memory_mb': {'mean': 9000.0, 'max': 10000.0},
        'throughput': {'mean': 100.0},
        'latency_ms': {'p95': 50.0, 'mean': 30.0}
    }
    
    stage_stats = {
        'preprocess': {'avg_latency_ms': 10.0},
        'inference': {'avg_latency_ms': 80.0},
        'postprocess': {'avg_latency_ms': 5.0}
    }
    
    bottlenecks = agent.analyze(metrics, stage_stats)
    
    print("Detected Bottlenecks:")
    for b in bottlenecks:
        print(f"  - {b.type.value}: {b.description} (severity: {b.severity:.2f})")
    
    report = agent.generate_report(bottlenecks)
    print("\nReport:")
    print(report)

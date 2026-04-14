"""
Collector Agent - 数据采集Agent
负责采集Pipeline各阶段的性能指标
"""

import time
import psutil
import torch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import deque
import threading
import numpy as np

from src.core.registry import MODEL_REGISTRY


@dataclass
class PerformanceMetrics:
    """性能指标数据结构"""
    timestamp: float
    latency_ms: float
    memory_mb: float
    gpu_utilization: float
    gpu_memory_mb: float
    fps: float
    throughput: float
    cpu_percent: float
    
    # Stage级指标
    stage_latencies: Dict[str, float] = field(default_factory=dict)
    stage_memory: Dict[str, float] = field(default_factory=dict)


@MODEL_REGISTRY.register()
class CollectorAgent:
    """
    数据采集Agent
    
    持续监控Pipeline性能，采集以下指标：
    - 延迟 (latency)
    - 内存占用 (memory)
    - GPU利用率 (gpu_utilization)
    - GPU显存 (gpu_memory)
    - FPS (fps)
    - 吞吐量 (throughput)
    - Stage级延迟和内存
    
    Args:
        sampling_interval: 采样间隔（秒）
        buffer_size: 数据缓冲区大小
        metrics: 要采集的指标列表
    """
    
    def __init__(
        self,
        sampling_interval: float = 1.0,
        buffer_size: int = 1000,
        metrics: Optional[List[str]] = None
    ):
        self.sampling_interval = sampling_interval
        self.buffer_size = buffer_size
        self.metrics = metrics or [
            'latency', 'memory', 'gpu_utilization', 
            'gpu_memory', 'fps', 'throughput', 'cpu_percent'
        ]
        
        # 数据缓冲区
        self.metrics_buffer: deque = deque(maxlen=buffer_size)
        self.stage_metrics: Dict[str, List[Dict]] = {}
        
        # 运行状态
        self.is_collecting = False
        self.collection_thread: Optional[threading.Thread] = None
        
        # 统计信息
        self.total_samples = 0
        self.start_time: Optional[float] = None
        
        # 阶段计时器
        self.stage_timers: Dict[str, float] = {}
        self.stage_memory_snapshots: Dict[str, float] = {}
    
    def start_collection(self):
        """开始采集"""
        if self.is_collecting:
            return
        
        self.is_collecting = True
        self.start_time = time.time()
        
        # 启动采集线程
        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        
        print(f"[CollectorAgent] Started collection with interval {self.sampling_interval}s")
    
    def stop_collection(self):
        """停止采集"""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=2.0)
        
        print(f"[CollectorAgent] Stopped collection. Total samples: {self.total_samples}")
    
    def _collection_loop(self):
        """采集循环"""
        while self.is_collecting:
            metrics = self._collect_once()
            self.metrics_buffer.append(metrics)
            self.total_samples += 1
            
            time.sleep(self.sampling_interval)
    
    def _collect_once(self) -> PerformanceMetrics:
        """采集一次数据"""
        timestamp = time.time()
        
        # CPU和内存
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        memory_mb = memory.used / 1024 / 1024
        
        # GPU信息
        gpu_utilization = 0.0
        gpu_memory_mb = 0.0
        
        if torch.cuda.is_available():
            gpu_utilization = torch.cuda.utilization()
            gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        
        # 计算FPS和吞吐量（基于历史数据）
        fps = self._calculate_fps()
        throughput = self._calculate_throughput()
        
        # 延迟（基于最近处理时间）
        latency_ms = self._estimate_latency()
        
        return PerformanceMetrics(
            timestamp=timestamp,
            latency_ms=latency_ms,
            memory_mb=memory_mb,
            gpu_utilization=gpu_utilization,
            gpu_memory_mb=gpu_memory_mb,
            fps=fps,
            throughput=throughput,
            cpu_percent=cpu_percent,
            stage_latencies=self._get_stage_latencies(),
            stage_memory=self._get_stage_memory()
        )
    
    def _calculate_fps(self) -> float:
        """计算FPS"""
        if len(self.metrics_buffer) < 2:
            return 0.0
        
        recent = list(self.metrics_buffer)[-10:]
        if len(recent) < 2:
            return 0.0
        
        time_span = recent[-1].timestamp - recent[0].timestamp
        if time_span > 0:
            return len(recent) / time_span
        return 0.0
    
    def _calculate_throughput(self) -> float:
        """计算吞吐量（样本/秒）"""
        # 简化为FPS * batch_size
        fps = self._calculate_fps()
        return fps * 16  # 假设batch_size=16
    
    def _estimate_latency(self) -> float:
        """估计延迟"""
        # 基于Stage延迟求和
        stage_latencies = self._get_stage_latencies()
        if stage_latencies:
            return sum(stage_latencies.values()) * 1000  # 转为ms
        return 0.0
    
    def start_stage_timer(self, stage_name: str):
        """开始Stage计时"""
        self.stage_timers[stage_name] = time.time()
        
        # 记录内存快照
        if torch.cuda.is_available():
            self.stage_memory_snapshots[stage_name] = torch.cuda.memory_allocated()
        else:
            self.stage_memory_snapshots[stage_name] = psutil.virtual_memory().used
    
    def end_stage_timer(self, stage_name: str) -> float:
        """结束Stage计时，返回耗时（秒）"""
        if stage_name not in self.stage_timers:
            return 0.0
        
        elapsed = time.time() - self.stage_timers[stage_name]
        
        # 记录Stage指标
        if stage_name not in self.stage_metrics:
            self.stage_metrics[stage_name] = []
        
        # 计算内存增量
        memory_delta = 0.0
        if stage_name in self.stage_memory_snapshots:
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated()
            else:
                current_memory = psutil.virtual_memory().used
            memory_delta = (current_memory - self.stage_memory_snapshots[stage_name]) / 1024 / 1024
        
        self.stage_metrics[stage_name].append({
            'timestamp': time.time(),
            'latency': elapsed,
            'memory_delta_mb': memory_delta
        })
        
        # 清理旧数据
        if len(self.stage_metrics[stage_name]) > self.buffer_size:
            self.stage_metrics[stage_name].pop(0)
        
        return elapsed
    
    def _get_stage_latencies(self) -> Dict[str, float]:
        """获取各Stage最近延迟"""
        result = {}
        for stage_name, metrics in self.stage_metrics.items():
            if metrics:
                result[stage_name] = metrics[-1]['latency']
        return result
    
    def _get_stage_memory(self) -> Dict[str, float]:
        """获取各Stage内存占用"""
        result = {}
        for stage_name, metrics in self.stage_metrics.items():
            if metrics:
                result[stage_name] = metrics[-1]['memory_delta_mb']
        return result
    
    def get_recent_metrics(self, n: int = 10) -> List[PerformanceMetrics]:
        """获取最近n条指标"""
        return list(self.metrics_buffer)[-n:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.metrics_buffer:
            return {}
        
        metrics_list = list(self.metrics_buffer)
        
        def calc_stats(values):
            if not values:
                return {}
            return {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'p50': np.percentile(values, 50),
                'p95': np.percentile(values, 95),
                'p99': np.percentile(values, 99)
            }
        
        return {
            'latency_ms': calc_stats([m.latency_ms for m in metrics_list]),
            'memory_mb': calc_stats([m.memory_mb for m in metrics_list]),
            'gpu_utilization': calc_stats([m.gpu_utilization for m in metrics_list]),
            'fps': calc_stats([m.fps for m in metrics_list]),
            'throughput': calc_stats([m.throughput for m in metrics_list]),
            'total_samples': self.total_samples,
            'duration_seconds': time.time() - self.start_time if self.start_time else 0
        }
    
    def get_stage_statistics(self) -> Dict[str, Dict]:
        """获取Stage级统计"""
        result = {}
        for stage_name, metrics in self.stage_metrics.items():
            if metrics:
                latencies = [m['latency'] for m in metrics]
                memory_deltas = [m['memory_delta_mb'] for m in metrics]
                
                result[stage_name] = {
                    'avg_latency_ms': np.mean(latencies) * 1000,
                    'max_latency_ms': np.max(latencies) * 1000,
                    'avg_memory_delta_mb': np.mean(memory_deltas),
                    'call_count': len(metrics)
                }
        return result
    
    def reset(self):
        """重置所有数据"""
        self.metrics_buffer.clear()
        self.stage_metrics.clear()
        self.stage_timers.clear()
        self.stage_memory_snapshots.clear()
        self.total_samples = 0
        self.start_time = None


if __name__ == "__main__":
    # 测试
    agent = CollectorAgent(sampling_interval=0.5)
    agent.start_collection()
    
    # 模拟一些Stage
    for i in range(5):
        agent.start_stage_timer('preprocess')
        time.sleep(0.1)
        agent.end_stage_timer('preprocess')
        
        agent.start_stage_timer('inference')
        time.sleep(0.2)
        agent.end_stage_timer('inference')
        
        time.sleep(0.5)
    
    agent.stop_collection()
    
    print("\nStatistics:")
    print(agent.get_statistics())
    print("\nStage Statistics:")
    print(agent.get_stage_statistics())

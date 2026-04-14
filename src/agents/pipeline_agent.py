"""
Pipeline Agent System - 4-Agent协作系统主入口
整合Collector、Analyzer、Optimizer、Executor四个Agent
"""

import time
import threading
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

from src.core.config import Config
from src.core.registry import MODEL_REGISTRY
from .collector_agent import CollectorAgent
from .analyzer_agent import AnalyzerAgent
from .optimizer_agent import OptimizerAgent
from .executor_agent import ExecutorAgent


@dataclass
class PipelineState:
    """Pipeline状态"""
    is_running: bool = False
    current_mode: str = 'balanced'  # high_precision, balanced, high_speed
    optimization_round: int = 0
    last_bottlenecks: Optional[list] = None
    last_plans: Optional[list] = None


@MODEL_REGISTRY.register()
class PipelineAgentSystem:
    """
    4-Agent协作系统
    
    工作流：
    1. CollectorAgent: 持续采集性能指标
    2. AnalyzerAgent: 分析瓶颈
    3. OptimizerAgent: 生成优化方案
    4. ExecutorAgent: 执行方案并A/B测试
    
    支持三种模式：
    - high_precision: 高精度模式（牺牲速度）
    - balanced: 均衡模式（默认）
    - high_speed: 高速模式（牺牲精度）
    
    Args:
        config: 系统配置
        test_function: 测试函数，用于A/B测试
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        test_function: Optional[Callable] = None
    ):
        self.config = Config(config)
        self.test_function = test_function
        
        # 初始化4个Agent
        agent_cfg = self.config.get('agent', {})
        
        self.collector = CollectorAgent(
            sampling_interval=agent_cfg.get('sampling_interval', 1.0),
            buffer_size=1000
        )
        
        self.analyzer = AnalyzerAgent(
            bottleneck_threshold=agent_cfg.get('bottleneck_threshold', 0.8),
            analysis_depth=agent_cfg.get('analysis_depth', 'stage')
        )
        
        self.optimizer = OptimizerAgent(
            strategies=agent_cfg.get('strategies', [
                'batch_size_tuning',
                'precision_calibration',
                'tensorrt_optimization',
                'memory_optimization'
            ])
        )
        
        self.executor = ExecutorAgent(
            auto_apply=agent_cfg.get('auto_apply', False),
            require_approval=agent_cfg.get('require_approval', True),
            ab_test_config=agent_cfg.get('ab_test', {})
        )
        
        # 状态
        self.state = PipelineState()
        self.monitor_thread: Optional[threading.Thread] = None
        
        # 模式配置
        self.mode_configs = {
            'high_precision': {
                'model.conf_threshold': 0.15,
                'model.nms_threshold': 0.45,
                'model.sam_refinement.enabled': True,
                'inference.batch_size': 8
            },
            'balanced': {
                'model.conf_threshold': 0.25,
                'model.nms_threshold': 0.45,
                'model.sam_refinement.enabled': True,
                'inference.batch_size': 16
            },
            'high_speed': {
                'model.conf_threshold': 0.35,
                'model.nms_threshold': 0.50,
                'model.sam_refinement.enabled': False,
                'inference.batch_size': 32,
                'deployment.tensorrt.enabled': True,
                'deployment.tensorrt.fp16': True
            }
        }
    
    def start(self):
        """启动Agent系统"""
        print("[PipelineAgentSystem] Starting 4-Agent system...")
        
        self.state.is_running = True
        
        # 启动Collector
        self.collector.start_collection()
        
        # 启动监控循环
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        print("[PipelineAgentSystem] System started successfully")
    
    def stop(self):
        """停止Agent系统"""
        print("[PipelineAgentSystem] Stopping system...")
        
        self.state.is_running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        self.collector.stop_collection()
        
        print("[PipelineAgentSystem] System stopped")
    
    def _monitor_loop(self):
        """监控循环"""
        check_interval = self.config.get('agent', {}).get('check_interval', 30)
        
        while self.state.is_running:
            time.sleep(check_interval)
            
            if not self.state.is_running:
                break
            
            # 执行一次优化循环
            self._optimization_cycle()
    
    def _optimization_cycle(self):
        """单次优化循环"""
        print(f"\n[PipelineAgentSystem] Optimization cycle #{self.state.optimization_round + 1}")
        
        # 1. 获取Collector数据
        stats = self.collector.get_statistics()
        stage_stats = self.collector.get_stage_statistics()
        
        print(f"[Collector] Current FPS: {stats.get('fps', {}).get('mean', 0):.1f}")
        print(f"[Collector] Latency: {stats.get('latency_ms', {}).get('mean', 0):.1f}ms")
        
        # 2. Analyzer分析瓶颈
        bottlenecks = self.analyzer.analyze(stats, stage_stats)
        self.state.last_bottlenecks = bottlenecks
        
        if not bottlenecks:
            print("[Analyzer] No significant bottlenecks detected")
            return
        
        print(f"[Analyzer] Detected {len(bottlenecks)} bottlenecks:")
        for b in bottlenecks[:3]:  # 只显示前3个
            print(f"  - {b.type.value}: {b.description}")
        
        # 3. Optimizer生成方案
        current_config = self.config.to_dict()
        plans = self.optimizer.generate_plans(bottlenecks, current_config)
        self.state.last_plans = plans
        
        if not plans:
            print("[Optimizer] No optimization plans generated")
            return
        
        print(f"[Optimizer] Generated {len(plans)} optimization plans")
        
        # 4. Executor执行方案
        if self.test_function:
            for plan in plans[:1]:  # 一次只执行一个方案
                print(f"[Executor] Executing plan: {plan.strategy.value}")
                
                result = self.executor.execute(
                    plan,
                    current_config,
                    self.test_function
                )
                
                print(f"[Executor] Result: {result.status.value} - {result.message}")
                
                if result.status.value == 'success':
                    # 更新配置
                    self._update_config(plan.config_changes)
        
        self.state.optimization_round += 1
    
    def _update_config(self, changes: Dict[str, Any]):
        """更新配置"""
        for key, value in changes.items():
            keys = key.split('.')
            current = self.config._cfg_dict
            
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            current[keys[-1]] = value
    
    def switch_mode(self, mode: str):
        """
        切换性能模式
        
        Args:
            mode: high_precision, balanced, high_speed
        """
        if mode not in self.mode_configs:
            raise ValueError(f"Unknown mode: {mode}. Available: {list(self.mode_configs.keys())}")
        
        print(f"[PipelineAgentSystem] Switching to {mode} mode")
        
        self.state.current_mode = mode
        config_changes = self.mode_configs[mode]
        
        # 应用配置变更
        self._update_config(config_changes)
        
        print(f"[PipelineAgentSystem] Mode switched to {mode}")
        print(f"  Config changes: {config_changes}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'is_running': self.state.is_running,
            'current_mode': self.state.current_mode,
            'optimization_round': self.state.optimization_round,
            'collector_stats': self.collector.get_statistics(),
            'collector_stage_stats': self.collector.get_stage_statistics(),
            'last_bottlenecks': [
                {
                    'type': b.type.value,
                    'severity': b.severity,
                    'description': b.description
                }
                for b in (self.state.last_bottlenecks or [])
            ],
            'execution_summary': self.executor.get_execution_summary()
        }
    
    def manual_optimize(self) -> Dict[str, Any]:
        """手动触发一次优化"""
        self._optimization_cycle()
        return self.get_status()
    
    def get_recommendations(self) -> list:
        """获取当前优化建议"""
        if not self.state.last_bottlenecks:
            return ["暂无瓶颈，系统运行良好"]
        
        recommendations = []
        for b in self.state.last_bottlenecks:
            recommendations.append(b.recommendation)
        
        return list(set(recommendations))  # 去重


if __name__ == "__main__":
    # 测试
    config = {
        'agent': {
            'sampling_interval': 1.0,
            'check_interval': 10,
            'auto_apply': False,
            'require_approval': True
        }
    }
    
    # 模拟测试函数
    def mock_test(config):
        import random
        return {
            'latency_ms': {'mean': 30 + random.random() * 10},
            'throughput': {'mean': 100 + random.random() * 20},
            'memory_mb': {'mean': 4000 + random.random() * 500},
            'gpu_utilization': {'mean': 70 + random.random() * 20}
        }
    
    system = PipelineAgentSystem(config, mock_test)
    
    # 启动系统
    system.start()
    
    # 模拟一些Stage计时
    import time
    for i in range(3):
        system.collector.start_stage_timer('preprocess')
        time.sleep(0.1)
        system.collector.end_stage_timer('preprocess')
        
        system.collector.start_stage_timer('inference')
        time.sleep(0.2)
        system.collector.end_stage_timer('inference')
        
        time.sleep(1)
    
    # 手动触发优化
    system.manual_optimize()
    
    # 获取状态
    status = system.get_status()
    print(f"\nSystem Status:")
    print(f"  Mode: {status['current_mode']}")
    print(f"  Optimization rounds: {status['optimization_round']}")
    
    # 停止系统
    system.stop()

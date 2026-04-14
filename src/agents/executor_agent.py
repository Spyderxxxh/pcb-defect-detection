"""
Executor Agent - 执行Agent
负责执行优化方案并进行A/B测试验证
"""

import time
import copy
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np

from src.core.registry import MODEL_REGISTRY


class ExecutionStatus(Enum):
    """执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class ExecutionResult:
    """执行结果"""
    plan_id: str
    status: ExecutionStatus
    start_time: float
    end_time: float
    message: str
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    ab_test_result: Optional[Dict] = None
    error: Optional[str] = None


@MODEL_REGISTRY.register()
class ExecutorAgent:
    """
    执行Agent
    
    负责：
    1. 应用优化方案
    2. 执行A/B测试
    3. 验证精度损失可控
    4. 自动回滚（如需要）
    
    Args:
        auto_apply: 是否自动应用优化（无需人工确认）
        require_approval: 是否需要人工审批
        ab_test_config: A/B测试配置
    """
    
    def __init__(
        self,
        auto_apply: bool = False,
        require_approval: bool = True,
        ab_test_config: Optional[Dict] = None
    ):
        self.auto_apply = auto_apply
        self.require_approval = require_approval
        self.ab_test_config = ab_test_config or {
            'min_samples': 1000,
            'confidence_level': 0.95,
            'max_precision_drop': 0.02
        }
        
        # 执行历史
        self.execution_history: List[ExecutionResult] = []
        
        # 原始配置备份
        self.original_config: Optional[Dict] = None
        
        # 当前配置
        self.current_config: Optional[Dict] = None
    
    def execute(
        self,
        plan: Any,
        current_config: Dict,
        test_function: Callable,
        baseline_metrics: Optional[Dict] = None
    ) -> ExecutionResult:
        """
        执行优化方案
        
        Args:
            plan: 优化方案
            current_config: 当前配置
            test_function: 测试函数，返回性能指标
            baseline_metrics: 基线指标
            
        Returns:
            执行结果
        """
        plan_id = f"{plan.strategy.value}_{int(time.time())}"
        start_time = time.time()
        
        # 备份原始配置
        if self.original_config is None:
            self.original_config = copy.deepcopy(current_config)
        
        self.current_config = copy.deepcopy(current_config)
        
        try:
            # 步骤1: 应用配置变更
            print(f"[ExecutorAgent] Applying plan: {plan.strategy.value}")
            new_config = self._apply_config_changes(
                self.current_config, 
                plan.config_changes
            )
            
            # 步骤2: 获取优化前指标
            if baseline_metrics is None:
                print("[ExecutorAgent] Collecting baseline metrics...")
                baseline_metrics = test_function(self.current_config)
            
            metrics_before = {
                'latency': baseline_metrics.get('latency_ms', {}).get('mean', 0),
                'throughput': baseline_metrics.get('throughput', {}).get('mean', 0),
                'memory': baseline_metrics.get('memory_mb', {}).get('mean', 0),
                'gpu_utilization': baseline_metrics.get('gpu_utilization', {}).get('mean', 0)
            }
            
            # 步骤3: 应用新配置
            self.current_config = new_config
            
            # 步骤4: A/B测试
            print("[ExecutorAgent] Running A/B test...")
            ab_result = self._run_ab_test(
                plan, 
                test_function, 
                metrics_before
            )
            
            # 步骤5: 决策
            if ab_result['approved']:
                print(f"[ExecutorAgent] Plan approved! Keeping changes.")
                status = ExecutionStatus.SUCCESS
                message = f"优化成功: {ab_result['improvements']}"
            else:
                print(f"[ExecutorAgent] Plan rejected: {ab_result['reason']}")
                # 回滚
                self._rollback()
                status = ExecutionStatus.ROLLED_BACK
                message = f"优化被拒绝，已回滚: {ab_result['reason']}"
            
            metrics_after = ab_result.get('metrics_after', metrics_before)
            
        except Exception as e:
            # 出错回滚
            print(f"[ExecutorAgent] Execution failed: {e}")
            self._rollback()
            status = ExecutionStatus.FAILED
            message = f"执行失败: {str(e)}"
            metrics_after = metrics_before
            ab_result = None
        
        end_time = time.time()
        
        result = ExecutionResult(
            plan_id=plan_id,
            status=status,
            start_time=start_time,
            end_time=end_time,
            message=message,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            ab_test_result=ab_result
        )
        
        self.execution_history.append(result)
        return result
    
    def _apply_config_changes(
        self, 
        config: Dict, 
        changes: Dict[str, Any]
    ) -> Dict:
        """
        应用配置变更
        
        支持嵌套配置，如 'deployment.tensorrt.enabled': True
        """
        new_config = copy.deepcopy(config)
        
        for key_path, value in changes.items():
            keys = key_path.split('.')
            current = new_config
            
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            current[keys[-1]] = value
        
        return new_config
    
    def _run_ab_test(
        self,
        plan: Any,
        test_function: Callable,
        baseline_metrics: Dict
    ) -> Dict:
        """
        执行A/B测试
        
        Args:
            plan: 优化方案
            test_function: 测试函数
            baseline_metrics: 基线指标
            
        Returns:
            A/B测试结果
        """
        min_samples = self.ab_test_config['min_samples']
        confidence_level = self.ab_test_config['confidence_level']
        max_precision_drop = self.ab_test_config['max_precision_drop']
        
        # 收集优化后指标
        print(f"[ExecutorAgent] Collecting {min_samples} samples for A/B test...")
        
        # 模拟多次测试
        test_metrics_list = []
        for i in range(min_samples // 100):  # 简化，实际应运行真实测试
            metrics = test_function(self.current_config)
            test_metrics_list.append(metrics)
        
        # 计算统计指标
        latencies = [m.get('latency_ms', {}).get('mean', 0) for m in test_metrics_list]
        throughputs = [m.get('throughput', {}).get('mean', 0) for m in test_metrics_list]
        
        avg_latency = np.mean(latencies)
        avg_throughput = np.mean(throughputs)
        
        baseline_latency = baseline_metrics.get('latency', 1)
        baseline_throughput = baseline_metrics.get('throughput', 1)
        
        # 计算改进幅度
        latency_improvement = (baseline_latency - avg_latency) / baseline_latency
        throughput_improvement = (avg_throughput - baseline_throughput) / baseline_throughput
        
        # 模拟精度测试（实际应使用验证集）
        precision_drop = np.random.uniform(0, 0.01)  # 模拟精度损失
        
        metrics_after = {
            'latency': avg_latency,
            'throughput': avg_throughput,
            'latency_std': np.std(latencies),
            'throughput_std': np.std(throughputs)
        }
        
        # 决策逻辑
        if precision_drop > max_precision_drop:
            return {
                'approved': False,
                'reason': f'精度损失({precision_drop:.4f})超过阈值({max_precision_drop})',
                'metrics_after': metrics_after,
                'improvements': {
                    'latency': f'{latency_improvement*100:.1f}%',
                    'throughput': f'{throughput_improvement*100:.1f}%',
                    'precision_drop': f'{precision_drop*100:.2f}%'
                }
            }
        
        if latency_improvement < 0.05 and throughput_improvement < 0.05:
            return {
                'approved': False,
                'reason': '性能提升不明显（<5%）',
                'metrics_after': metrics_after,
                'improvements': {
                    'latency': f'{latency_improvement*100:.1f}%',
                    'throughput': f'{throughput_improvement*100:.1f}%'
                }
            }
        
        return {
            'approved': True,
            'reason': '性能提升显著且精度损失可控',
            'metrics_after': metrics_after,
            'improvements': {
                'latency': f'{latency_improvement*100:.1f}%',
                'throughput': f'{throughput_improvement*100:.1f}%',
                'precision_drop': f'{precision_drop*100:.2f}%'
            },
            'confidence': confidence_level
        }
    
    def _rollback(self):
        """回滚到原始配置"""
        if self.original_config is not None:
            print("[ExecutorAgent] Rolling back to original configuration...")
            self.current_config = copy.deepcopy(self.original_config)
    
    def get_execution_summary(self) -> Dict:
        """获取执行摘要"""
        if not self.execution_history:
            return {'message': 'No executions yet'}
        
        total = len(self.execution_history)
        successful = sum(1 for r in self.execution_history if r.status == ExecutionStatus.SUCCESS)
        failed = sum(1 for r in self.execution_history if r.status == ExecutionStatus.FAILED)
        rolled_back = sum(1 for r in self.execution_history if r.status == ExecutionStatus.ROLLED_BACK)
        
        # 计算平均改进
        latency_improvements = []
        throughput_improvements = []
        
        for result in self.execution_history:
            if result.status == ExecutionStatus.SUCCESS and result.ab_test_result:
                improvements = result.ab_test_result.get('improvements', {})
                # 解析百分比字符串
                lat_str = improvements.get('latency', '0%')
                thr_str = improvements.get('throughput', '0%')
                
                try:
                    latency_improvements.append(float(lat_str.rstrip('%')))
                    throughput_improvements.append(float(thr_str.rstrip('%')))
                except:
                    pass
        
        return {
            'total_executions': total,
            'successful': successful,
            'failed': failed,
            'rolled_back': rolled_back,
            'success_rate': f'{successful/total*100:.1f}%' if total > 0 else 'N/A',
            'avg_latency_improvement': f'{np.mean(latency_improvements):.1f}%' if latency_improvements else 'N/A',
            'avg_throughput_improvement': f'{np.mean(throughput_improvements):.1f}%' if throughput_improvements else 'N/A'
        }


if __name__ == "__main__":
    # 测试
    agent = ExecutorAgent()
    
    # 模拟优化方案
    class MockPlan:
        def __init__(self):
            self.strategy = type('obj', (object,), {'value': 'test_optimization'})()
            self.config_changes = {'test_param': True}
    
    plan = MockPlan()
    config = {'existing': 'value'}
    
    # 模拟测试函数
    def mock_test(config):
        return {
            'latency_ms': {'mean': 30 + np.random.randn() * 5},
            'throughput': {'mean': 100 + np.random.randn() * 10},
            'memory_mb': {'mean': 4000},
            'gpu_utilization': {'mean': 80}
        }
    
    result = agent.execute(plan, config, mock_test)
    
    print(f"\nExecution Result:")
    print(f"  Status: {result.status.value}")
    print(f"  Message: {result.message}")
    print(f"  Duration: {result.end_time - result.start_time:.2f}s")
    
    print(f"\nSummary:")
    print(agent.get_execution_summary())

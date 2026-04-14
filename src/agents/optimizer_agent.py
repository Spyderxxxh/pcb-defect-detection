"""
Optimizer Agent - 优化Agent
负责生成性能优化方案
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from src.core.registry import MODEL_REGISTRY


class OptimizationStrategy(Enum):
    """优化策略类型"""
    BATCH_SIZE_TUNING = "batch_size_tuning"
    PRECISION_CALIBRATION = "precision_calibration"
    TENSORRT_OPTIMIZATION = "tensorrt_optimization"
    MODEL_PRUNING = "model_pruning"
    OPERATOR_FUSION = "operator_fusion"
    MEMORY_OPTIMIZATION = "memory_optimization"
    PARALLELIZATION = "parallelization"
    CACHE_OPTIMIZATION = "cache_optimization"


@dataclass
class OptimizationPlan:
    """优化方案"""
    strategy: OptimizationStrategy
    target_stage: Optional[str]
    priority: int  # 1-10，优先级
    description: str
    config_changes: Dict[str, Any]
    expected_improvement: Dict[str, float]  # 预期改进幅度
    risk_level: str  # low, medium, high
    rollback_plan: str


@MODEL_REGISTRY.register()
class OptimizerAgent:
    """
    优化Agent
    
    根据Analyzer识别的瓶颈，生成优化方案：
    - batch_size调优
    - 精度校准 (FP16/INT8)
    - TensorRT优化
    - 模型剪枝
    - 算子融合
    - 内存优化
    
    Args:
        strategies: 启用的优化策略
        max_concurrent: 最大并发优化数
    """
    
    def __init__(
        self,
        strategies: Optional[List[str]] = None,
        max_concurrent: int = 3
    ):
        self.strategies = strategies or [
            'batch_size_tuning',
            'precision_calibration',
            'tensorrt_optimization',
            'memory_optimization'
        ]
        self.max_concurrent = max_concurrent
        
        # 优化历史
        self.optimization_history: List[Dict] = []
        
        # 策略配置
        self.strategy_configs = {
            'batch_size_tuning': {
                'min_batch': 1,
                'max_batch': 64,
                'step': 2,
                'target_gpu_util': 85.0
            },
            'precision_calibration': {
                'modes': ['fp16', 'int8'],
                'max_precision_drop': 0.02
            },
            'tensorrt_optimization': {
                'fp16': True,
                'int8': False,
                'max_workspace_size': 1073741824
            },
            'model_pruning': {
                'sparsity_levels': [0.1, 0.2, 0.3],
                'max_precision_drop': 0.03
            }
        }
    
    def generate_plans(
        self,
        bottlenecks: List[Any],
        current_config: Dict[str, Any]
    ) -> List[OptimizationPlan]:
        """
        生成优化方案
        
        Args:
            bottlenecks: 瓶颈列表
            current_config: 当前配置
            
        Returns:
            优化方案列表
        """
        plans = []
        
        for bottleneck in bottlenecks:
            if bottleneck.type.value == 'compute_bound':
                plans.extend(self._plan_for_compute_bound(bottleneck, current_config))
            elif bottleneck.type.value == 'memory_bound':
                plans.extend(self._plan_for_memory_bound(bottleneck, current_config))
            elif bottleneck.type.value == 'io_bound':
                plans.extend(self._plan_for_io_bound(bottleneck, current_config))
            elif bottleneck.type.value == 'stage_imbalance':
                plans.extend(self._plan_for_stage_imbalance(bottleneck, current_config))
            elif bottleneck.type.value == 'latency_spike':
                plans.extend(self._plan_for_latency_spike(bottleneck, current_config))
        
        # 按优先级排序
        plans.sort(key=lambda x: x.priority, reverse=True)
        
        return plans[:self.max_concurrent]
    
    def _plan_for_compute_bound(
        self, 
        bottleneck: Any, 
        config: Dict
    ) -> List[OptimizationPlan]:
        """针对计算瓶颈的优化方案"""
        plans = []
        
        # 方案1: TensorRT优化
        if 'tensorrt_optimization' in self.strategies:
            plans.append(OptimizationPlan(
                strategy=OptimizationStrategy.TENSORRT_OPTIMIZATION,
                target_stage=bottleneck.stage,
                priority=9,
                description="使用TensorRT进行推理优化，包括算子融合和内核自动调优",
                config_changes={
                    'deployment.tensorrt.enabled': True,
                    'deployment.tensorrt.fp16': True
                },
                expected_improvement={
                    'latency': 0.30,  # 预期减少30%
                    'throughput': 0.40  # 预期提升40%
                },
                risk_level='low',
                rollback_plan='禁用TensorRT，回退到原始PyTorch模型'
            ))
        
        # 方案2: FP16混合精度
        if 'precision_calibration' in self.strategies:
            plans.append(OptimizationPlan(
                strategy=OptimizationStrategy.PRECISION_CALIBRATION,
                target_stage=bottleneck.stage,
                priority=8,
                description="启用FP16混合精度推理，减少计算量",
                config_changes={
                    'training.fp16': True,
                    'deployment.tensorrt.fp16': True
                },
                expected_improvement={
                    'latency': 0.20,
                    'memory': 0.30
                },
                risk_level='low',
                rollback_plan='禁用FP16，使用FP32推理'
            ))
        
        # 方案3: 模型剪枝
        if 'model_pruning' in self.strategies:
            plans.append(OptimizationPlan(
                strategy=OptimizationStrategy.MODEL_PRUNING,
                target_stage=bottleneck.stage,
                priority=6,
                description="对模型进行结构化剪枝，减少参数量和计算量",
                config_changes={
                    'model.pruning.enabled': True,
                    'model.pruning.sparsity': 0.2
                },
                expected_improvement={
                    'latency': 0.25,
                    'model_size': 0.30
                },
                risk_level='medium',
                rollback_plan='加载原始未剪枝模型权重'
            ))
        
        return plans
    
    def _plan_for_memory_bound(
        self, 
        bottleneck: Any, 
        config: Dict
    ) -> List[OptimizationPlan]:
        """针对内存瓶颈的优化方案"""
        plans = []
        
        current_batch = config.get('dataset', {}).get('batch_size', 16)
        
        # 方案1: 减小batch_size
        if 'batch_size_tuning' in self.strategies and current_batch > 1:
            new_batch = max(1, current_batch // 2)
            plans.append(OptimizationPlan(
                strategy=OptimizationStrategy.BATCH_SIZE_TUNING,
                target_stage=None,
                priority=10,
                description=f"减小batch_size: {current_batch} -> {new_batch}",
                config_changes={
                    'dataset.batch_size': new_batch
                },
                expected_improvement={
                    'memory': 0.40,
                    'latency': -0.05  # 轻微增加
                },
                risk_level='low',
                rollback_plan=f'恢复batch_size为{current_batch}'
            ))
        
        # 方案2: 梯度检查点
        plans.append(OptimizationPlan(
            strategy=OptimizationStrategy.MEMORY_OPTIMIZATION,
            target_stage=None,
            priority=8,
            description="启用梯度检查点，用计算换内存",
            config_changes={
                'training.gradient_checkpointing': True
            },
            expected_improvement={
                'memory': 0.50,
                'latency': 0.10  # 轻微增加
            },
            risk_level='low',
            rollback_plan='禁用梯度检查点'
        ))
        
        # 方案3: INT8量化
        if 'precision_calibration' in self.strategies:
            plans.append(OptimizationPlan(
                strategy=OptimizationStrategy.PRECISION_CALIBRATION,
                target_stage=None,
                priority=7,
                description="使用INT8量化，大幅降低显存占用",
                config_changes={
                    'deployment.quantization.enabled': True,
                    'deployment.quantization.dtype': 'int8'
                },
                expected_improvement={
                    'memory': 0.50,
                    'latency': 0.30
                },
                risk_level='medium',
                rollback_plan='回退到FP16或FP32'
            ))
        
        return plans
    
    def _plan_for_io_bound(
        self, 
        bottleneck: Any, 
        config: Dict
    ) -> List[OptimizationPlan]:
        """针对IO瓶颈的优化方案"""
        plans = []
        
        current_workers = config.get('dataset', {}).get('num_workers', 4)
        
        # 方案1: 增加num_workers
        plans.append(OptimizationPlan(
            strategy=OptimizationStrategy.PARALLELIZATION,
            target_stage='data_loading',
            priority=9,
            description=f"增加数据加载并行度: {current_workers} -> {current_workers * 2}",
            config_changes={
                'dataset.num_workers': current_workers * 2,
                'dataset.pin_memory': True
            },
            expected_improvement={
                'throughput': 0.30,
                'gpu_utilization': 0.20
            },
            risk_level='low',
            rollback_plan=f'恢复num_workers为{current_workers}'
        ))
        
        # 方案2: 数据预加载和缓存
        plans.append(OptimizationPlan(
            strategy=OptimizationStrategy.CACHE_OPTIMIZATION,
            target_stage='data_loading',
            priority=7,
            description="启用数据预加载和缓存",
            config_changes={
                'dataset.prefetch_factor': 4,
                'dataset.persistent_workers': True
            },
            expected_improvement={
                'throughput': 0.20,
                'latency': 0.15
            },
            risk_level='low',
            rollback_plan='禁用预加载'
        ))
        
        return plans
    
    def _plan_for_stage_imbalance(
        self, 
        bottleneck: Any, 
        config: Dict
    ) -> List[OptimizationPlan]:
        """针对Stage不平衡的优化方案"""
        plans = []
        
        stage = bottleneck.stage
        
        if stage == 'preprocess':
            plans.append(OptimizationPlan(
                strategy=OptimizationStrategy.PARALLELIZATION,
                target_stage=stage,
                priority=8,
                description="预处理并行化，使用多线程/GPU加速",
                config_changes={
                    'dataset.preprocess_on_gpu': True,
                    'dataset.num_preprocess_workers': 4
                },
                expected_improvement={
                    'latency': 0.40
                },
                risk_level='low',
                rollback_plan='使用CPU单线程预处理'
            ))
        
        elif stage == 'inference':
            # 推理阶段慢，使用TensorRT或批处理优化
            plans.append(OptimizationPlan(
                strategy=OptimizationStrategy.TENSORRT_OPTIMIZATION,
                target_stage=stage,
                priority=9,
                description="使用TensorRT优化推理阶段",
                config_changes={
                    'deployment.tensorrt.enabled': True
                },
                expected_improvement={
                    'latency': 0.35
                },
                risk_level='low',
                rollback_plan='禁用TensorRT'
            ))
        
        elif stage == 'postprocess':
            plans.append(OptimizationPlan(
                strategy=OptimizationStrategy.OPERATOR_FUSION,
                target_stage=stage,
                priority=7,
                description="后处理算子融合，减少kernel launch开销",
                config_changes={
                    'model.fuse_postprocess': True
                },
                expected_improvement={
                    'latency': 0.25
                },
                risk_level='low',
                rollback_plan='分离后处理算子'
            ))
        
        return plans
    
    def _plan_for_latency_spike(
        self, 
        bottleneck: Any, 
        config: Dict
    ) -> List[OptimizationPlan]:
        """针对延迟突增的优化方案"""
        plans = []
        
        # 方案: 内存池优化
        plans.append(OptimizationPlan(
            strategy=OptimizationStrategy.MEMORY_OPTIMIZATION,
            target_stage=None,
            priority=8,
            description="启用CUDA内存池，减少内存分配延迟",
            config_changes={
                'deployment.cuda_memory_pool': True
            },
            expected_improvement={
                'latency': 0.20,
                'latency_variance': 0.40
            },
            risk_level='low',
            rollback_plan='禁用内存池'
        ))
        
        return plans
    
    def evaluate_plan(self, plan: OptimizationPlan, 
                      ab_test_results: Optional[Dict] = None) -> Dict[str, Any]:
        """
        评估优化方案效果
        
        Args:
            plan: 优化方案
            ab_test_results: A/B测试结果
            
        Returns:
            评估报告
        """
        if ab_test_results is None:
            return {
                'plan': plan.strategy.value,
                'status': 'pending',
                'message': '等待A/B测试结果'
            }
        
        # 检查精度损失
        precision_drop = ab_test_results.get('precision_drop', 0)
        max_allowed = self.strategy_configs.get(
            'precision_calibration', {}
        ).get('max_precision_drop', 0.02)
        
        if precision_drop > max_allowed:
            return {
                'plan': plan.strategy.value,
                'status': 'rejected',
                'reason': f'精度损失({precision_drop:.3f})超过阈值({max_allowed:.3f})',
                'recommendation': '尝试更保守的优化参数或放弃此方案'
            }
        
        # 检查性能提升
        latency_improvement = ab_test_results.get('latency_improvement', 0)
        throughput_improvement = ab_test_results.get('throughput_improvement', 0)
        
        if latency_improvement < 0.05 and throughput_improvement < 0.05:
            return {
                'plan': plan.strategy.value,
                'status': 'ineffective',
                'reason': '性能提升不明显',
                'recommendation': '尝试其他优化方案'
            }
        
        return {
            'plan': plan.strategy.value,
            'status': 'approved',
            'improvements': {
                'latency': f'{latency_improvement*100:.1f}%',
                'throughput': f'{throughput_improvement*100:.1f}%',
                'precision_drop': f'{precision_drop*100:.2f}%'
            }
        }


if __name__ == "__main__":
    # 测试
    agent = OptimizerAgent()
    
    # 模拟瓶颈
    class MockBottleneck:
        def __init__(self, type_value, stage=None):
            self.type = type(type('obj', (object,), {'value': type_value})())
            self.stage = stage
    
    bottlenecks = [
        MockBottleneck('compute_bound', 'inference'),
        MockBottleneck('memory_bound')
    ]
    
    config = {
        'dataset': {'batch_size': 16, 'num_workers': 4},
        'deployment': {}
    }
    
    plans = agent.generate_plans(bottlenecks, config)
    
    print("Generated Optimization Plans:")
    for plan in plans:
        print(f"\n  Strategy: {plan.strategy.value}")
        print(f"  Priority: {plan.priority}")
        print(f"  Description: {plan.description}")
        print(f"  Expected Improvement: {plan.expected_improvement}")

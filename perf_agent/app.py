"""
LangGraph 多 Agent 自动化性能分析系统
面试演示 Demo

架构：
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Collector  │ -> │   Analyzer  │ -> │ Optimizer   │ -> │  Executor   │
│  (采集Agent) │    │  (分析Agent) │    │ (优化Agent)  │    │  (执行Agent) │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                  │                  │                  │
       v                  v                  v                  v
  采集指标            识别瓶颈           生成方案           执行优化
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
from dataclasses import dataclass
import json
import time


# ==================== 数据模型 ====================

@dataclass
class StageMetric:
    """Stage 级别性能指标"""
    stage_name: str
    latency_ms: float
    memory_mb: float
    gpu_util: float  # %
    throughput: float  # items/sec
    error_rate: float  # %


@dataclass
class Bottleneck:
    """性能瓶颈"""
    stage: str
    type: str  # "memory", "latency", "gpu", "throughput"
    severity: str  # "critical", "high", "medium"
    description: str
    suggestion: str


@dataclass
class Optimization:
    """优化方案"""
    name: str
    target_stage: str
    changes: List[str]
    expected_improvement: str
    risk: str  # "low", "medium", "high"


# ==================== Agent 实现 ====================

class CollectorAgent:
    """采集 Agent - 收集性能指标"""
    
    def __init__(self):
        self.metrics: List[StageMetric] = []
    
    def collect(self) -> List[Dict]:
        """模拟采集性能指标"""
        print("\n📊 [Collector] 正在采集性能指标...")
        
        # 模拟 Pipeline 各 Stage 的指标
        stages = [
            StageMetric("data_ingestion", 45.2, 512, 35, 1000, 0.1),
            StageMetric("preprocessing", 120.5, 1024, 65, 950, 0.05),
            StageMetric("model_inference", 280.0, 2048, 92, 880, 0.02),
            StageMetric("postprocessing", 35.8, 384, 25, 870, 0.01),
            StageMetric("output_writer", 18.3, 256, 15, 865, 0.0),
        ]
        
        self.metrics = stages
        
        # 输出采集结果
        print("\n  Stage Name        | Latency(ms) | Memory(MB) | GPU(%) | Throughput")
        print("  " + "-" * 70)
        for m in stages:
            print(f"  {m.stage_name:<18} | {m.latency_ms:>11.1f} | {m.memory_mb:>10.0f} | {m.gpu_util:>6.0f} | {m.throughput:>10.0f}")
        
        return [
            {
                "stage": m.stage_name,
                "latency_ms": m.latency_ms,
                "memory_mb": m.memory_mb,
                "gpu_util": m.gpu_util,
                "throughput": m.throughput,
                "error_rate": m.error_rate
            }
            for m in stages
        ]


class AnalyzerAgent:
    """分析 Agent - 识别性能瓶颈"""
    
    def analyze(self, metrics: List[Dict]) -> List[Dict]:
        """分析性能数据，识别瓶颈"""
        print("\n🔍 [Analyzer] 正在分析性能瓶颈...")
        
        bottlenecks = []
        
        for m in metrics:
            # GPU 利用率分析
            if m["gpu_util"] > 85:
                bottlenecks.append({
                    "stage": m["stage"],
                    "type": "gpu",
                    "severity": "critical" if m["gpu_util"] > 90 else "high",
                    "description": f"GPU 利用率过高 ({m['gpu_util']}%)",
                    "suggestion": "考虑使用 TensorRT 加速或增大 batch size"
                })
            
            # 内存分析
            if m["memory_mb"] > 1800:
                bottlenecks.append({
                    "stage": m["stage"],
                    "type": "memory",
                    "severity": "high",
                    "description": f"内存占用过高 ({m['memory_mb']}MB)",
                    "suggestion": "启用梯度累积或减少 batch size"
                })
            
            # 延迟分析
            if m["latency_ms"] > 200:
                bottlenecks.append({
                    "stage": m["stage"],
                    "type": "latency",
                    "severity": "critical" if m["latency_ms"] > 250 else "high",
                    "description": f"推理延迟过高 ({m['latency_ms']}ms)",
                    "suggestion": "考虑模型剪枝、量化或使用 TensorRT"
                })
        
        # 输出瓶颈分析
        print(f"\n  发现 {len(bottlenecks)} 个性能瓶颈:")
        for i, b in enumerate(bottlenecks, 1):
            severity_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡"}.get(b["severity"], "⚪")
            print(f"  {i}. {severity_emoji} [{b['severity'].upper()}] {b['stage']} - {b['type']}")
            print(f"     问题: {b['description']}")
            print(f"     建议: {b['suggestion']}")
        
        return bottlenecks


class OptimizerAgent:
    """优化 Agent - 生成优化方案"""
    
    def __init__(self):
        self.mode = "balanced"  # high_precision / balanced / high_speed
    
    def set_mode(self, mode: str):
        """设置优化目标模式"""
        self.mode = mode
        print(f"\n⚙️  [Optimizer] 目标模式: {mode}")
    
    def optimize(self, bottlenecks: List[Dict], mode: str = "balanced") -> List[Dict]:
        """生成优化方案"""
        print(f"\n🛠️  [Optimizer] 正在生成优化方案 (模式: {mode})...")
        
        optimizations = []
        
        for b in bottlenecks:
            opt = self._generate_optimization(b, mode)
            optimizations.append(opt)
        
        # 输出优化方案
        print(f"\n  生成 {len(optimizations)} 个优化方案:")
        for i, opt in enumerate(optimizations, 1):
            print(f"  {i}. 📋 {opt['name']}")
            print(f"     目标 Stage: {opt['target_stage']}")
            print(f"     预期提升: {opt['expected_improvement']}")
            print(f"     风险等级: {opt['risk']}")
            for change in opt['changes']:
                print(f"       - {change}")
        
        return optimizations
    
    def _generate_optimization(self, bottleneck: Dict, mode: str) -> Dict:
        """根据瓶颈类型生成优化方案"""
        
        if bottleneck["type"] == "gpu":
            if mode == "high_speed":
                return {
                    "name": "TensorRT FP16 加速",
                    "target_stage": bottleneck["stage"],
                    "changes": ["启用 FP16 推理", "转换为 TensorRT Engine"],
                    "expected_improvement": "延迟降低 50%+",
                    "risk": "medium"
                }
            else:
                return {
                    "name": "Batch Size 优化",
                    "target_stage": bottleneck["stage"],
                    "changes": ["增大 batch size 到 32", "启用动态批处理"],
                    "expected_improvement": "吞吐量提升 30%",
                    "risk": "low"
                }
        
        elif bottleneck["type"] == "memory":
            return {
                "name": "内存优化方案",
                "target_stage": bottleneck["stage"],
                "changes": ["启用混合精度", "使用梯度检查点", "优化数据加载"],
                "expected_improvement": "内存降低 40%",
                "risk": "low"
            }
        
        elif bottleneck["type"] == "latency":
            if mode == "high_precision":
                return {
                    "name": "模型蒸馏",
                    "target_stage": bottleneck["stage"],
                    "changes": ["使用知识蒸馏小模型", "保留关键特征层"],
                    "expected_improvement": "延迟降低 30%，精度损失 <1%",
                    "risk": "medium"
                }
            else:
                return {
                    "name": "模型量化",
                    "target_stage": bottleneck["stage"],
                    "changes": ["INT8 量化", "校准数据集微调"],
                    "expected_improvement": "延迟降低 60%",
                    "risk": "low"
                }
        
        return {
            "name": "通用优化",
            "target_stage": bottleneck["stage"],
            "changes": ["待分析"],
            "expected_improvement": "待评估",
            "risk": "unknown"
        }


class ExecutorAgent:
    """执行 Agent - 执行优化并验证"""
    
    def execute(self, optimizations: List[Dict]) -> Dict:
        """执行优化方案"""
        print("\n🚀 [Executor] 正在执行优化方案...")
        
        results = []
        
        for i, opt in enumerate(optimizations, 1):
            print(f"\n  执行 {i}/{len(optimizations)}: {opt['name']}...")
            
            # 模拟执行
            time.sleep(0.5)
            
            # 模拟 A/B 测试结果
            ab_result = {
                "optimization": opt["name"],
                "latency_before": 280.0,
                "latency_after": 180.0,
                "throughput_before": 880,
                "throughput_after": 1200,
                "accuracy_before": 0.985,
                "accuracy_after": 0.983,
                "passed": True
            }
            
            results.append(ab_result)
            
            print(f"     ✅ 延迟: {ab_result['latency_before']}ms -> {ab_result['latency_after']}ms (↓{((ab_result['latency_before']-ab_result['latency_after'])/ab_result['latency_before']*100):.1f}%)")
            print(f"     ✅ 吞吐量: {ab_result['throughput_before']} -> {ab_result['throughput_after']} (↑{((ab_result['throughput_after']-ab_result['throughput_before'])/ab_result['throughput_before']*100):.1f}%)")
            print(f"     ✅ 精度: {ab_result['accuracy_before']*100:.1f}% -> {ab_result['accuracy_after']*100:.1f}% (Δ{(ab_result['accuracy_after']-ab_result['accuracy_before'])*100:+.2f}%)")
        
        return {
            "total": len(optimizations),
            "passed": len([r for r in results if r["passed"]]),
            "results": results
        }


# ==================== LangGraph 工作流 ====================

class PipelineState(TypedDict):
    """Pipeline 状态"""
    metrics: List[Dict]
    bottlenecks: List[Dict]
    optimizations: List[Dict]
    execution_result: Dict
    mode: str


def create_pipeline():
    """创建 LangGraph Pipeline"""
    
    # 初始化 Agent
    collector = CollectorAgent()
    analyzer = AnalyzerAgent()
    optimizer = OptimizerAgent()
    executor = ExecutorAgent()
    
    # 创建图
    graph = StateGraph(PipelineState)
    
    # 添加节点
    graph.add_node("collect", lambda state: {"metrics": collector.collect()})
    graph.add_node("analyze", lambda state: {"bottlenecks": analyzer.analyze(state["metrics"])})
    graph.add_node("optimize", lambda state: {"optimizations": optimizer.optimize(state["bottlenecks"], state.get("mode", "balanced"))})
    graph.add_node("execute", lambda state: {"execution_result": executor.execute(state["optimizations"])})
    
    # 添加边
    graph.add_edge("__start__", "collect")
    graph.add_edge("collect", "analyze")
    graph.add_edge("analyze", "optimize")
    graph.add_edge("optimize", "execute")
    graph.add_edge("execute", END)
    
    return graph.compile()


# ==================== 主程序 ====================

def run_demo(mode: str = "balanced"):
    """运行演示"""
    
    print("=" * 70)
    print("  🎯 LangGraph 多 Agent 自动化性能分析系统")
    print("  📌 面试 Demo - 基于 LangGraph 的 Pipeline 优化")
    print("=" * 70)
    print(f"\n🎬 启动 Pipeline (模式: {mode})")
    
    # 创建 Pipeline
    pipeline = create_pipeline()
    
    # 运行
    result = pipeline.invoke({
        "mode": mode,
        "metrics": [],
        "bottlenecks": [],
        "optimizations": [],
        "execution_result": {}
    })
    
    # 输出总结
    print("\n" + "=" * 70)
    print("  📋 Pipeline 执行完成 - 总结报告")
    print("=" * 70)
    
    exec_result = result["execution_result"]
    print(f"\n  ✅ 优化方案总数: {exec_result['total']}")
    print(f"  ✅ 执行成功: {exec_result['passed']}")
    print(f"  ❌ 执行失败: {exec_result['total'] - exec_result['passed']}")
    
    avg_latency_improvement = sum(
        (r["latency_before"] - r["latency_after"]) / r["latency_before"] * 100
        for r in exec_result["results"]
    ) / len(exec_result["results"])
    
    avg_throughput_improvement = sum(
        (r["throughput_after"] - r["throughput_before"]) / r["throughput_before"] * 100
        for r in exec_result["results"]
    ) / len(exec_result["results"])
    
    print(f"\n  📈 平均延迟改善: {avg_latency_improvement:.1f}%")
    print(f"  📈 平均吞吐量提升: {avg_throughput_improvement:.1f}%")
    
    print("\n" + "=" * 70)
    print("  演示结束 - 感谢聆听！")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    
    # 支持命令行参数
    mode = sys.argv[1] if len(sys.argv) > 1 else "balanced"
    
    # 可选: high_precision / balanced / high_speed
    valid_modes = ["high_precision", "balanced", "high_speed"]
    if mode not in valid_modes:
        print(f"Usage: python app.py [{'|'.join(valid_modes)}]")
        sys.exit(1)
    
    run_demo(mode)
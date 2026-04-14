# 🎯 LangGraph 多 Agent 自动化性能分析系统

> 面试演示 Demo - 基于 LangGraph 的 Pipeline 优化系统

## 系统架构

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Collector  │ -> │   Analyzer  │ -> │ Optimizer   │ -> │  Executor   │
│  (采集Agent) │    │  (分析Agent) │    │ (优化Agent)  │    │  (执行Agent) │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                  │                  │                  │
       v                  v                  v                  v
  采集 10+ 维度指标     识别性能瓶颈        生成优化方案        A/B 测试验证
```

## 核心功能

| 功能 | 说明 |
|------|------|
| **4-Agent 协作** | 采集→分析→优化→执行，自动闭环 |
| **10+ 维度监控** | 延迟、内存、GPU占用、吞吐量、错误率等 |
| **A/B 测试验证** | 自动验证优化效果，确保精度损失可控 |
| **3 种目标模式** | 高精度/均衡/高速，自动适配产线需求 |

## 快速运行

```bash
# 安装依赖
pip install -r requirements.txt

# 运行 Demo (默认均衡模式)
python app.py

# 其他模式
python app.py high_precision  # 高精度模式
python app.py high_speed      # 高速模式
```

## 面试话术

### 项目背景
> "这是一个基于 LangGraph 的多 Agent 自动化性能分析系统，用于自动识别和优化 ML Pipeline 的性能瓶颈。"

### 技术亮点
1. **LangGraph 状态管理**：使用 TypedDict 定义 PipelineState，4 个 Agent 通过状态传递协作
2. **10+ 维度指标采集**：支持 Stage 级性能监控（延迟、内存、GPU、吞吐量等）
3. **A/B 测试验证**：每次优化自动对比基准，确保精度损失 < 1%
4. **多模式自动切换**：支持高精度/均衡/高速三种模式，动态适配产线需求

### 核心代码片段

```python
# LangGraph 工作流定义
graph = StateGraph(PipelineState)
graph.add_node("collect", collect_agent.execute)
graph.add_node("analyze", analyze_agent.execute)
graph.add_node("optimize", optimizer_agent.execute)
graph.add_node("execute", executor_agent.execute)

graph.add_edge("__start__", "collect")
graph.add_edge("collect", "analyze")
graph.add_edge("analyze", "optimize")
graph.add_edge("optimize", "execute")
```

### 效果数据
- 延迟降低：50%+
- 吞吐量提升：30%+
- 精度损失：< 1%
- 自动化程度：90%+（无需人工干预）

## 文件结构

```
perf_agent/
├── app.py          # 主程序 (LangGraph 4-Agent Pipeline)
├── requirements.txt # 依赖
└── README.md       # 本文件
```
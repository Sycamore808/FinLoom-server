# FinLoom 量化投资引擎

FinLoom ("金融（Fin）+ 编织（Loom）") 是一个FIN-R1赋能的自适应量化投资引擎，旨在将数据、因子、模型、用户需求等元素编织成个性化的投资组合。

## 项目概述

FinLoom系统以 **FIN-R1 模型** 作为自然语言理解入口，支持用户通过文本输入表达投资需求（如"找高成长性中小盘股"），系统自动解析并生成对应的量化策略。

整体流程为：
```
用户输入 → NLP 解析 → 策略生成 → 数据处理 → 回测/实盘 → 风控校验 → 交易执行
```

## 核心功能模块

1. **环境模块 (module_00_environment)** - 系统环境检测和配置管理
2. **数据管道模块 (module_01_data_pipeline)** - 数据采集、处理和存储
3. **特征工程模块 (module_02_feature_engineering)** - 特征提取和因子发现
4. **AI模型模块 (module_03_ai_models)** - 机器学习模型集成
5. **市场分析模块 (module_04_market_analysis)** - 市场趋势和情绪分析
6. **风险管理模块 (module_05_risk_management)** - 风险控制和仓位管理
7. **监控告警模块 (module_06_monitoring_alerting)** - 系统监控和告警
8. **优化模块 (module_07_optimization)** - 参数优化和组合优化
9. **执行模块 (module_08_execution)** - 交易执行和订单管理
10. **回测模块 (module_09_backtesting)** - 策略回测和性能分析
11. **AI交互模块 (module_10_ai_interaction)** - 自然语言处理和需求解析
12. **可视化模块 (module_11_visualization)** - 图表展示和报告生成

## 技术栈

- **Python**: 3.9+
- **PyTorch**: 2.0+ (仅用于 FIN-R1 模型推理)
- **DuckDB**: 本地数据分析
- **Akshare**: 中国金融市场数据获取
- **TA-Lib**: 技术分析指标计算
- **Plotly**: 交互式图表展示
- **FastAPI**: RESTful API服务

## 安装和使用

### 环境准备

1. 确保已安装Python 3.9+
2. 安装依赖包:
   ```bash
   pip install -r requirements.txt
   ```

### 运行系统

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python main.py
```

### API访问

系统启动后可通过以下端点访问:

- `http://localhost:8000/` - API根路径
- `http://localhost:8000/health` - 健康检查
- `http://localhost:8000/api/v1/analyze` - 投资需求分析

### 使用示例

#### 1. API使用示例

```bash
# 运行API使用示例
python examples/api_usage_example.py
```

#### 2. 数据收集示例

```bash
# 运行数据收集示例
python examples/data_collection_example.py
```

#### 3. 回测示例

```bash
# 运行回测示例
python examples/backtest_example.py
```

#### 4. 投资需求分析API调用

```bash
# 使用curl测试API
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "我想找一些高成长性的中小盘股票，风险承受能力中等，投资期限1-3年"}'
```

## 配置文件

- `config/system_config.yaml` - 系统配置
- `config/model_config.yaml` - 模型配置
- `config/trading_config.yaml` - 交易配置

## 示例

查看 `examples/sample_strategy.py` 了解如何使用系统构建量化策略。

## 开发指南

请参考 `docs/developer_guide/项目规范.md` 了解详细的开发规范和模块间通信协议。
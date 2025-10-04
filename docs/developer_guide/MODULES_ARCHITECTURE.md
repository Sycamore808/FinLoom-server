# FinLoom 模块架构总览

## 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户交互层                                │
│  Module 10 (AI Interaction) + Module 11 (Visualization)         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────────┐
│                        应用服务层                                 │
│  Module 04 (Market Analysis) + Module 06 (Monitoring)           │
│  Module 08 (Execution) + Module 09 (Backtesting)                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────────┐
│                        策略决策层                                 │
│  Module 03 (AI Models) + Module 05 (Risk Management)            │
│  Module 07 (Optimization)                                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────────┐
│                        数据处理层                                 │
│  Module 02 (Feature Engineering) + Module 01 (Data Pipeline)    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────────┐
│                        基础设施层                                 │
│  Module 00 (Environment) + Common (工具库)                       │
└─────────────────────────────────────────────────────────────────┘
```

## 模块功能总览

### 已完善模块 (Module 00-04)

#### Module 00 - 环境配置模块
**功能**: 系统环境检测、配置管理、依赖安装
**对外接口**: 编程接口
**数据库**: 无专用数据库
**主要类**:
- `ConfigLoader`: 配置加载器
- `DependencyInstaller`: 依赖安装器
- `EnvironmentChecker`: 环境检查器
- `HealthMonitor`: 健康监控器

#### Module 01 - 数据管道模块
**功能**: 数据采集、清洗、存储
**对外接口**: 编程接口
**数据库**: `data/finloom.db`
**主要类**:
- `AkshareDataCollector`: 中国A股数据采集
- `ChineseAlternativeDataCollector`: 宏观数据采集
- `ChineseFundamentalCollector`: 财务数据采集
- `DatabaseManager`: 数据库管理器
- `DataCleaner`: 数据清洗
- `DataValidator`: 数据验证

#### Module 02 - 特征工程模块
**功能**: 技术指标计算、因子发现、时序特征、图特征
**对外接口**: 编程接口
**数据库**: `data/module02_features.db`
**主要类**:
- `TechnicalIndicators`: 技术指标计算
- `FactorAnalyzer`: 因子分析
- `NeuralFactorDiscovery`: 神经因子发现
- `TimeSeriesFeatures`: 时序特征提取
- `GraphAnalyzer`: 图特征分析
- `FeatureDatabaseManager`: 特征数据库管理

#### Module 03 - AI模型模块
**功能**: 深度学习、集成学习、在线学习、强化学习
**对外接口**: 编程接口
**数据库**: `data/module03_ai_models.db`
**主要类**:
- `LSTMModel`: LSTM时序预测
- `TransformerPredictor`: Transformer预测
- `EnsemblePredictor`: 集成预测
- `OnlineLearner`: 在线学习
- `PPOAgent`: PPO强化学习
- `AIModelDatabaseManager`: AI模型数据库管理

#### Module 04 - 市场分析模块
**功能**: 多智能体分析、情感分析、异常检测
**对外接口**: 编程接口 + REST API
**数据库**: `data/module04_market_analysis.db`
**主要类**:
- `FINR1SentimentAnalyzer`: FIN-R1情感分析
- `EnhancedNewsSentimentAnalyzer`: 增强情感分析
- `PriceAnomalyDetector`: 价格异常检测
- `MarketRegimeDetector`: 市场状态检测
- `CorrelationAnalyzer`: 相关性分析

### 待完善模块 (Module 05-11)

#### Module 05 - 风险管理模块 ✅
**功能**: 风险评估、仓位管理、止损止盈、压力测试
**对外接口**: 编程接口
**数据库**: `data/module05_risk_management.db`
**主要类**:
- `PortfolioRiskAnalyzer`: 投资组合风险分析
- `VaRCalculator`: VaR/CVaR计算
- `KellyCriterion`: 凯利公式仓位管理
- `StopLossManager`: 止损管理
- `ExposureMonitor`: 风险敞口监控
- `StressTestEngine`: 压力测试引擎

**模块间调用**:
- 调用Module 01获取历史价格数据
- 调用Module 02获取技术指标(ATR等)
- 调用Module 03获取AI风险预测
- 调用Module 04获取市场情感数据
- 为Module 08提供风险检查服务

#### Module 06 - 监控告警模块 ✅
**功能**: 系统监控、性能追踪、市场预警、通知服务
**对外接口**: 编程接口 + REST API + WebSocket
**数据库**: `data/module06_monitoring_alerting.db`
**主要类**:
- `SystemHealthMonitor`: 系统健康监控
- `PerformanceMonitor`: 性能监控
- `MarketWatchdog`: 市场监控
- `RiskAlertEngine`: 风险预警
- `NotificationManager`: 通知管理
- `ReportScheduler`: 报告调度

**模块间调用**:
- 监控所有模块的性能和状态
- 调用Module 01检查数据采集状态
- 调用Module 05监控风险指标
- 调用Module 08监控订单执行
- 为所有模块提供监控和告警服务

#### Module 07 - 优化模块 ✅
**功能**: 投资组合优化、策略优化、超参数调优
**对外接口**: 编程接口
**数据库**: `data/module07_optimization.db`
**主要类**:
- `MarkowitzOptimizer`: 马科维茨优化
- `BlackLittermanOptimizer`: Black-Litterman优化
- `RiskParityOptimizer`: 风险平价优化
- `StrategyOptimizer`: 策略参数优化
- `BayesianOptimizer`: 贝叶斯超参数优化
- `NSGAOptimizer`: NSGA-II多目标优化

**模块间调用**:
- 调用Module 01获取历史数据
- 调用Module 02优化因子参数
- 调用Module 03优化模型超参数
- 调用Module 05考虑风险约束
- 为Module 08提供最优配置

#### Module 08 - 执行模块
**功能**: 订单管理、交易执行、算法交易、滑点控制
**对外接口**: 编程接口 + REST API
**数据库**: `data/module08_execution.db`
**主要类**:
- `OrderManager`: 订单管理器
- `OrderRouter`: 订单路由
- `ExecutionAlgorithm`: 执行算法(TWAP, VWAP等)
- `SignalGenerator`: 信号生成
- `SignalFilter`: 信号过滤
- `BrokerConnector`: 券商接口连接器
- `ExecutionMonitor`: 执行监控
- `TransactionLogger`: 交易日志

**模块间调用**:
- 调用Module 03获取AI交易信号
- 调用Module 04获取市场分析
- 调用Module 05进行风险检查
- 调用Module 06记录执行事件
- 调用Module 07获取最优执行参数
- 为Module 09提供真实执行数据

**主要流程**:
```
信号生成 → 信号过滤 → 风险检查 → 订单创建 → 订单路由 → 执行算法 → 交易执行 → 执行监控
```

#### Module 09 - 回测模块
**功能**: 历史回测、性能分析、风险归因
**对外接口**: 编程接口 + REST API
**数据库**: `data/module09_backtesting.db`
**主要类**:
- `BacktestEngine`: 回测引擎
- `PerformanceAnalyzer`: 绩效分析器
- `RiskAttribution`: 风险归因
- `TransactionSimulator`: 交易模拟器
- `MarketImpactModel`: 市场冲击模型
- `WalkForwardAnalyzer`: 滚动窗口分析
- `ReportGenerator`: 回测报告生成器

**模块间调用**:
- 调用Module 01获取历史数据
- 调用Module 02获取历史特征
- 调用Module 03获取模型预测
- 调用Module 04获取历史情感
- 调用Module 05评估历史风险
- 调用Module 07优化策略参数
- 调用Module 08模拟订单执行
- 调用Module 11生成回测图表

**回测流程**:
```
加载数据 → 计算特征 → 生成信号 → 风险检查 → 模拟执行 → 计算收益 → 性能分析 → 生成报告
```

#### Module 10 - AI交互模块
**功能**: 自然语言理解、需求解析、策略推荐
**对外接口**: 编程接口 + REST API + WebSocket
**数据库**: `data/module10_ai_interaction.db`
**主要类**:
- `NLPProcessor`: 自然语言处理器
- `RequirementParser`: 需求解析器
- `StrategyGenerator`: 策略生成器
- `StrategyRecommender`: 策略推荐器
- `ConversationManager`: 对话管理器
- `KnowledgeBase`: 知识库
- `FinR1Integration`: FIN-R1模型集成

**模块间调用**:
- 调用Module 01获取数据状态
- 调用Module 02获取可用因子
- 调用Module 03获取可用模型
- 调用Module 04获取市场洞察
- 调用Module 05评估风险
- 调用Module 07优化策略
- 调用Module 09回测策略
- 调用Module 11可视化结果

**交互流程**:
```
用户输入 → NLP解析 → 需求理解 → 策略生成 → 参数优化 → 回测验证 → 风险评估 → 结果展示 → 策略推荐
```

**示例对话**:
```
用户: "我想要一个适合中小盘成长股的量化策略"
系统: 
1. 解析需求: 市值偏好(中小盘)、风格偏好(成长)
2. 生成策略: 基于PE-G因子 + 动量因子的多因子策略
3. 回测验证: 夏普比率1.82，年化收益25%，最大回撤15%
4. 风险评估: 风险等级中等，建议仓位不超过30%
5. 推荐方案: [展示详细策略配置和回测报告]
```

#### Module 11 - 可视化模块
**功能**: 图表生成、仪表板、实时可视化
**对外接口**: 编程接口 + REST API
**数据库**: `data/module11_visualization.db`
**主要类**:
- `ChartGenerator`: 图表生成器
- `DashboardManager`: 仪表板管理器
- `InteractiveVisualizer`: 交互式可视化
- `ReportBuilder`: 报告构建器
- `TemplateEngine`: 模板引擎
- `ExportManager`: 导出管理器

**模块间调用**:
- 从Module 01可视化市场数据
- 从Module 02可视化技术指标
- 从Module 03可视化模型预测
- 从Module 04可视化市场分析
- 从Module 05可视化风险指标
- 从Module 06可视化系统监控
- 从Module 07可视化优化结果
- 从Module 08可视化交易执行
- 从Module 09可视化回测结果
- 从Module 10可视化AI对话

**图表类型**:
- K线图、分时图
- 技术指标图
- 因子分布图
- 收益曲线图
- 风险分布图
- 相关性热力图
- 帕累托前沿图
- 订单执行分析图
- 回测绩效图
- 仪表盘组件

## 数据流向图

```
┌─────────────────────────────────────────────────────────────┐
│                      数据采集层                              │
│  Module 01: 原始数据 → 清洗数据 → 数据库存储                 │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ↓
┌─────────────────────────────────────────────────────────────┐
│                      特征提取层                              │
│  Module 02: 清洗数据 → 技术指标 → 因子 → 特征数据库           │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ↓
┌─────────────────────────────────────────────────────────────┐
│                   预测与分析层                               │
│  Module 03: 特征 → AI模型 → 预测结果                         │
│  Module 04: 数据 → 市场分析 → 分析报告                       │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ↓
┌─────────────────────────────────────────────────────────────┐
│                   决策优化层                                 │
│  Module 05: 预测 → 风险评估 → 风险报告                       │
│  Module 07: 历史数据 → 优化 → 最优参数                       │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ↓
┌─────────────────────────────────────────────────────────────┐
│                   执行与验证层                               │
│  Module 08: 信号 → 风险检查 → 订单执行 → 成交记录             │
│  Module 09: 策略 → 回测引擎 → 绩效报告                       │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ↓
┌─────────────────────────────────────────────────────────────┐
│                   展示交互层                                 │
│  Module 10: 用户需求 → AI理解 → 策略推荐                     │
│  Module 11: 各模块数据 → 可视化 → 图表展示                   │
│  Module 06: 系统状态 → 监控告警 → 通知推送                   │
└─────────────────────────────────────────────────────────────┘
```

## 典型业务流程

### 流程1: 策略开发流程

```
1. [用户输入] → Module 10 解析需求
   ↓
2. [数据准备] → Module 01 采集数据 → Module 02 计算特征
   ↓
3. [策略构建] → Module 03 训练模型 → Module 07 优化参数
   ↓
4. [风险评估] → Module 05 风险分析
   ↓
5. [回测验证] → Module 09 历史回测 → Module 11 结果展示
   ↓
6. [策略部署] → Module 08 实盘执行 → Module 06 实时监控
```

### 流程2: 日常交易流程

```
1. [市场监控] → Module 06 监控市场 + Module 04 分析市场
   ↓
2. [信号生成] → Module 03 AI预测 → Module 08 生成信号
   ↓
3. [风险检查] → Module 05 风险评估 → 通过/拒绝
   ↓
4. [订单执行] → Module 08 创建订单 → 执行算法 → 成交
   ↓
5. [实时监控] → Module 06 监控执行 + Module 11 展示仪表板
   ↓
6. [风险预警] → Module 06 检测异常 → 发送告警
```

### 流程3: 投资组合再平衡流程

```
1. [定时触发] → Module 06 调度任务
   ↓
2. [风险评估] → Module 05 分析当前组合风险
   ↓
3. [市场分析] → Module 04 市场情感 + 异常检测
   ↓
4. [组合优化] → Module 07 计算最优权重
   ↓
5. [调仓决策] → 对比当前与目标权重
   ↓
6. [订单生成] → Module 08 生成调仓订单
   ↓
7. [执行监控] → Module 06 + Module 11 监控展示
```

### 流程4: 模型优化流程

```
1. [性能监控] → Module 06 监控模型性能下降
   ↓
2. [数据准备] → Module 01 获取最新数据 → Module 02 计算特征
   ↓
3. [参数优化] → Module 07 超参数调优
   ↓
4. [模型训练] → Module 03 重新训练模型
   ↓
5. [回测验证] → Module 09 验证新模型性能
   ↓
6. [上线部署] → Module 03 部署新模型 → Module 06 监控
```

## API接口总览

### REST API模块

| 模块 | API端点 | 说明 |
|------|---------|------|
| Module 04 | `/api/v1/market/*` | 市场分析API |
| Module 06 | `/api/v1/monitoring/*` | 监控告警API |
| Module 08 | `/api/v1/execution/*` | 交易执行API |
| Module 09 | `/api/v1/backtest/*` | 回测分析API |
| Module 10 | `/api/v1/ai/*` | AI交互API |
| Module 11 | `/api/v1/visualization/*` | 可视化API |

### WebSocket推送

| 模块 | WebSocket端点 | 推送内容 |
|------|---------------|----------|
| Module 06 | `/ws/monitoring` | 系统状态、告警信息 |
| Module 08 | `/ws/execution` | 订单状态、成交信息 |
| Module 10 | `/ws/chat` | AI对话交互 |
| Module 11 | `/ws/realtime` | 实时图表数据 |

### 编程接口

所有模块都提供Python编程接口，供其他模块调用：

```python
# 模块间调用示例
from module_01_data_pipeline import AkshareDataCollector
from module_02_feature_engineering import TechnicalIndicators
from module_03_ai_models import LSTMModel
from module_04_market_analysis import get_sentiment_analyzer
from module_05_risk_management import PortfolioRiskAnalyzer
from module_06_monitoring_alerting import NotificationManager
from module_07_optimization import MarkowitzOptimizer
from module_08_execution import OrderManager
from module_09_backtesting import BacktestEngine
from module_10_ai_interaction import NLPProcessor
from module_11_visualization import ChartGenerator
```

## 数据库架构

### 各模块专用数据库

| 模块 | 数据库文件 | 主要表 |
|------|-----------|--------|
| Module 01 | `finloom.db` | stock_prices, stock_info, macro_data |
| Module 02 | `module02_features.db` | technical_indicators, factors, neural_factors |
| Module 03 | `module03_ai_models.db` | models, training_history, predictions |
| Module 04 | `module04_market_analysis.db` | sentiment_analysis, anomalies, agent_analysis |
| Module 05 | `module05_risk_management.db` | portfolio_risk, stop_loss, stress_tests |
| Module 06 | `module06_monitoring_alerting.db` | health_status, alerts, performance_metrics |
| Module 07 | `module07_optimization.db` | portfolio_opt, strategy_opt, hyperparameters |
| Module 08 | `module08_execution.db` | orders, trades, execution_metrics |
| Module 09 | `module09_backtesting.db` | backtest_results, performance, trades |
| Module 10 | `module10_ai_interaction.db` | conversations, strategies, recommendations |
| Module 11 | `module11_visualization.db` | charts, dashboards, reports |

### 数据库关系

```
finloom.db (原始数据)
    ↓
module02_features.db (特征数据)
    ↓
module03_ai_models.db (模型预测)
    ↓
┌───────────────┬───────────────┬───────────────┐
│               │               │               │
module04_*.db   module05_*.db   module07_*.db  module09_*.db
(市场分析)      (风险管理)      (优化结果)      (回测结果)
                                                │
                                                ↓
                                        module08_*.db
                                        (执行记录)
                                                │
                                                ↓
                                        module06_*.db
                                        (监控数据)
```

## 配置文件架构

```yaml
config/
├── system_config.yaml          # 系统全局配置
├── model_config.yaml           # AI模型配置
├── trading_config.yaml         # 交易配置
├── risk_limits.yaml            # 风险限额配置
├── optimization_config.yaml    # 优化配置
└── monitoring_config.yaml      # 监控配置
```

## 模块开发规范

### 1. 代码结构
```python
module_XX_name/
├── __init__.py                 # 模块初始化，导出主要类
├── moduleXX_README.md          # 模块文档
├── core/                       # 核心功能
├── utils/                      # 工具函数
├── api/                        # API接口（如有）
├── database/                   # 数据库管理
└── config/                     # 配置文件
```

### 2. 命名规范
- 类名: `PascalCase`
- 函数名: `snake_case`
- 常量名: `UPPER_SNAKE_CASE`
- 配置类: `XxxConfig`
- 结果类: `XxxResult`
- 管理器类: `XxxManager`

### 3. 文档规范
- 每个模块必须有README.md
- 包含概述、快速开始、API参考、使用示例
- 说明与其他模块的集成方式
- 提供测试代码

### 4. 数据库规范
- 每个模块使用独立的SQLite数据库
- 数据库文件命名: `moduleXX_name.db`
- 提供统一的数据库管理器类
- 实现CRUD基本操作

### 5. 接口规范
- 编程接口: 所有模块必须提供
- REST API: 面向用户的模块提供
- WebSocket: 需要实时推送的模块提供
- 使用统一的错误处理和日志系统

## 测试规范

### 测试文件结构
```python
tests/
├── test_module01_data_pipeline.py
├── test_module02_feature_engineering.py
├── test_module03_ai_models.py
├── test_module04_market_analysis.py
├── test_module05_risk_management.py
├── test_module06_monitoring_alerting.py
├── test_module07_optimization.py
├── test_module08_execution.py
├── test_module09_backtesting.py
├── test_module10_ai_interaction.py
└── test_module11_visualization.py
```

### 测试覆盖要求
- 单元测试: 核心功能类和方法
- 集成测试: 模块间调用
- 端到端测试: 完整业务流程
- 性能测试: 关键操作的性能基准

## 部署架构

### 开发环境
```bash
conda activate study
python main.py
```

### 生产环境
```bash
# 使用Docker部署
docker-compose up -d

# 或使用systemd服务
systemctl start finloom
```

### 服务端口分配
- Main API: `8000`
- WebSocket: `8001`
- Monitoring: `8002`
- AI Interaction: `8003`

## 监控和运维

### 关键监控指标
- 系统健康: CPU、内存、磁盘使用率
- API性能: 响应时间、成功率、QPS
- 数据质量: 数据完整性、延迟、准确性
- 模型性能: 预测准确率、延迟
- 交易执行: 成交率、滑点、延迟
- 风险指标: VaR、回撤、仓位

### 日志级别
- DEBUG: 详细调试信息
- INFO: 一般信息
- WARNING: 警告信息
- ERROR: 错误信息
- CRITICAL: 严重错误

### 告警级别
- INFO: 信息通知
- LOW: 低优先级告警
- MEDIUM: 中优先级告警
- HIGH: 高优先级告警
- CRITICAL: 严重告警

## 总结

FinLoom量化投资系统通过11个专业模块的协同工作，构建了一个完整的量化投资解决方案：

1. **数据层** (Module 00-02): 数据采集、清洗、特征工程
2. **智能层** (Module 03-04): AI模型、市场分析
3. **决策层** (Module 05-07): 风险管理、优化
4. **执行层** (Module 08-09): 交易执行、回测验证
5. **交互层** (Module 10-11): AI对话、可视化展示
6. **监控层** (Module 06): 全方位系统监控

各模块通过清晰的接口定义和数据流向，实现了松耦合、高内聚的系统架构，既保证了各模块的独立性，又确保了整体系统的协同性。


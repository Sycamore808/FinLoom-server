# FinLoom 后端与前端API对应关系文档

## 📋 文档概述

本文档详细说明了FinLoom量化交易系统后端各模块API与前端Vue应用API的对应关系，帮助开发者理解系统架构和数据流向。

**生成时间**: 2025-01-08
**系统版本**: FinLoom v2.0

---

## 🏗️ 系统架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                     前端 Vue3 应用                           │
│                 (web-vue/src/services/api.js)               │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP REST API
                         │
┌────────────────────────┴────────────────────────────────────┐
│                    主服务器 (main.py)                        │
│                   API路由 + FastAPI                          │
└──┬──────┬──────┬───────┬────────┬─────────┬────────┬───────┘
   │      │      │       │        │         │        │
   ▼      ▼      ▼       ▼        ▼         ▼        ▼
Module Module Module Module Module Module Module
  01     02     03     04     05     06     10
数据   特征   AI    市场   风险   监控   AI交互
管道   工程   模型   分析   管理   告警
```

---

## 📡 API接口类型说明

### 后端模块接口类型

| 模块 | 接口类型 | 说明 |
|-----|---------|------|
| **Module 01** | 编程接口 | 不提供REST API，供其他模块调用 |
| **Module 02** | 编程接口 | 不提供REST API，供其他模块调用 |
| **Module 03** | 编程接口 | 不提供REST API，供其他模块调用 |
| **Module 04** | REST API | ✅ 提供完整的REST API接口 |
| **Module 05** | 编程接口 | 不提供REST API，供其他模块调用 |
| **Module 06** | 编程接口 + REST API | 监控数据通过REST API暴露 |
| **Module 07** | 编程接口 | 不提供REST API，供其他模块调用 |
| **Module 08** | 编程接口 | 不提供REST API，供其他模块调用 |
| **Module 09** | 编程接口 | 不提供REST API，供其他模块调用 |
| **Module 10** | REST API | ✅ 提供AI对话REST API接口 |
| **Module 11** | 编程接口 | 生成可视化，供前端直接使用 |

---

## 🔗 前端API与后端模块对应关系

### 1. 健康检查 API

#### 前端调用
```javascript
// web-vue/src/services/api.js
api.health.check() // GET /health
api.health.ready() // GET /v1/ready
```

#### 后端实现
```
主服务器 (main.py)
├── GET /health
│   └── 调用系统健康检查
│       └── Module 06: SystemMonitor.get_health_status()
│
└── GET /v1/ready
    └── 检查所有模块就绪状态
```

#### 数据流
```
前端 → main.py → Module 06 (监控告警) → 返回健康状态
```

---

### 2. AI对话 API

#### 前端调用
```javascript
// web-vue/src/services/api.js
api.chat.send(message, conversationId) // POST /chat
api.chat.aiChat(text, amount, riskTolerance) // POST /v1/ai/chat
```

#### 后端实现
```
主服务器 (main.py)
├── POST /chat
│   └── Module 10: DialogueManager.process_user_input()
│       ├── 调用 NLPProcessor (自然语言处理)
│       ├── 调用 IntentClassifier (意图识别)
│       ├── 调用 RequirementParser (需求解析)
│       └── 调用 ResponseGenerator (生成响应)
│
└── POST /v1/ai/chat
    └── Module 10: FINR1Integration.process_request()
        ├── FIN-R1模型推理
        ├── 调用 ParameterMapper (参数映射)
        └── 调用 RecommendationEngine (生成推荐)
```

#### 数据流
```
前端输入 → main.py → Module 10 (AI交互)
                      ├→ FIN-R1模型
                      ├→ RequirementParser
                      ├→ ParameterMapper
                      │   ├→ Module 05 (风险参数)
                      │   ├→ Module 07 (优化参数)
                      │   └→ Module 09 (回测参数)
                      └→ RecommendationEngine
                          ├→ Module 01 (获取数据)
                          ├→ Module 04 (市场分析)
                          └→ Module 05 (风险评估)
                          └→ 返回推荐策略
```

#### Module 10 提供的接口
- `DialogueManager`: 对话管理
- `RequirementParser`: 投资需求解析
- `ParameterMapper`: 参数映射到各模块
- `RecommendationEngine`: 策略推荐
- `FINR1Integration`: FIN-R1模型集成

---

### 3. 仪表盘 API

#### 前端调用
```javascript
// web-vue/src/services/api.js
api.dashboard.getMetrics() // GET /v1/dashboard/metrics
```

#### 后端实现
```
主服务器 (main.py)
└── GET /v1/dashboard/metrics
    ├── Module 01: 获取市场数据概览
    │   └── get_database_manager().get_database_stats()
    │
    ├── Module 05: 获取投资组合指标
    │   ├── PortfolioRiskAnalyzer.analyze_portfolio_risk()
    │   └── get_risk_database_manager().get_portfolio_risk_history()
    │
    ├── Module 06: 获取系统监控指标
    │   ├── PerformanceMonitor.get_metrics_summary()
    │   └── SystemMonitor.get_system_status()
    │
    └── Module 09: 获取回测结果
        └── get_backtest_database_manager().list_backtests()
```

#### 数据流
```
前端 → main.py → 聚合多个模块数据
                  ├→ Module 01: 市场数据统计
                  ├→ Module 05: 风险指标
                  ├→ Module 06: 系统状态
                  └→ Module 09: 回测结果
                  └→ 返回综合指标
```

---

### 4. 投资组合 API

#### 前端调用
```javascript
// web-vue/src/services/api.js
api.portfolio.getPositions() // GET /v1/portfolio/positions
```

#### 后端实现
```
主服务器 (main.py)
└── GET /v1/portfolio/positions
    ├── Module 05: PortfolioMonitor
    │   ├── 获取当前持仓
    │   ├── 计算未实现盈亏
    │   └── 计算风险指标
    │
    └── Module 01: 获取实时价格
        └── AkshareDataCollector.fetch_realtime_data()
```

#### 涉及的模块功能
- **Module 05**: 
  - `PortfolioMonitor.get_portfolio_metrics()`
  - `get_risk_database_manager().get_portfolio_risk_history()`
- **Module 01**: 
  - `AkshareDataCollector.fetch_realtime_data()` (实时价格)

---

### 5. 交易记录 API

#### 前端调用
```javascript
// web-vue/src/services/api.js
api.trades.getRecent() // GET /v1/trades/recent
```

#### 后端实现
```
主服务器 (main.py)
└── GET /v1/trades/recent
    └── Module 08: ExecutionDatabaseManager
        ├── get_execution_database_manager().get_trades()
        └── get_execution_database_manager().get_orders()
```

#### 涉及的模块功能
- **Module 08**: 
  - `ExecutionDatabaseManager.get_trades()` (成交记录)
  - `ExecutionDatabaseManager.get_orders()` (订单记录)
  - `ExecutionInterface.get_execution_summary()` (执行摘要)

---

### 6. 回测 API

#### 前端调用
```javascript
// web-vue/src/services/api.js
api.backtest.run(params) // POST /v1/backtest/run
```

#### 后端实现
```
主服务器 (main.py)
└── POST /v1/backtest/run
    ├── 1. Module 01: 获取历史数据
    │   └── AkshareDataCollector.fetch_stock_history()
    │
    ├── 2. Module 02: 计算技术指标 (可选)
    │   └── TechnicalIndicators.calculate_all_indicators()
    │
    ├── 3. Module 09: 执行回测
    │   ├── BacktestEngine.load_market_data()
    │   ├── BacktestEngine.set_strategy()
    │   ├── BacktestEngine.run()
    │   └── BacktestDatabaseManager.save_backtest_result()
    │
    ├── 4. Module 09: 性能分析
    │   └── PerformanceAnalyzer.analyze()
    │
    └── 5. Module 11: 生成报告
        └── ReportBuilder.generate_performance_report()
```

#### 完整数据流
```
前端提交回测参数
    ↓
main.py 解析参数
    ↓
Module 01: 获取股票历史数据
    ↓
Module 02: 计算技术指标特征
    ↓
Module 09: BacktestEngine执行回测
    ├→ 应用交易策略
    ├→ 模拟订单执行
    ├→ 计算收益和风险
    └→ 保存到SQLite数据库
    ↓
Module 09: PerformanceAnalyzer分析
    └→ 计算夏普比率、最大回撤等
    ↓
Module 11: ReportBuilder生成报告
    └→ 生成HTML/Excel报告
    ↓
返回回测结果给前端
```

---

### 7. 数据管理 API

#### 前端调用
```javascript
// web-vue/src/services/api.js
api.data.collect(params) // POST /v1/data/collect
api.data.getOverview() // GET /v1/data/overview
```

#### 后端实现
```
主服务器 (main.py)
├── POST /v1/data/collect
│   └── Module 01: 数据采集
│       ├── AkshareDataCollector.fetch_stock_list()
│       ├── AkshareDataCollector.fetch_stock_history()
│       ├── ChineseAlternativeDataCollector.fetch_macro_economic_data()
│       ├── ChineseFundamentalCollector.fetch_financial_statements()
│       └── get_database_manager().save_stock_prices()
│
└── GET /v1/data/overview
    └── Module 01: 数据概览
        └── get_database_manager().get_database_stats()
```

#### Module 01 提供的数据采集功能
- **AkshareDataCollector**: 
  - A股行情数据、实时数据、股票列表
- **ChineseAlternativeDataCollector**: 
  - 宏观数据、新闻数据、板块数据、市场概况
- **ChineseFundamentalCollector**: 
  - 财务报表、财务指标、分红历史、股东信息

---

### 8. 市场分析 API

#### 前端调用
```javascript
// web-vue/src/services/api.js
api.market.getOverview() // GET /v1/market/overview
api.market.analysis.detectAnomaly(params) // POST /v1/analysis/anomaly/detect
api.market.analysis.analyzeCorrelation(params) // POST /v1/analysis/correlation/analyze
api.market.analysis.detectRegime(params) // POST /v1/analysis/regime/detect
api.market.analysis.analyzeSentiment(params) // POST /v1/analysis/sentiment/analyze
api.market.analysis.aggregateSentiment(params) // POST /v1/analysis/sentiment/aggregate
```

#### 后端实现
```
主服务器 (main.py)
├── GET /v1/market/overview
│   └── Module 04: 市场概览
│       ├── MarketMonitor.get_market_summary()
│       └── MarketMonitor.calculate_market_regime()
│
├── POST /v1/analysis/anomaly/detect
│   └── Module 04: 异常检测
│       └── AnomalyDetector.detect_anomalies()
│
├── POST /v1/analysis/correlation/analyze
│   └── Module 04: 相关性分析
│       └── CorrelationCalculator.calculate_correlation_matrix()
│
├── POST /v1/analysis/regime/detect
│   └── Module 04: 市场状态检测
│       └── RegimeDetector.detect_market_regimes()
│
├── POST /v1/analysis/sentiment/analyze
│   └── Module 04: 情感分析
│       └── FINR1SentimentAnalyzer.analyze_stock_sentiment()
│
└── POST /v1/analysis/sentiment/aggregate
    └── Module 04: 情感聚合
        └── EnhancedNewsSentimentAnalyzer.analyze_comprehensive_sentiment()
```

#### Module 04 完整功能集
- **情感分析**: 
  - `FINR1SentimentAnalyzer`: FIN-R1模型情感分析
  - `EnhancedNewsSentimentAnalyzer`: 新闻+基本面综合情感分析
  - `MarketSentimentAggregator`: 市场情感聚合
  
- **异常检测**: 
  - `AnomalyDetector`: 价格、成交量异常检测
  - `MultiDimensionalAnomalyDetector`: 多维度异常检测
  
- **相关性分析**: 
  - `CorrelationCalculator`: 相关性矩阵计算
  - `RollingCorrelationAnalyzer`: 滚动相关性
  
- **市场状态检测**: 
  - `RegimeDetector`: HMM/GMM市场状态检测
  - `MarketMonitor`: 实时市场监控

---

### 9. 通用分析 API (兼容旧版本)

#### 前端调用
```javascript
// web-vue/src/services/api.js
api.analyze(params) // POST /v1/analyze
```

#### 后端实现
```
主服务器 (main.py)
└── POST /v1/analyze (兼容接口)
    ├── 根据params.type路由到不同模块:
    │
    ├── type = "sentiment" → Module 04
    ├── type = "anomaly" → Module 04
    ├── type = "correlation" → Module 04
    ├── type = "risk" → Module 05
    └── type = "backtest" → Module 09
```

---

## 📊 后端模块间调用关系

### Module 01 → 其他模块
```
Module 01 (数据管道)
    ├→ Module 02 (提供原始数据)
    ├→ Module 03 (提供训练数据)
    ├→ Module 04 (提供市场数据)
    ├→ Module 05 (提供价格数据)
    ├→ Module 06 (监控数据源)
    ├→ Module 07 (提供优化数据)
    ├→ Module 08 (提供交易数据)
    ├→ Module 09 (提供回测数据)
    ├→ Module 10 (提供市场数据)
    └→ Module 11 (提供可视化数据)
```

### Module 02 → 其他模块
```
Module 02 (特征工程)
    ├→ Module 03 (提供训练特征)
    ├→ Module 04 (提供技术指标)
    ├→ Module 05 (提供风险特征)
    ├→ Module 07 (提供优化特征)
    ├→ Module 08 (提供交易信号)
    ├→ Module 09 (提供回测特征)
    └→ Module 11 (提供图表数据)
```

### Module 03 → 其他模块
```
Module 03 (AI模型)
    ├→ Module 04 (提供预测)
    ├→ Module 05 (提供风险预测)
    ├→ Module 08 (提供交易信号)
    └→ Module 09 (提供策略预测)
```

### Module 04 → 其他模块
```
Module 04 (市场分析)
    ├→ Module 05 (市场风险评估)
    ├→ Module 08 (市场信号)
    ├→ Module 10 (市场情报)
    └→ Module 11 (分析可视化)
```

### Module 05 → 其他模块
```
Module 05 (风险管理)
    ├→ Module 06 (风险告警)
    ├→ Module 08 (风险过滤)
    ├→ Module 09 (风险回测)
    └→ Module 11 (风险报告)
```

### Module 10 → 其他模块
```
Module 10 (AI交互)
    ├→ Module 01 (获取市场数据)
    ├→ Module 04 (获取市场分析)
    ├→ Module 05 (参数映射: 风险管理)
    ├→ Module 07 (参数映射: 优化)
    ├→ Module 09 (参数映射: 回测)
    └→ Module 11 (生成推荐报告)
```

---

## 🗄️ 数据库使用情况

### 各模块专用数据库

| 模块 | 数据库路径 | 存储内容 |
|-----|-----------|---------|
| **Module 01** | `data/finloom.db` | 股票价格、宏观数据、新闻数据 |
| **Module 02** | `data/module02_features.db` | 技术指标、因子数据、图特征 |
| **Module 03** | `data/module03_ai_models.db` | 模型参数、训练历史、预测结果 |
| **Module 04** | `data/module04_market_analysis.db` | 分析结果、情感数据、异常记录 |
| **Module 05** | `data/module05_risk_management.db` | 风险指标、组合数据、止损记录 |
| **Module 06** | `data/module06_monitoring.db` | 系统监控、告警记录、性能指标 |
| **Module 07** | `data/module07_optimization.db` | 优化结果、参数记录 |
| **Module 08** | `data/module08_execution.db` | 订单记录、成交记录、执行指标 |
| **Module 09** | `data/module09_backtest.db` | 回测结果、交易记录、绩效指标 |
| **Module 10** | `data/module10_ai_interaction.db` | 对话记录、需求解析、推荐记录 |
| **Module 11** | `data/module11_visualization.db` | 图表数据、报告记录、缓存数据 |

---

## 🔄 典型业务流程示例

### 流程1: 用户通过AI对话创建投资策略

```
1. 前端: api.chat.aiChat("投资100万，期限3年，风险适中")
   ↓
2. main.py: 路由到 /v1/ai/chat
   ↓
3. Module 10: DialogueManager处理对话
   ├→ NLPProcessor: 分词、实体识别
   ├→ IntentClassifier: 识别意图 (create_strategy)
   ├→ RequirementParser: 解析投资需求
   │   └→ 输出: ParsedRequirement
   │       ├─ investment_amount: 1000000
   │       ├─ investment_horizon: LONG_TERM
   │       └─ risk_tolerance: MODERATE
   └→ ParameterMapper: 映射到各模块参数
       ├→ risk_params (for Module 05)
       ├→ strategy_params (for Module 08)
       └→ optimization_params (for Module 07)
   ↓
4. Module 10: RecommendationEngine生成推荐
   ├→ 调用 Module 01: 获取市场数据
   ├→ 调用 Module 04: 获取市场分析
   ├→ 调用 Module 05: 评估风险
   └→ 生成3个投资组合推荐
   ↓
5. Module 10: 保存到数据库
   ├→ user_requirements表
   ├→ strategy_recommendations表
   └→ dialogue_sessions表
   ↓
6. 返回推荐给前端显示
```

### 流程2: 用户执行回测

```
1. 前端: api.backtest.run({
     symbols: ["000001", "600036"],
     start_date: "2023-01-01",
     end_date: "2024-12-31",
     strategy: "MA_CROSSOVER"
   })
   ↓
2. main.py: 路由到 /v1/backtest/run
   ↓
3. Module 01: 获取历史数据
   └→ AkshareDataCollector.fetch_stock_history()
      └→ 保存到 finloom.db
   ↓
4. Module 02: 计算技术指标
   └→ TechnicalIndicators.calculate_all_indicators()
      └→ 保存到 module02_features.db
   ↓
5. Module 09: 执行回测
   ├→ BacktestEngine.load_market_data()
   ├→ BacktestEngine.set_strategy(ma_crossover)
   ├→ BacktestEngine.run()
   │   ├→ 调用 Module 08: 生成交易信号
   │   ├→ 调用 Module 05: 风险检查
   │   └→ TransactionSimulator: 模拟交易
   └→ 保存到 module09_backtest.db
   ↓
6. Module 09: 性能分析
   └→ PerformanceAnalyzer.analyze()
      ├→ 计算夏普比率
      ├→ 计算最大回撤
      └→ 生成绩效指标
   ↓
7. Module 11: 生成报告
   └→ ReportBuilder.generate_performance_report()
      ├→ 生成JSON数据
      └→ 保存到 module11_visualization.db
   ↓
8. 返回回测结果给前端
```

### 流程3: 实时监控与告警

```
1. Module 06: SystemMonitor定时运行
   ├→ 监控CPU、内存、磁盘
   ├→ 监控各模块健康状态
   └→ 保存到 module06_monitoring.db
   ↓
2. Module 06: PerformanceMonitor检查指标
   ├→ 检测CPU > 80%
   └→ 触发告警规则
   ↓
3. Module 06: AlertManager处理告警
   ├→ 创建告警记录
   ├→ 保存到数据库
   └→ 调用 NotificationManager
       ├→ 发送邮件通知
       └→ 发送Webhook通知
   ↓
4. 前端: api.health.check() 获取健康状态
   ↓
5. 显示告警信息给用户
```

---

## 📝 前端API完整清单

### 已实现的API (web-vue/src/services/api.js)

```javascript
// 1. 健康检查
api.health.check()              // GET /health
api.health.ready()              // GET /v1/ready

// 2. AI对话
api.chat.send(message, convId)  // POST /chat
api.chat.aiChat(text, amount, risk) // POST /v1/ai/chat

// 3. 仪表盘
api.dashboard.getMetrics()      // GET /v1/dashboard/metrics

// 4. 投资组合
api.portfolio.getPositions()    // GET /v1/portfolio/positions

// 5. 交易记录
api.trades.getRecent()          // GET /v1/trades/recent

// 6. 回测
api.backtest.run(params)        // POST /v1/backtest/run

// 7. 数据管理
api.data.collect(params)        // POST /v1/data/collect
api.data.getOverview()          // GET /v1/data/overview

// 8. 市场分析
api.market.getOverview()        // GET /v1/market/overview
api.market.analysis.detectAnomaly(params)    // POST /v1/analysis/anomaly/detect
api.market.analysis.analyzeCorrelation(params) // POST /v1/analysis/correlation/analyze
api.market.analysis.detectRegime(params)     // POST /v1/analysis/regime/detect
api.market.analysis.analyzeSentiment(params) // POST /v1/analysis/sentiment/analyze
api.market.analysis.aggregateSentiment(params) // POST /v1/analysis/sentiment/aggregate

// 9. 通用分析 (兼容)
api.analyze(params)             // POST /v1/analyze
```

### 建议扩展的API (待实现)

```javascript
// 1. 风险管理API
api.risk.analyzePortfolio(portfolio)    // POST /v1/risk/portfolio/analyze
api.risk.calculateVaR(returns)          // POST /v1/risk/var/calculate
api.risk.checkStopLoss(positions)       // POST /v1/risk/stoploss/check

// 2. 优化API
api.optimization.runBacktest(params)    // POST /v1/optimization/backtest
api.optimization.optimizeParameters(params) // POST /v1/optimization/parameters
api.optimization.getOptimalWeights(returns) // POST /v1/optimization/weights

// 3. 执行API
api.execution.submitOrder(order)        // POST /v1/execution/order/submit
api.execution.getOrderStatus(orderId)   // GET /v1/execution/order/{id}
api.execution.getExecutionSummary()     // GET /v1/execution/summary

// 4. 可视化API
api.visualization.generateChart(config) // POST /v1/visualization/chart
api.visualization.generateReport(data)  // POST /v1/visualization/report
api.visualization.getDashboard(type)    // GET /v1/visualization/dashboard/{type}

// 5. 监控告警API
api.monitoring.getSystemStatus()        // GET /v1/monitoring/system/status
api.monitoring.getAlerts()              // GET /v1/monitoring/alerts
api.monitoring.acknowledgeAlert(id)     // POST /v1/monitoring/alert/{id}/ack
```

---

## 🛠️ 开发建议

### 1. 后端API开发规范

```python
# main.py - 示例API端点

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class BacktestRequest(BaseModel):
    symbols: List[str]
    start_date: str
    end_date: str
    strategy: str
    initial_capital: float = 1000000

@app.post("/v1/backtest/run")
async def run_backtest(request: BacktestRequest):
    """
    回测API端点
    
    调用链:
    1. Module 01: 获取历史数据
    2. Module 02: 计算技术指标
    3. Module 09: 执行回测
    4. Module 11: 生成报告
    """
    try:
        # 1. 获取数据 (Module 01)
        from module_01_data_pipeline import AkshareDataCollector
        collector = AkshareDataCollector()
        market_data = {}
        for symbol in request.symbols:
            data = collector.fetch_stock_history(
                symbol, request.start_date, request.end_date
            )
            market_data[symbol] = data
        
        # 2. 计算特征 (Module 02)
        from module_02_feature_engineering import TechnicalIndicators
        calculator = TechnicalIndicators()
        features = {}
        for symbol, data in market_data.items():
            features[symbol] = calculator.calculate_all_indicators(data)
        
        # 3. 执行回测 (Module 09)
        from module_09_backtesting import BacktestEngine, BacktestConfig
        config = BacktestConfig(
            start_date=datetime.strptime(request.start_date, "%Y%m%d"),
            end_date=datetime.strptime(request.end_date, "%Y%m%d"),
            initial_capital=request.initial_capital,
            strategy_name=request.strategy
        )
        engine = BacktestEngine(config)
        engine.load_market_data(request.symbols, market_data)
        result = engine.run()
        
        # 4. 生成报告 (Module 11)
        from module_11_visualization import ReportBuilder
        report_builder = ReportBuilder()
        report = report_builder.generate_performance_report(
            performance_data=result.to_dict(),
            metrics=result.get_metrics()
        )
        
        return {
            "success": True,
            "backtest_id": engine.backtest_id,
            "result": result.to_dict(),
            "report_path": report['file_path']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 2. 前端API调用规范

```javascript
// web-vue/src/services/api.js - 扩展示例

export const api = {
  // ... 现有API
  
  // 风险管理API
  risk: {
    analyzePortfolio: (portfolio) =>
      apiClient.post('/v1/risk/portfolio/analyze', portfolio),
    
    calculateVaR: (returns, confidenceLevel = 0.95) =>
      apiClient.post('/v1/risk/var/calculate', { 
        returns, 
        confidence_level: confidenceLevel 
      }),
    
    checkStopLoss: (positions) =>
      apiClient.post('/v1/risk/stoploss/check', { positions })
  },
  
  // 优化API
  optimization: {
    optimizeParameters: (params) =>
      apiClient.post('/v1/optimization/parameters', params),
    
    getOptimalWeights: (returns, constraints = {}) =>
      apiClient.post('/v1/optimization/weights', { 
        returns, 
        constraints 
      })
  },
  
  // 执行API
  execution: {
    submitOrder: (order) =>
      apiClient.post('/v1/execution/order/submit', order),
    
    getOrderStatus: (orderId) =>
      apiClient.get(`/v1/execution/order/${orderId}`),
    
    getExecutionSummary: () =>
      apiClient.get('/v1/execution/summary')
  }
}
```

### 3. Vue组件使用示例

```vue
<!-- web-vue/src/views/dashboard/RiskView.vue -->
<template>
  <div class="risk-dashboard">
    <h2>风险分析</h2>
    
    <div v-if="loading">加载中...</div>
    
    <div v-else>
      <div class="risk-metrics">
        <StatCard 
          title="VaR (95%)" 
          :value="riskData.var_95" 
          format="percent"
        />
        <StatCard 
          title="最大回撤" 
          :value="riskData.max_drawdown" 
          format="percent"
        />
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { api } from '@/services/api'
import StatCard from '@/components/ui/StatCard.vue'

const loading = ref(false)
const riskData = ref({})

const loadRiskData = async () => {
  loading.value = true
  try {
    // 调用后端风险分析API
    const portfolio = await api.portfolio.getPositions()
    const result = await api.risk.analyzePortfolio(portfolio.data)
    
    riskData.value = result.data
  } catch (error) {
    console.error('加载风险数据失败:', error)
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  loadRiskData()
})
</script>
```

---

## 📚 参考文档

### 后端模块文档
- Module 01: `/module_01_data_pipeline/module01_README.md`
- Module 02: `/module_02_feature_engineering/module02_README.md`
- Module 03: `/module_03_ai_models/module03_README.md`
- Module 04: `/module_04_market_analysis/module04_README.md`
- Module 05: `/module_05_risk_management/module05_README.md`
- Module 06: `/module_06_monitoring_alerting/module06_README.md`
- Module 07: `/module_07_optimization/module07_README.md`
- Module 08: `/module_08_execution/module08_README.md`
- Module 09: `/module_09_backtesting/module09_README.md`
- Module 10: `/module_10_ai_interaction/module10_README.md`
- Module 11: `/module_11_visualization/module11_README.md`

### 前端文档
- Vue前端: `/web-vue/README.md`
- API服务: `/web-vue/src/services/api.js`

---

## 🔄 版本历史

| 版本 | 日期 | 更新内容 |
|-----|------|---------|
| 1.0 | 2025-01-08 | 初始版本，完整梳理后端与前端API对应关系 |

---

## 📧 联系方式

如有问题或建议，请联系开发团队。

**FinLoom量化交易系统 - 让量化投资更简单**







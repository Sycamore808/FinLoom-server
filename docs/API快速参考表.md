# FinLoom API快速参考表

## 📊 前端API → 后端模块映射表

| 前端API | HTTP方法 | 路由 | 主要调用模块 | 辅助模块 | 功能说明 |
|---------|---------|------|------------|----------|---------|
| `api.health.check()` | GET | `/health` | Module 06 | - | 系统健康检查 |
| `api.health.ready()` | GET | `/v1/ready` | Module 06 | - | 系统就绪状态 |
| `api.chat.send()` | POST | `/chat` | Module 10 | - | AI对话（简化版） |
| `api.chat.aiChat()` | POST | `/v1/ai/chat` | Module 10 | 01, 04, 05 | FIN-R1对话 + 推荐 |
| `api.chat.createConversation()` | POST | `/v1/chat/conversation` | Module 10 | - | 创建新对话会话 |
| `api.chat.getConversations()` | GET | `/v1/chat/conversations` | Module 10 | - | 获取对话列表 |
| `api.chat.getHistory()` | GET | `/v1/chat/history/{id}` | Module 10 | - | 获取对话历史 |
| `api.chat.deleteConversation()` | DELETE | `/v1/chat/conversation/{id}` | Module 10 | - | 删除对话 |
| `api.chat.searchConversations()` | GET | `/v1/chat/search` | Module 10 | - | 搜索对话 |
| `api.chat.addFavorite()` | POST | `/v1/chat/favorite` | Module 10 | - | 收藏对话 |
| `api.chat.removeFavorite()` | DELETE | `/v1/chat/favorite/{id}` | Module 10 | - | 取消收藏对话 |
| `api.chat.getFavorites()` | GET | `/v1/chat/favorites` | Module 10 | - | 获取收藏列表 |
| `api.chat.checkFavorite()` | GET | `/v1/chat/favorite/check/{id}` | Module 10 | - | 检查收藏状态 |
| `api.chat.updateFavorite()` | PUT | `/v1/chat/favorite/{id}` | Module 10 | - | 更新收藏信息 |
| `api.strategy.generate()` | POST | `/v1/strategy/generate` | Module 10, 07 | - | 生成投资策略 |
| `api.strategy.save()` | POST | `/v1/strategy/save` | Module 07 | - | 保存策略 |
| `api.strategy.list()` | GET | `/v1/strategy/list` | Module 07 | - | 获取策略列表 |
| `api.strategy.get()` | GET | `/v1/strategy/{id}` | Module 07 | - | 获取策略详情 |
| `api.strategy.delete()` | DELETE | `/v1/strategy/{id}` | Module 07 | - | 删除策略 |
| `api.strategy.duplicate()` | POST | `/v1/strategy/{id}/duplicate` | Module 07 | - | 复制策略 |
| `api.strategy.optimize()` | POST | `/v1/strategy/optimize` | Module 07 | - | 优化策略参数 |
| `api.strategy.templates.list()` | GET | `/v1/strategy/templates` | Module 07 | - | 获取策略模板 |
| `api.strategy.templates.get()` | GET | `/v1/strategy/templates/{id}` | Module 07 | - | 获取模板详情 |
| `api.strategy.templates.createFrom()` | POST | `/v1/strategy/from-template/{id}` | Module 07 | - | 从模板创建策略 |
| `api.dashboard.getMetrics()` | GET | `/v1/dashboard/metrics` | Module 05, 06 | 01, 09 | 仪表盘综合指标 |
| `api.portfolio.getPositions()` | GET | `/v1/portfolio/positions` | Module 05 | 01 | 投资组合持仓 |
| `api.trades.getRecent()` | GET | `/v1/trades/recent` | Module 08 | - | 最近交易记录 |
| `api.backtest.run()` | POST | `/v1/backtest/run` | Module 09 | 01, 02, 11 | 执行回测 |
| `api.data.collect()` | POST | `/v1/data/collect` | Module 01 | - | 数据采集 |
| `api.data.getOverview()` | GET | `/v1/data/overview` | Module 01 | - | 数据概览 |
| `api.market.getOverview()` | GET | `/v1/market/overview` | Module 04 | 01 | 市场概览 |
| `api.market.analysis.detectAnomaly()` | POST | `/v1/analysis/anomaly/detect` | Module 04 | - | 异常检测 |
| `api.market.analysis.analyzeCorrelation()` | POST | `/v1/analysis/correlation/analyze` | Module 04 | - | 相关性分析 |
| `api.market.analysis.detectRegime()` | POST | `/v1/analysis/regime/detect` | Module 04 | - | 市场状态检测 |
| `api.market.analysis.analyzeSentiment()` | POST | `/v1/analysis/sentiment/analyze` | Module 04 | - | 情感分析 |
| `api.market.analysis.aggregateSentiment()` | POST | `/v1/analysis/sentiment/aggregate` | Module 04 | - | 情感聚合 |
| `api.analyze()` | POST | `/v1/analyze` | 路由分发 | 04, 05, 09 | 通用分析接口 |

---

## 🗄️ 后端模块数据库速查

| 模块 | 数据库文件 | 主要数据表 |
|-----|-----------|----------|
| Module 01 | `finloom.db` | stock_prices, macro_data, news_data |
| Module 02 | `module02_features.db` | technical_indicators, factors, graph_features |
| Module 03 | `module03_ai_models.db` | models, training_history, predictions |
| Module 04 | `module04_market_analysis.db` | sentiment_analysis, anomaly_records, correlations |
| Module 05 | `module05_risk_management.db` | portfolio_risk, stop_loss_records, positions |
| Module 06 | `module06_monitoring.db` | system_metrics, alerts, performance_logs |
| Module 07 | `module07_optimization.db` | optimization_results, parameter_records |
| Module 08 | `module08_execution.db` | orders, trades, execution_metrics |
| Module 09 | `module09_backtest.db` | backtest_results, trade_records, performance |
| Module 10 | `module10_ai_interaction.db` | user_requirements, strategy_recommendations, dialogue_sessions |
| Module 11 | `module11_visualization.db` | charts, dashboards, reports, export_history |

---

## 🔄 后端模块依赖关系

### 模块调用链

```
┌──────────────────────────────────────────────────────┐
│ Module 01 (数据管道) - 基础数据层                     │
│ 为所有模块提供数据                                    │
└────────────┬─────────────────────────────────────────┘
             │
     ┌───────┴────────┬──────────┬──────────┐
     ▼                ▼          ▼          ▼
┌─────────┐    ┌──────────┐  ┌──────┐  ┌──────┐
│Module 02│    │Module 03 │  │Mod 04│  │Mod 05│
│特征工程 │    │AI模型    │  │分析  │  │风险  │
└────┬────┘    └─────┬────┘  └───┬──┘  └───┬──┘
     │               │           │         │
     └───────────────┴───────────┴─────────┤
                                           │
                ┌──────────────────────────┤
                ▼                          ▼
         ┌──────────┐              ┌──────────┐
         │Module 08 │              │Module 09 │
         │执行      │              │回测      │
         └──────────┘              └─────┬────┘
                                         │
                                         ▼
                                  ┌──────────┐
                                  │Module 11 │
                                  │可视化    │
                                  └──────────┘

┌─────────────────────────────────────────────┐
│ Module 10 (AI交互) - 协调层                  │
│ 调用多个模块提供统一的AI交互接口             │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ Module 06 (监控告警) - 监控层                │
│ 监控所有模块的运行状态                       │
└─────────────────────────────────────────────┘
```

---

## 📦 核心模块功能速查

### Module 01 - 数据管道
**核心功能**: 数据采集、存储、管理
**主要类**: 
- `AkshareDataCollector` - A股数据采集
- `ChineseAlternativeDataCollector` - 宏观/新闻数据
- `ChineseFundamentalCollector` - 财务数据

**提供数据**: 行情、宏观、新闻、财务、实时价格

---

### Module 02 - 特征工程
**核心功能**: 技术指标计算、因子发现、图特征
**主要类**:
- `TechnicalIndicators` - 技术指标
- `FactorCalculator` - 因子计算
- `GraphConstructor` - 图网络构建

**提供数据**: MA/RSI/MACD等指标、自定义因子

---

### Module 03 - AI模型
**核心功能**: 深度学习、强化学习、集成学习
**主要类**:
- `LSTMModel` - 时序预测
- `GRUModel` - 时序预测
- `DQNAgent` - 强化学习交易

**提供数据**: 价格预测、交易信号、模型评分

---

### Module 04 - 市场分析 ⭐
**核心功能**: 情感分析、异常检测、相关性分析、市场状态检测
**主要类**:
- `FINR1SentimentAnalyzer` - FIN-R1情感分析
- `AnomalyDetector` - 异常检测
- `CorrelationCalculator` - 相关性分析
- `RegimeDetector` - HMM市场状态检测

**REST API**: ✅ 完整暴露

---

### Module 05 - 风险管理
**核心功能**: 风险分析、组合优化、止损、压力测试
**主要类**:
- `PortfolioRiskAnalyzer` - 风险分析
- `MeanVarianceOptimizer` - 均值方差优化
- `DynamicStopLoss` - 动态止损

**提供数据**: VaR、夏普比率、最大回撤、仓位建议

---

### Module 06 - 监控告警
**核心功能**: 系统监控、性能监控、告警管理
**主要类**:
- `SystemMonitor` - 系统监控
- `PerformanceMonitor` - 性能监控
- `AlertManager` - 告警管理

**提供数据**: CPU/内存/磁盘状态、告警记录

---

### Module 07 - 优化
**核心功能**: 超参数调优、多目标优化、策略优化
**主要类**:
- `GridSearchOptimizer` - 网格搜索
- `BayesianOptimizer` - 贝叶斯优化
- `GeneticOptimizer` - 遗传算法

**提供数据**: 最优参数、帕累托前沿

---

### Module 08 - 交易执行
**核心功能**: 订单管理、执行算法、市场影响模型
**主要类**:
- `OrderManager` - 订单管理
- `ExecutionInterface` - 执行接口
- `SignalGenerator` - 信号生成

**提供数据**: 订单记录、成交记录、执行摘要

---

### Module 09 - 回测验证
**核心功能**: 回测引擎、绩效分析、走查分析
**主要类**:
- `BacktestEngine` - 回测引擎
- `PerformanceAnalyzer` - 绩效分析
- `WalkForwardAnalyzer` - 走查分析

**提供数据**: 回测结果、绩效指标、交易记录

---

### Module 10 - AI交互 ⭐
**核心功能**: 自然语言理解、对话管理、参数映射、推荐引擎
**主要类**:
- `DialogueManager` - 对话管理
- `RequirementParser` - 需求解析
- `ParameterMapper` - 参数映射
- `RecommendationEngine` - 推荐引擎
- `FINR1Integration` - FIN-R1集成

**REST API**: ✅ 完整暴露

**核心流程**:
```
用户输入 → NLP处理 → 意图识别 → 需求解析 
         → 参数映射 → 调用其他模块 → 生成推荐 → 返回响应
```

---

### Module 11 - 可视化
**核心功能**: 图表生成、报告构建、仪表板、数据导出
**主要类**:
- `ChartGenerator` - 图表生成
- `ReportBuilder` - 报告构建（JSON/CSV/Excel）
- `DashboardManager` - 仪表板管理
- `InteractiveVisualizer` - 交互式可视化

**提供数据**: HTML图表、JSON报告、Excel报告

---

## 🚀 典型使用场景

### 场景1: 用户对话创建策略

```
前端 → api.chat.aiChat("投资100万，风险适中")
      ↓
Module 10 → 解析需求 → 映射参数
           ↓           ↓
      Module 05   Module 07
      (风险)      (优化)
           ↓
      生成推荐 → 返回前端
```

### 场景2: 执行回测

```
前端 → api.backtest.run({symbols, dates, strategy})
      ↓
Module 01 → 获取历史数据
      ↓
Module 02 → 计算技术指标
      ↓
Module 09 → 执行回测
      ↓
Module 11 → 生成报告 → 返回前端
```

### 场景3: 市场分析

```
前端 → api.market.analysis.analyzeSentiment({symbol})
      ↓
Module 04 → FINR1SentimentAnalyzer
      ↓
      分析新闻情感 → 返回前端
```

---

## 📱 前端页面与API对应

| 前端页面 | 路由 | 主要API |
|---------|------|--------|
| 启动页 | `/` | - |
| 登录页 | `/login` | - |
| 首页 | `/home` | `api.health.check()` |
| 仪表盘概览 | `/dashboard` | `api.dashboard.getMetrics()` |
| 投资组合 | `/dashboard/portfolio` | `api.portfolio.getPositions()` |
| 交易记录 | `/dashboard/trades` | `api.trades.getRecent()` |
| 策略回测 | `/dashboard/backtest` | `api.backtest.run()` |
| 数据管理 | `/dashboard/data` | `api.data.collect()`, `api.data.getOverview()` |
| 市场分析 | `/dashboard/market` | `api.market.getOverview()`, `api.market.analysis.*` |
| AI对话 | `/dashboard/chat` | `api.chat.aiChat()` |
| 新对话 | `/dashboard/chat/new` | `api.chat.createConversation()` |
| 历史记录 | `/dashboard/chat/history` | `api.chat.getConversations()`, `api.chat.searchConversations()` |
| 收藏对话 | `/dashboard/chat/favorites` | `api.chat.getFavorites()`, `api.chat.addFavorite()` |
| 策略模式 | `/dashboard/strategy` | `api.chat.send()`, `api.backtest.run()` |
| 创建策略 | `/dashboard/strategy/create` | `api.strategy.generate()`, `api.strategy.save()` |
| 策略库 | `/dashboard/strategy/library` | `api.strategy.list()`, `api.strategy.duplicate()` |
| 策略模板 | `/dashboard/strategy/templates` | `api.strategy.templates.list()`, `api.strategy.templates.createFrom()` |

---

## 🔧 扩展建议

### 待实现的前端API

```javascript
// 1. 风险管理
api.risk.analyzePortfolio(portfolio)
api.risk.calculateVaR(returns)

// 2. 优化
api.optimization.optimizeParameters(params)
api.optimization.getOptimalWeights(returns)

// 3. 执行
api.execution.submitOrder(order)
api.execution.getOrderStatus(orderId)

// 4. 可视化
api.visualization.generateChart(config)
api.visualization.generateReport(data)

// 5. 监控
api.monitoring.getSystemStatus()
api.monitoring.getAlerts()
```

### 待实现的后端API端点

```python
# main.py

# 风险管理
@app.post("/v1/risk/portfolio/analyze")
@app.post("/v1/risk/var/calculate")

# 优化
@app.post("/v1/optimization/parameters")
@app.post("/v1/optimization/weights")

# 执行
@app.post("/v1/execution/order/submit")
@app.get("/v1/execution/order/{id}")

# 可视化
@app.post("/v1/visualization/chart")
@app.post("/v1/visualization/report")

# 监控
@app.get("/v1/monitoring/system/status")
@app.get("/v1/monitoring/alerts")
```

---

## 📚 相关文档

- 完整API文档: `docs/API对应关系文档.md`
- 前端README: `web-vue/README.md`
- 后端模块README: `module_XX_*/moduleXX_README.md`

---

**FinLoom量化交易系统 - API快速参考**


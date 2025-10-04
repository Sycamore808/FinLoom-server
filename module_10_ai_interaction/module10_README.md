# Module 10 - AI交互模块

## 概述

AI交互模块是 FinLoom 量化交易系统的智能交互入口，集成FIN-R1模型提供自然语言理解能力，支持用户通过对话方式表达投资需求，系统自动解析并生成量化策略。

## 主要功能

### 1. 自然语言处理 (NLP Processing)
- **NLPProcessor**: 自然语言处理器
- **RequirementParser**: 投资需求解析
- **IntentRecognition**: 意图识别
- **EntityExtraction**: 实体抽取

### 2. 策略生成 (Strategy Generation)
- **StrategyGenerator**: 策略自动生成
- **StrategyTemplate**: 策略模板库
- **ParameterOptimizer**: 参数自动优化
- **StrategyValidator**: 策略验证

### 3. 对话管理 (Conversation Management)
- **ConversationManager**: 多轮对话管理
- **ContextTracker**: 上下文追踪
- **SessionManager**: 会话管理
- **DialogueStateManager**: 对话状态管理

### 4. 知识库 (Knowledge Base)
- **FinancialKnowledgeBase**: 金融知识库
- **StrategyLibrary**: 策略库
- **FactorDatabase**: 因子数据库
- **ModelRegistry**: 模型注册表

### 5. 推荐系统 (Recommendation System)
- **StrategyRecommender**: 策略推荐
- **RiskProfiler**: 风险画像
- **PortfolioAdvisor**: 组合建议
- **PersonalizedSuggester**: 个性化推荐

## 快速开始

### 基础使用示例

```python
from module_10_ai_interaction import (
    NLPProcessor,
    StrategyGenerator,
    ConversationManager,
    StrategyRecommender
)

# 1. 初始化AI交互系统
nlp_processor = NLPProcessor()
strategy_gen = StrategyGenerator()
conversation_mgr = ConversationManager()
recommender = StrategyRecommender()

# 2. 用户输入自然语言需求
user_input = "我想要一个适合中小盘成长股的量化策略，风险偏好中等，期望年化收益20%以上"

# 3. 解析用户需求
parsed_requirement = nlp_processor.parse_requirement(user_input)

print("需求解析结果:")
print(f"  市值偏好: {parsed_requirement['market_cap']}")  # ['小盘', '中盘']
print(f"  风格偏好: {parsed_requirement['style']}")       # '成长'
print(f"  风险偏好: {parsed_requirement['risk_profile']}")  # '中等'
print(f"  收益目标: {parsed_requirement['return_target']}")  # 0.20
print(f"  投资周期: {parsed_requirement['time_horizon']}")  # '中期'

# 4. 生成策略
strategy = strategy_gen.generate_strategy(parsed_requirement)

print("\n生成的策略:")
print(f"  策略名称: {strategy['name']}")
print(f"  策略类型: {strategy['type']}")
print(f"  因子组合: {strategy['factors']}")
print(f"  选股数量: {strategy['num_stocks']}")
print(f"  调仓频率: {strategy['rebalance_frequency']}")
print(f"  止损阈值: {strategy['stop_loss']}")

# 5. 回测验证
from module_09_backtesting import BacktestEngine

backtest = BacktestEngine(config)
backtest_result = backtest.run_backtest(
    strategy=strategy['function'],
    symbols=strategy['universe'],
    context=strategy['context']
)

# 6. 生成推荐
recommendation = recommender.generate_recommendation(
    strategy=strategy,
    backtest_result=backtest_result,
    user_profile=parsed_requirement
)

print("\n策略推荐:")
print(f"  推荐度: {recommendation['score']:.1f}/10")
print(f"  回测收益: {recommendation['backtest_return']:.2%}")
print(f"  夏普比率: {recommendation['sharpe_ratio']:.2f}")
print(f"  最大回撤: {recommendation['max_drawdown']:.2%}")
print(f"  风险评级: {recommendation['risk_rating']}")
print(f"  建议: {recommendation['advice']}")

# 7. 多轮对话
conversation = conversation_mgr.start_conversation(user_id='user_001')

while True:
    user_message = input("\n用户: ")
    if user_message.lower() in ['退出', 'quit', 'exit']:
        break
    
    # 处理用户输入
    response = conversation_mgr.process_message(
        conversation_id=conversation.id,
        message=user_message
    )
    
    print(f"助手: {response['text']}")
    
    # 如果需要执行操作
    if response['action']:
        action_result = conversation_mgr.execute_action(
            action=response['action'],
            parameters=response['parameters']
        )
        print(f"操作结果: {action_result}")

print("\n✅ AI交互完成！")
```

## API 参考

### NLPProcessor

自然语言处理器。

#### 主要方法

**parse_requirement(text: str) -> Dict[str, Any]**
- 解析投资需求
- 返回结构化需求

**extract_entities(text: str) -> List[Dict]**
- 提取命名实体
- 识别股票、指标、时间等

**recognize_intent(text: str) -> str**
- 识别用户意图
- 'create_strategy', 'backtest', 'optimize', 'query'等

#### 使用示例
```python
nlp = NLPProcessor()

requirement = nlp.parse_requirement("找一些低估值的大盘蓝筹股")
print(requirement)
# {'market_cap': ['大盘'], 'valuation': '低估值', 'style': '蓝筹'}
```

### StrategyGenerator

策略自动生成器。

#### 主要方法

**generate_strategy(requirement: Dict) -> Dict[str, Any]**
- 根据需求生成策略
- 返回完整策略配置

**optimize_parameters(strategy: Dict, data: pd.DataFrame) -> Dict**
- 优化策略参数
- 使用Module 07优化模块

**validate_strategy(strategy: Dict) -> ValidationResult**
- 验证策略合理性

#### 使用示例
```python
strategy_gen = StrategyGenerator()

strategy = strategy_gen.generate_strategy({
    'market_cap': ['小盘', '中盘'],
    'style': '成长',
    'risk_profile': '中等'
})
```

### ConversationManager

多轮对话管理器。

#### 主要方法

**start_conversation(user_id: str) -> Conversation**
- 启动新对话

**process_message(conversation_id: str, message: str) -> Dict[str, Any]**
- 处理用户消息
- 返回助手回复

**get_conversation_history(conversation_id: str) -> List[Dict]**
- 获取对话历史

**end_conversation(conversation_id: str) -> bool**
- 结束对话

#### 使用示例
```python
conv_mgr = ConversationManager()

conversation = conv_mgr.start_conversation('user_001')
response = conv_mgr.process_message(conversation.id, "帮我设计一个策略")
```

### StrategyRecommender

策略推荐系统。

#### 主要方法

**generate_recommendation(strategy: Dict, backtest_result: BacktestResult, user_profile: Dict) -> Dict[str, Any]**
- 生成策略推荐

**calculate_recommendation_score(strategy: Dict, performance: Dict, user_profile: Dict) -> float**
- 计算推荐分数

**explain_recommendation(recommendation: Dict) -> str**
- 解释推荐理由

#### 使用示例
```python
recommender = StrategyRecommender()

recommendation = recommender.generate_recommendation(
    strategy=strategy,
    backtest_result=backtest_result,
    user_profile=user_profile
)
```

## REST API 端点

### 对话接口
```
POST /api/v1/ai/conversation/start
POST /api/v1/ai/conversation/message
GET /api/v1/ai/conversation/{conversation_id}/history
POST /api/v1/ai/conversation/{conversation_id}/end
```

### 策略生成接口
```
POST /api/v1/ai/strategy/generate
POST /api/v1/ai/strategy/optimize
GET /api/v1/ai/strategy/templates
POST /api/v1/ai/strategy/backtest
```

### 推荐接口
```
GET /api/v1/ai/recommend/strategies
POST /api/v1/ai/recommend/portfolio
GET /api/v1/ai/recommend/factors
```

## WebSocket接口

```javascript
// 连接WebSocket
const ws = new WebSocket('ws://localhost:8000/ws/ai/chat');

// 发送消息
ws.send(JSON.stringify({
    type: 'message',
    conversation_id: 'conv_123',
    message: '帮我分析一下000001的投资价值'
}));

// 接收回复
ws.onmessage = (event) => {
    const response = JSON.parse(event.data);
    console.log('助手:', response.text);
};
```

## 与其他模块集成

### 完整的AI驱动投资流程
```python
async def ai_driven_investment_workflow(user_input: str):
    """AI驱动的完整投资流程"""
    
    # 1. 解析需求 (Module 10)
    nlp = NLPProcessor()
    requirement = nlp.parse_requirement(user_input)
    
    # 2. 生成策略 (Module 10)
    strategy_gen = StrategyGenerator()
    strategy = strategy_gen.generate_strategy(requirement)
    
    # 3. 获取数据 (Module 01)
    from module_01_data_pipeline import AkshareDataCollector
    collector = AkshareDataCollector()
    data = collector.fetch_stock_history(strategy['symbols'], ...)
    
    # 4. 计算特征 (Module 02)
    from module_02_feature_engineering import TechnicalIndicators
    calculator = TechnicalIndicators()
    features = calculator.calculate_all_indicators(data)
    
    # 5. 训练模型 (Module 03)
    from module_03_ai_models import LSTMModel
    model = LSTMModel(config)
    model.train(features, targets)
    
    # 6. 优化参数 (Module 07)
    from module_07_optimization import StrategyOptimizer
    optimizer = StrategyOptimizer(config)
    best_params = optimizer.optimize_strategy(strategy, data)
    
    # 7. 回测验证 (Module 09)
    from module_09_backtesting import BacktestEngine
    backtest = BacktestEngine(config)
    backtest_result = backtest.run_backtest(strategy, symbols)
    
    # 8. 风险评估 (Module 05)
    from module_05_risk_management import PortfolioRiskAnalyzer
    risk_analyzer = PortfolioRiskAnalyzer(config)
    risk_assessment = risk_analyzer.analyze_portfolio_risk(portfolio, returns)
    
    # 9. 生成推荐 (Module 10)
    recommender = StrategyRecommender()
    recommendation = recommender.generate_recommendation(
        strategy, backtest_result, requirement
    )
    
    # 10. 可视化展示 (Module 11)
    from module_11_visualization import ChartGenerator
    chart_gen = ChartGenerator()
    charts = chart_gen.generate_strategy_report(
        strategy, backtest_result, recommendation
    )
    
    return {
        'requirement': requirement,
        'strategy': strategy,
        'backtest': backtest_result,
        'risk': risk_assessment,
        'recommendation': recommendation,
        'charts': charts
    }
```

## 测试和示例

### 运行测试
```bash
cd /Users/victor/Desktop/25fininnov/FinLoom-server
python tests/module10_ai_interaction_test.py
```

## 配置说明

### 环境变量
- `MODULE10_DB_PATH`: AI交互数据库路径
- `MODULE10_FIN_R1_MODEL`: FIN-R1模型路径
- `MODULE10_KNOWLEDGE_BASE`: 知识库路径

### 配置文件
```yaml
# config/ai_interaction_config.yaml
nlp:
  model: 'fin-r1'
  language: 'zh-CN'
  confidence_threshold: 0.7

strategy_generation:
  default_template: 'multi_factor'
  auto_optimize: true
  backtest_validation: true

conversation:
  max_history_length: 50
  session_timeout: 3600
  enable_context_tracking: true

recommendation:
  min_backtest_period: 252
  min_sharpe_ratio: 1.0
  max_drawdown_threshold: 0.20
```

## 总结

Module 10 提供了智能化的投资交互体验：

### 功能完整性 ✅
- ✓ 自然语言理解和需求解析
- ✓ 策略自动生成和优化
- ✓ 多轮对话管理
- ✓ 智能推荐系统

### 集成能力 ✅
- ✓ 编排所有模块完成完整流程
- ✓ FIN-R1模型集成
- ✓ REST API和WebSocket支持
- ✓ 知识库和模板系统

**结论**: Module 10 是系统的智能大脑，让量化投资变得简单易用。


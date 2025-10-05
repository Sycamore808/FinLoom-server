# Module 10 - AI交互模块

## 概述

AI交互模块是 FinLoom 量化交易系统的智能交互入口，集成FIN-R1模型提供自然语言理解能力，支持用户通过对话方式表达投资需求，系统自动解析并生成量化策略参数。

## 主要功能

### 1. 自然语言理解 (NLP)
- **NLPProcessor**: 中文金融NLP处理
- **IntentClassifier**: 用户意图识别
- **RequirementParser**: 投资需求结构化解析

### 2. 对话管理 (Dialogue Management)
- **DialogueManager**: 多轮对话状态管理
- **ConversationHistoryManager**: 对话历史存储与查询

### 3. 参数映射 (Parameter Mapping)
- **ParameterMapper**: 需求到系统参数映射
- **模块参数适配**: 适配各模块的参数格式

### 4. 智能推荐 (Recommendation)
- **RecommendationEngine**: 投资组合和策略推荐
- **个性化推荐**: 基于用户画像的推荐

### 5. FIN-R1集成
- **FINR1Integration**: FIN-R1模型调用封装
- **深度语义理解**: 投资需求深度解析

## 快速开始

### 基础使用示例

```python
from module_10_ai_interaction import (
    DialogueManager,
    RequirementParser,
    ParameterMapper,
    RecommendationEngine,
    get_database_manager
)

# 1. 初始化组件
dialogue_mgr = DialogueManager()
parser = RequirementParser()
mapper = ParameterMapper()
recommender = RecommendationEngine()
db_manager = get_database_manager()

# 2. 启动对话
conversation = dialogue_mgr.start_conversation(user_id='user_001')
print(f"会话ID: {conversation.session_id}")

# 3. 处理用户输入
user_input = "我想投资10万元，期限3年，希望年化收益15%以上，风险适中"
result = dialogue_mgr.process_user_input(conversation.session_id, user_input)
print(f"系统响应: {result['response']}")
print(f"当前状态: {result['state']}")

# 4. 解析投资需求
parsed = parser.parse_requirement(user_input)
print(f"投资金额: {parsed.investment_amount}")
print(f"风险偏好: {parsed.risk_tolerance}")
print(f"投资期限: {parsed.investment_horizon}")

# 5. 映射到系统参数
system_params = mapper.map_to_system_parameters(parsed)
print(f"风险参数: {system_params['risk_params']}")
print(f"策略参数: {system_params['strategy_params']}")

# 6. 生成推荐
user_profile = {
    'risk_tolerance': 'moderate',
    'investment_horizon': 'long_term',
    'goals': ['wealth_growth']
}
market_conditions = {
    'trend': 'neutral',
    'volatility': 'medium'
}
recommendations = recommender.generate_portfolio_recommendations(
    user_profile, market_conditions, num_recommendations=3
)
for rec in recommendations:
    print(f"推荐: {rec.name} (适合度: {rec.suitability_score:.2f})")
    print(f"  预期收益: {rec.expected_metrics['expected_return']:.2%}")
    print(f"  风险指标: VaR={rec.risk_metrics['var_95']:.2%}")

# 7. 保存到数据库
requirement_id = db_manager.save_user_requirement(
    user_id='user_001',
    session_id=conversation.session_id,
    raw_input=user_input,
    parsed_data=parsed.to_dict(),
    system_parameters=system_params
)
print(f"需求ID: {requirement_id}")
```

## API 参考

### RequirementParser

投资需求解析器，将自然语言需求转换为结构化数据。

#### 主要方法

**parse_requirement(text: str) -> ParsedRequirement**
- 解析用户投资需求
- 返回：`ParsedRequirement` 对象
- 示例：
```python
parser = RequirementParser()
parsed = parser.parse_requirement("投资50万，低风险，长期持有")
print(parsed.investment_amount)  # 500000.0
print(parsed.risk_tolerance)  # RiskTolerance.CONSERVATIVE
print(parsed.investment_horizon)  # InvestmentHorizon.LONG_TERM
```

**extract_risk_preferences(text: str) -> Tuple[Optional[RiskTolerance], float]**
- 提取风险偏好
- 返回：(风险等级, 置信度)
- 示例：
```python
risk, confidence = parser.extract_risk_preferences("我比较保守，不想亏钱")
print(risk)  # RiskTolerance.CONSERVATIVE
print(confidence)  # 0.7
```

**parse_investment_goals(text: str) -> Tuple[List[InvestmentGoal], float]**
- 解析投资目标
- 返回：(目标列表, 置信度)

**identify_constraints(text: str) -> List[InvestmentConstraint]**
- 识别投资约束
- 返回：约束列表

**map_to_system_parameters(parsed_req: ParsedRequirement) -> Dict[str, Any]**
- 映射到系统参数
- 返回：系统参数字典

### NLPProcessor

自然语言处理器，提供分词、实体识别、情感分析等功能。

#### 主要方法

**tokenize(text: str, use_pos_tagging: bool = False) -> List[str]**
- 中文分词
- 参数：
  - `text`: 输入文本
  - `use_pos_tagging`: 是否使用词性标注
- 返回：分词列表
- 示例：
```python
processor = NLPProcessor()
tokens = processor.tokenize("我想买入平安银行的股票")
print(tokens)  # ['我', '想', '买入', '平安银行', '的', '股票']
```

**extract_entities(text: str) -> List[TextEntity]**
- 提取命名实体（金额、日期、百分比、组织）
- 返回：实体列表
- 示例：
```python
entities = processor.extract_entities("投资10万元，预期收益15%")
for entity in entities:
    print(f"{entity.entity_type}: {entity.value} -> {entity.normalized_value}")
# MONEY: 10万元 -> 100000.0
# PERCENT: 15% -> 0.15
```

**analyze_sentiment(text: str, domain: str = "finance") -> SentimentResult**
- 情感分析
- 返回：`SentimentResult` 对象
- 示例：
```python
sentiment = processor.analyze_sentiment("市场走势强劲，看好未来表现")
print(sentiment.sentiment)  # 'positive'
print(sentiment.score)  # 0.8
print(sentiment.confidence)  # 0.85
```

**extract_keywords(text: str, top_k: int = 10, method: str = "tfidf") -> List[Tuple[str, float]]**
- 提取关键词
- 参数：
  - `method`: 'tfidf' 或 'textrank'
- 返回：(关键词, 权重) 列表

### IntentClassifier

意图分类器，识别用户输入的意图类型。

#### 主要方法

**classify(text: str, context: Optional[Dict[str, Any]] = None) -> Tuple[str, float, Dict[str, Any]]**
- 分类用户意图
- 返回：(意图, 置信度, 提取的实体)
- 支持的意图类型：
  - `greeting`: 问候
  - `goodbye`: 告别
  - `create_strategy`: 创建策略
  - `modify_strategy`: 修改策略
  - `query_price`: 查询价格
  - `query_performance`: 查询业绩
  - `query_risk`: 查询风险
  - `buy_signal`: 买入信号
  - `sell_signal`: 卖出信号
  - `market_analysis`: 市场分析
  - `ask_advice`: 寻求建议
  - `backtest`: 回测
  - `optimize`: 优化
  - `confirm`: 确认
  - `reject`: 拒绝
  - `help`: 帮助
  - `explain`: 解释说明
- 示例：
```python
classifier = IntentClassifier()
intent, confidence, entities = classifier.classify("帮我设计一个投资策略")
print(intent)  # 'create_strategy'
print(confidence)  # 0.9
```

**extract_entities(text: str) -> Dict[str, Any]**
- 提取实体信息
- 返回：实体字典

### DialogueManager

对话管理器，管理多轮对话的状态和流程。

#### 主要方法

**start_conversation(user_id: str) -> DialogueContext**
- 启动新对话
- 返回：对话上下文
- 示例：
```python
mgr = DialogueManager()
context = mgr.start_conversation('user_001')
print(context.session_id)  # 会话ID
print(context.current_state)  # DialogueState.GREETING
```

**process_user_input(session_id: str, user_input: str) -> Dict[str, Any]**
- 处理用户输入
- 返回：处理结果字典
  - `response`: 系统响应文本
  - `state`: 当前对话状态
  - `intent`: 识别的意图
  - `confidence`: 置信度
  - `turn_count`: 回合数
  - `needs_clarification`: 是否需要澄清
  - `collected_info`: 收集到的信息
- 示例：
```python
result = mgr.process_user_input(session_id, "我想投资10万")
print(result['response'])
print(result['state'])  # 当前状态
```

**get_conversation_context(session_id: str) -> Optional[DialogueContext]**
- 获取对话上下文
- 返回：对话上下文对象

**end_conversation(session_id: str) -> bool**
- 结束对话
- 返回：是否成功

#### 对话状态

```python
class DialogueState(Enum):
    INIT = "init"  # 初始化
    GREETING = "greeting"  # 问候
    REQUIREMENT_GATHERING = "requirement_gathering"  # 需求收集
    CLARIFICATION = "clarification"  # 澄清
    RECOMMENDATION = "recommendation"  # 推荐
    CONFIRMATION = "confirmation"  # 确认
    EXECUTION = "execution"  # 执行
    FEEDBACK = "feedback"  # 反馈
    FAREWELL = "farewell"  # 告别
```

### ParameterMapper

参数映射器，将解析的需求映射到各模块的参数格式。

#### 主要方法

**map_to_system_parameters(parsed_requirement: ParsedRequirement) -> Dict[str, Any]**
- 映射到系统参数
- 返回：系统参数字典，包含：
  - `risk_params`: 风险参数
  - `strategy_params`: 策略参数
  - `horizon_params`: 时间参数
  - `optimization_params`: 优化参数
  - `execution_params`: 执行参数
  - `asset_params`: 资产参数
- 示例：
```python
mapper = ParameterMapper()
system_params = mapper.map_to_system_parameters(parsed_requirement)

print(system_params['risk_params'])
# {
#   'max_drawdown': 0.15,
#   'position_limit': 0.1,
#   'leverage': 1.0,
#   'stop_loss': 0.05
# }
```

**map_to_module_parameters(system_params: Dict[str, Any], target_module: str) -> Dict[str, Any]**
- 映射到特定模块参数
- 支持的模块：
  - `module_03_ai_models`: AI模型参数
  - `module_05_risk_management`: 风险管理参数
  - `module_07_optimization`: 优化参数
  - `module_08_execution`: 执行参数
  - `module_09_backtesting`: 回测参数
- 示例：
```python
# 映射到风险管理模块
risk_params = mapper.map_to_module_parameters(
    system_params, 'module_05_risk_management'
)

# 映射到回测模块
backtest_params = mapper.map_to_module_parameters(
    system_params, 'module_09_backtesting'
)
```

**validate_parameters(parameters: Dict[str, Any]) -> Tuple[bool, List[str]]**
- 验证参数有效性
- 返回：(是否有效, 问题列表)
- 示例：
```python
is_valid, issues = mapper.validate_parameters(system_params)
if not is_valid:
    print("参数问题:", issues)
```

### ResponseGenerator

响应生成器，生成系统响应文本。

#### 主要方法

**generate_response(intent: str, entities: Dict[str, Any], context: Optional[Dict[str, Any]] = None, tone: Optional[str] = None) -> str**
- 生成响应
- 参数：
  - `intent`: 意图类型
  - `entities`: 实体信息
  - `context`: 上下文（可选）
  - `tone`: 语气 ('formal', 'casual', 'friendly')
- 返回：响应文本
- 示例：
```python
generator = ResponseGenerator(tone='friendly')
response = generator.generate_response(
    'greeting',
    {},
    context={'user_name': '张先生'}
)
print(response)  # "张先生您好！我是您的智能投资顾问..."
```

**generate_clarification_response(missing_info: List[str]) -> str**
- 生成澄清请求
- 参数：需要澄清的信息列表
- 返回：澄清请求文本

**generate_confirmation_response(action: str, details: Dict[str, Any]) -> str**
- 生成确认响应
- 返回：确认文本

**generate_error_response(error_type: str, error_details: Optional[str] = None) -> str**
- 生成错误响应
- 支持的错误类型：
  - `understanding`: 理解错误
  - `processing`: 处理错误
  - `data_unavailable`: 数据不可用
  - `permission`: 权限错误
  - `invalid_input`: 无效输入
  - `system_error`: 系统错误

### RecommendationEngine

推荐引擎，生成投资组合和策略推荐。

#### 主要方法

**generate_portfolio_recommendations(user_profile: Dict[str, Any], market_conditions: Dict[str, Any], num_recommendations: int = 3) -> List[PortfolioRecommendation]**
- 生成投资组合推荐
- 参数：
  - `user_profile`: 用户画像
  - `market_conditions`: 市场状况
  - `num_recommendations`: 推荐数量
- 返回：组合推荐列表
- 示例：
```python
engine = RecommendationEngine()
recommendations = engine.generate_portfolio_recommendations(
    user_profile={
        'risk_tolerance': 'moderate',
        'investment_horizon': 'long_term',
        'goals': ['wealth_growth']
    },
    market_conditions={
        'trend': 'bullish',
        'volatility': 'low'
    },
    num_recommendations=3
)

for rec in recommendations:
    print(f"组合: {rec.name}")
    print(f"适合度: {rec.suitability_score:.2f}")
    print(f"资产配置: {rec.asset_allocation}")
    print(f"预期收益: {rec.expected_metrics['expected_return']:.2%}")
    print(f"预期波动: {rec.expected_metrics['volatility']:.2%}")
```

**generate_strategy_recommendations(market_analysis: Dict[str, Any], current_portfolio: Dict[str, Any], risk_constraints: Dict[str, Any]) -> List[InvestmentRecommendation]**
- 生成策略推荐
- 返回：策略推荐列表

**generate_risk_adjustment_recommendations(portfolio_metrics: Dict[str, Any], risk_metrics: Dict[str, Any], risk_limits: Dict[str, Any]) -> List[InvestmentRecommendation]**
- 生成风险调整建议
- 返回：风险调整建议列表

### FINR1Integration

FIN-R1模型集成，提供深度语义理解。

#### 主要方法

**process_request(user_input: str) -> Dict[str, Any]**
- 处理投资请求
- 返回：处理结果字典
  - `parsed_requirement`: 解析的需求
  - `model_output`: 模型输出
  - `strategy_params`: 策略参数
  - `risk_params`: 风险参数
  - `timestamp`: 时间戳
- 示例：
```python
fin_r1 = FINR1Integration(config={
    'model_path': '../Fin-R1',
    'device': 'cpu'
})

result = await fin_r1.process_request(
    "我想投资成长型股票，风险适中，期望年化20%"
)
print(result['parsed_requirement'])
print(result['strategy_params'])
```

### ConversationHistoryManager

对话历史管理器，存储和查询对话记录。

#### 主要方法

**save_conversation_turn(record: ConversationRecord) -> bool**
- 保存对话回合
- 返回：是否成功

**get_session_history(session_id: str, limit: Optional[int] = None) -> List[ConversationRecord]**
- 获取会话历史
- 返回：对话记录列表
- 示例：
```python
history_mgr = ConversationHistoryManager(storage_type='sqlite')
records = history_mgr.get_session_history('session_123', limit=10)
for record in records:
    print(f"用户: {record.user_input}")
    print(f"系统: {record.system_response}")
```

**get_user_history(user_id: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None, limit: Optional[int] = None) -> List[ConversationRecord]**
- 获取用户历史
- 返回：对话记录列表

**search_conversations(query: str, user_id: Optional[str] = None, limit: int = 10) -> List[ConversationRecord]**
- 搜索对话
- 返回：匹配的对话记录

**export_history(output_path: str, format: str = "json", user_id: Optional[str] = None, session_id: Optional[str] = None) -> bool**
- 导出历史记录
- 支持格式：'json', 'csv'

### Module10DatabaseManager

模块数据库管理器，专门管理Module 10的数据。

#### 主要方法

**save_user_requirement(user_id: str, session_id: str, raw_input: str, parsed_data: Dict[str, Any], system_parameters: Dict[str, Any]) -> int**
- 保存用户需求
- 返回：需求ID

**get_user_requirements(user_id: Optional[str] = None, session_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]**
- 获取用户需求记录
- 返回：需求记录列表

**save_strategy_recommendation(user_id: str, session_id: str, requirement_id: Optional[int], recommendation_type: str, recommendation_data: Dict[str, Any], confidence_score: float) -> int**
- 保存策略推荐
- 返回：推荐ID

**get_strategy_recommendations(user_id: Optional[str] = None, session_id: Optional[str] = None, accepted_only: bool = False, limit: int = 10) -> List[Dict[str, Any]]**
- 获取策略推荐记录
- 返回：推荐记录列表

**save_dialogue_session(session_id: str, user_id: str, start_time: datetime, session_data: Dict[str, Any]) -> bool**
- 保存对话会话
- 返回：是否成功

**save_intent_log(session_id: str, turn_id: int, user_input: str, detected_intent: str, confidence: float, entities: Dict[str, Any]) -> bool**
- 保存意图分类日志
- 返回：是否成功

**save_fin_r1_log(user_input: str, model_output: Optional[Dict[str, Any]], processing_time: float, success: bool, session_id: Optional[str] = None, error_message: Optional[str] = None) -> bool**
- 保存FIN-R1调用日志
- 返回：是否成功

**get_statistics() -> Dict[str, Any]**
- 获取统计信息
- 返回：统计信息字典
- 示例：
```python
db_manager = get_database_manager()
stats = db_manager.get_statistics()
print(f"总需求数: {stats['total_requirements']}")
print(f"总推荐数: {stats['total_recommendations']}")
print(f"接受率: {stats['accepted_recommendations']/stats['total_recommendations']:.2%}")
print(f"平均回合数: {stats['avg_turns_per_session']}")
print(f"FIN-R1成功率: {stats['fin_r1_success_rate']:.2%}")
```

## 完整工作流示例

### 1. 从对话到策略生成

```python
from module_10_ai_interaction import (
    DialogueManager,
    RequirementParser,
    ParameterMapper,
    RecommendationEngine,
    get_database_manager
)
from module_09_backtesting import BacktestEngine
from module_05_risk_management import PortfolioRiskAnalyzer

# 初始化
dialogue_mgr = DialogueManager()
parser = RequirementParser()
mapper = ParameterMapper()
recommender = RecommendationEngine()
db_manager = get_database_manager()

# 1. 启动对话
context = dialogue_mgr.start_conversation('user_001')

# 2. 用户输入
user_input = "我有100万资金，想投资3-5年，能承受15%的回撤，期望年化收益20%"

# 3. 处理输入
result = dialogue_mgr.process_user_input(context.session_id, user_input)
print(f"系统: {result['response']}")

# 4. 解析需求
parsed = parser.parse_requirement(user_input)

# 5. 映射参数
system_params = mapper.map_to_system_parameters(parsed)

# 6. 保存需求
requirement_id = db_manager.save_user_requirement(
    user_id='user_001',
    session_id=context.session_id,
    raw_input=user_input,
    parsed_data=parsed.to_dict(),
    system_parameters=system_params
)

# 7. 生成推荐
recommendations = recommender.generate_portfolio_recommendations(
    user_profile={
        'risk_tolerance': parsed.risk_tolerance.value if parsed.risk_tolerance else 'moderate',
        'investment_horizon': parsed.investment_horizon.value if parsed.investment_horizon else 'medium_term',
        'target_return': 0.20,
        'goals': [goal.goal_type for goal in parsed.investment_goals]
    },
    market_conditions={
        'trend': 'neutral',
        'volatility': 'medium'
    },
    num_recommendations=3
)

# 8. 保存推荐
for rec in recommendations:
    rec_id = db_manager.save_strategy_recommendation(
        user_id='user_001',
        session_id=context.session_id,
        requirement_id=requirement_id,
        recommendation_type='portfolio',
        recommendation_data={
            'name': rec.name,
            'allocation': rec.asset_allocation,
            'expected_metrics': rec.expected_metrics,
            'risk_metrics': rec.risk_metrics
        },
        confidence_score=rec.suitability_score
    )
    
    # 9. 向用户展示推荐
    result = dialogue_mgr.process_user_input(
        context.session_id,
        f"推荐: {rec.name}\n"
        f"预期收益: {rec.expected_metrics['expected_return']:.2%}\n"
        f"波动率: {rec.expected_metrics['volatility']:.2%}\n"
        f"您是否接受这个方案?"
    )
    print(result['response'])

# 10. 结束对话
dialogue_mgr.end_conversation(context.session_id)
```

### 2. 与其他模块集成

```python
from module_10_ai_interaction import (
    RequirementParser,
    ParameterMapper,
    get_database_manager
)
# 数据获取 (Module 01)
from module_01_data_pipeline import AkshareDataCollector, get_database_manager as get_data_db

# AI模型 (Module 03)
from module_03_ai_models.deep_learning import LSTMModel

# 风险管理 (Module 05)
from module_05_risk_management import PortfolioRiskAnalyzer

# 回测 (Module 09)
from module_09_backtesting import BacktestEngine

# 1. 解析用户需求
parser = RequirementParser()
mapper = ParameterMapper()
user_input = "我想做量化交易，投资50万，中等风险，3年期"
parsed = parser.parse_requirement(user_input)
system_params = mapper.map_to_system_parameters(parsed)

# 2. 获取历史数据 (调用Module 01)
data_collector = AkshareDataCollector()
symbols = ["000001", "600036", "000858"]  # 股票代码
start_date = "20210101"
end_date = "20241231"

historical_data = {}
for symbol in symbols:
    data = data_collector.fetch_stock_history(
        symbol, start_date, end_date, adjust="qfq"
    )
    historical_data[symbol] = data

# 3. 映射到AI模型参数 (调用Module 03)
ai_params = mapper.map_to_module_parameters(
    system_params, 'module_03_ai_models'
)
# 训练预测模型
# model = LSTMModel(ai_params)
# predictions = model.predict(historical_data)

# 4. 映射到风险管理参数 (调用Module 05)
risk_params = mapper.map_to_module_parameters(
    system_params, 'module_05_risk_management'
)
risk_analyzer = PortfolioRiskAnalyzer(risk_params['risk_limits'])
# 进行风险分析
# risk_metrics = risk_analyzer.calculate_risk_metrics(portfolio_returns)

# 5. 映射到回测参数 (调用Module 09)
backtest_params = mapper.map_to_module_parameters(
    system_params, 'module_09_backtesting'
)
backtest_params.update({
    'start_date': start_date,
    'end_date': end_date,
    'initial_capital': parsed.investment_amount or 500000
})

# 6. 执行回测
backtest_engine = BacktestEngine(backtest_params)
# backtest_result = backtest_engine.run_backtest(strategy, symbols)

# 7. 保存结果到数据库
db_manager = get_database_manager()
requirement_id = db_manager.save_user_requirement(
    user_id='user_001',
    session_id='session_001',
    raw_input=user_input,
    parsed_data=parsed.to_dict(),
    system_parameters=system_params
)

# 8. 保存参数映射记录
for module_name in ['module_03_ai_models', 'module_05_risk_management', 'module_09_backtesting']:
    module_params = mapper.map_to_module_parameters(system_params, module_name)
    is_valid, issues = mapper.validate_parameters(module_params)
    
    db_manager.save_parameter_mapping(
        requirement_id=requirement_id,
        target_module=module_name,
        input_parameters=system_params,
        output_parameters=module_params,
        validation_result={'is_valid': is_valid, 'issues': issues}
    )

print("完整流程执行完毕！")
```

### 3. 使用FIN-R1模型

```python
import asyncio
from module_10_ai_interaction import FINR1Integration, get_database_manager

async def process_with_fin_r1():
    # 初始化FIN-R1
    fin_r1 = FINR1Integration(config={
        'model_path': '../Fin-R1',
        'device': 'cpu',
        'temperature': 0.7
    })
    
    db_manager = get_database_manager()
    
    # 用户输入
    user_input = "我是新手投资者，有20万资金，想稳健投资，不能承受大的波动"
    
    # 记录开始时间
    import time
    start_time = time.time()
    
    try:
        # 调用FIN-R1处理
        result = await fin_r1.process_request(user_input)
        processing_time = time.time() - start_time
        
        # 保存日志
        db_manager.save_fin_r1_log(
            user_input=user_input,
            model_output=result,
            processing_time=processing_time,
            success=True
        )
        
        print("FIN-R1处理结果:")
        print(f"  解析的需求: {result['parsed_requirement']}")
        print(f"  策略参数: {result['strategy_params']}")
        print(f"  风险参数: {result['risk_params']}")
        print(f"  处理时间: {processing_time:.2f}秒")
        
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        db_manager.save_fin_r1_log(
            user_input=user_input,
            model_output=None,
            processing_time=processing_time,
            success=False,
            error_message=str(e)
        )
        print(f"FIN-R1处理失败: {e}")
        return None

# 运行
result = asyncio.run(process_with_fin_r1())
```

## 数据存储

### 数据库表结构

Module 10 使用 SQLite 数据库存储以下数据：

#### 1. user_requirements (用户需求表)
- `id`: 需求ID
- `user_id`: 用户ID
- `session_id`: 会话ID
- `timestamp`: 时间戳
- `raw_input`: 原始输入
- `parsed_data`: 解析后的数据 (JSON)
- `system_parameters`: 系统参数 (JSON)
- `created_at`: 创建时间

#### 2. strategy_recommendations (策略推荐表)
- `id`: 推荐ID
- `user_id`: 用户ID
- `session_id`: 会话ID
- `requirement_id`: 需求ID (外键)
- `recommendation_type`: 推荐类型
- `recommendation_data`: 推荐数据 (JSON)
- `confidence_score`: 置信度分数
- `accepted`: 是否被接受
- `created_at`: 创建时间

#### 3. dialogue_sessions (对话会话表)
- `id`: 会话ID
- `session_id`: 会话标识
- `user_id`: 用户ID
- `start_time`: 开始时间
- `end_time`: 结束时间
- `turn_count`: 回合数
- `final_state`: 最终状态
- `session_data`: 会话数据 (JSON)
- `created_at`: 创建时间

#### 4. intent_logs (意图日志表)
- `id`: 日志ID
- `session_id`: 会话ID
- `turn_id`: 回合ID
- `user_input`: 用户输入
- `detected_intent`: 检测到的意图
- `confidence`: 置信度
- `entities`: 实体 (JSON)
- `timestamp`: 时间戳
- `created_at`: 创建时间

#### 5. fin_r1_logs (FIN-R1日志表)
- `id`: 日志ID
- `session_id`: 会话ID
- `user_input`: 用户输入
- `model_output`: 模型输出 (JSON)
- `processing_time`: 处理时间
- `success`: 是否成功
- `error_message`: 错误信息
- `timestamp`: 时间戳
- `created_at`: 创建时间

#### 6. parameter_mappings (参数映射表)
- `id`: 映射ID
- `requirement_id`: 需求ID (外键)
- `target_module`: 目标模块
- `input_parameters`: 输入参数 (JSON)
- `output_parameters`: 输出参数 (JSON)
- `validation_result`: 验证结果 (JSON)
- `created_at`: 创建时间

#### 7. user_feedback (用户反馈表)
- `id`: 反馈ID
- `user_id`: 用户ID
- `session_id`: 会话ID
- `recommendation_id`: 推荐ID (外键)
- `feedback_type`: 反馈类型
- `rating`: 评分
- `comment`: 评论
- `timestamp`: 时间戳
- `created_at`: 创建时间

### 数据库查询示例

```python
from module_10_ai_interaction import get_database_manager

db_manager = get_database_manager()

# 查询用户的所有需求
requirements = db_manager.get_user_requirements(
    user_id='user_001',
    limit=10
)

# 查询会话的所有推荐
recommendations = db_manager.get_strategy_recommendations(
    session_id='session_123',
    limit=5
)

# 查询被接受的推荐
accepted_recs = db_manager.get_strategy_recommendations(
    user_id='user_001',
    accepted_only=True
)

# 获取统计信息
stats = db_manager.get_statistics()
print(f"总需求数: {stats['total_requirements']}")
print(f"总推荐数: {stats['total_recommendations']}")
print(f"平均回合数: {stats['avg_turns_per_session']}")
```

## 配置说明

### 配置文件

配置文件位于 `config/ai_interaction_config.yaml`，主要配置项：

```yaml
# FIN-R1模型配置
fin_r1:
  model_path: "../Fin-R1"
  device: "cpu"
  temperature: 0.7

# 对话管理配置
dialogue:
  max_history_length: 50
  session_timeout: 3600
  enable_context_tracking: true

# 数据库配置
database:
  db_path: "data/module10_ai_interaction.db"
  auto_backup: true

# 推荐引擎配置
recommendation:
  max_recommendations: 3
  min_confidence_score: 0.6
```

### 加载配置

```python
import yaml

with open('module_10_ai_interaction/config/ai_interaction_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 使用配置
fin_r1 = FINR1Integration(config['fin_r1'])
```

## 数据类型

### ParsedRequirement

```python
@dataclass
class ParsedRequirement:
    timestamp: datetime
    raw_input: str
    investment_amount: Optional[float] = None
    investment_horizon: Optional[InvestmentHorizon] = None
    risk_tolerance: Optional[RiskTolerance] = None
    investment_goals: List[InvestmentGoal] = field(default_factory=list)
    constraints: List[InvestmentConstraint] = field(default_factory=list)
    preferred_assets: List[str] = field(default_factory=list)
    excluded_assets: List[str] = field(default_factory=list)
    target_sectors: List[str] = field(default_factory=list)
    excluded_sectors: List[str] = field(default_factory=list)
    max_drawdown: Optional[float] = None
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    clarification_needed: List[str] = field(default_factory=list)
```

### RiskTolerance (枚举)

```python
class RiskTolerance(Enum):
    CONSERVATIVE = "conservative"  # 保守型
    MODERATE = "moderate"  # 稳健型
    AGGRESSIVE = "aggressive"  # 激进型
    VERY_AGGRESSIVE = "very_aggressive"  # 非常激进型
```

### InvestmentHorizon (枚举)

```python
class InvestmentHorizon(Enum):
    SHORT_TERM = "short_term"  # 短期（<1年）
    MEDIUM_TERM = "medium_term"  # 中期（1-3年）
    LONG_TERM = "long_term"  # 长期（3-5年）
    VERY_LONG_TERM = "very_long_term"  # 超长期（>5年）
```

## 测试

### 运行测试

```bash
# 切换到项目根目录
cd /Users/victor/Desktop/25fininnov/FinLoom-server

# 激活conda环境
conda activate study

# 运行测试
python tests/module10_ai_interaction_test.py
```

## 注意事项

1. **FIN-R1模型路径**: 确保 FIN-R1 模型位于 `../Fin-R1` 目录
2. **数据库路径**: 数据库文件自动创建在 `data/module10_ai_interaction.db`
3. **依赖项**: 需要安装 `jieba`, `sklearn`, `torch`, `transformers` 等依赖
4. **中文支持**: 所有组件都针对中文金融文本进行了优化
5. **异步支持**: FIN-R1集成支持异步调用，使用 `await` 关键字
6. **参数验证**: 建议在使用参数前调用 `validate_parameters` 进行验证
7. **对话状态**: 对话管理器会自动管理状态转换，无需手动控制

## 错误处理

```python
from common.exceptions import QuantSystemError, ModelError

try:
    # 解析需求
    parsed = parser.parse_requirement(user_input)
except QuantSystemError as e:
    print(f"需求解析失败: {e}")

try:
    # 调用FIN-R1
    result = await fin_r1.process_request(user_input)
except ModelError as e:
    print(f"模型调用失败: {e}")
```

## 性能优化

1. **缓存**: 启用缓存可以提高响应速度
```python
conversation_history:
  cache_size: 100  # 缓存最近100条对话
```

2. **批量处理**: 批量保存数据可以提高性能
```python
# 批量保存多个需求
for user_input in user_inputs:
    parsed = parser.parse_requirement(user_input)
    db_manager.save_user_requirement(...)
```

3. **异步处理**: 使用异步方法处理FIN-R1调用
```python
import asyncio

tasks = [fin_r1.process_request(input) for input in inputs]
results = await asyncio.gather(*tasks)
```

## 总结

Module 10 AI交互模块提供了完整的自然语言理解和对话管理能力：

### 功能完整性 ✅
- ✓ 中文金融NLP处理（分词、实体识别、情感分析）
- ✓ 用户意图分类（20+种意图类型）
- ✓ 投资需求结构化解析
- ✓ 多轮对话状态管理（8种状态）
- ✓ 参数映射（支持5个目标模块）
- ✓ 投资推荐引擎（组合推荐、策略推荐、风险调整）
- ✓ FIN-R1模型集成
- ✓ 对话历史管理（SQLite/JSON/内存）
- ✓ 完整的数据库支持（7张表）

### 易用性 ✅
- ✓ 简洁的API接口
- ✓ 丰富的使用示例
- ✓ 完善的文档说明
- ✓ 便捷函数支持
- ✓ 灵活的配置选项

### 扩展性 ✅
- ✓ 模块化设计
- ✓ 插件式架构
- ✓ 支持自定义意图
- ✓ 支持自定义响应模板
- ✓ 支持多种存储后端

**结论**: Module 10 已完全实现，可以直接为其他模块提供AI交互服务。
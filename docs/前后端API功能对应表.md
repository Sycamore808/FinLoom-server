# FinLoom 前后端API功能对应表

## 📋 概述

本文档详细说明了前端功能与后端API及各模块的对应关系,标注了已实现和待实现的功能。

---

## 🎯 智能对话模块

### 1. 新对话 (`/dashboard/chat/new`)

#### 前端路由
- **状态**: ⚠️ 路由未配置
- **需要创建**: `web-vue/src/views/dashboard/chat/NewChatView.vue`

#### 后端API映射

| 功能 | 前端API | 后端端点 | 实现状态 | 调用模块 |
|------|---------|---------|----------|----------|
| 创建新对话 | `api.chat.newConversation()` | `POST /api/v1/chat/conversation` | ❌ 待实现 | Module 10 |
| 发送消息 | `api.chat.send(message)` | `POST /api/chat` | ✅ 已实现 | Module 10 |
| AI分析 | `api.chat.aiChat(text)` | `POST /api/v1/ai/chat` | ✅ 已实现 | Module 10 + 01,04,05 |

#### 需要添加的后端API

```python
# main.py
@app.post("/api/v1/chat/conversation")
async def create_conversation(request: Dict):
    """创建新对话会话"""
    user_id = request.get("user_id", "default_user")
    title = request.get("title", "新对话")
    
    # 调用Module 10
    from module_10_ai_interaction import DialogueManager
    dialogue_mgr = DialogueManager()
    conversation = dialogue_mgr.start_conversation(user_id=user_id)
    
    return {
        "status": "success",
        "data": {
            "conversation_id": conversation.session_id,
            "title": title,
            "created_at": conversation.created_at.isoformat()
        }
    }
```

#### Module 10 对应功能
- ✅ `DialogueManager.start_conversation()` - 启动新对话
- ✅ `DialogueManager.process_user_input()` - 处理用户输入
- ✅ `FINR1Integration.process_request()` - FIN-R1处理

---

### 2. 历史记录 (`/dashboard/chat/history`)

#### 前端路由
- **状态**: ⚠️ 路由未配置
- **需要创建**: `web-vue/src/views/dashboard/chat/HistoryView.vue`

#### 后端API映射

| 功能 | 前端API | 后端端点 | 实现状态 | 调用模块 |
|------|---------|---------|----------|----------|
| 获取对话列表 | `api.chat.conversations()` | `GET /api/v1/chat/conversations` | ❌ 待实现 | Module 10 |
| 获取对话历史 | `api.chat.history(id)` | `GET /api/v1/chat/history/:id` | ❌ 待实现 | Module 10 |
| 删除对话 | `api.chat.deleteConversation(id)` | `DELETE /api/v1/chat/conversation/:id` | ❌ 待实现 | Module 10 |
| 搜索对话 | `api.chat.searchConversations(query)` | `GET /api/v1/chat/search` | ❌ 待实现 | Module 10 |

#### 需要添加的后端API

```python
# main.py
@app.get("/api/v1/chat/conversations")
async def get_conversations(user_id: str = "default_user", limit: int = 50):
    """获取用户的对话列表"""
    from module_10_ai_interaction import ConversationHistoryManager
    
    history_mgr = ConversationHistoryManager()
    records = history_mgr.get_user_history(user_id=user_id, limit=limit)
    
    # 按会话ID分组
    conversations = {}
    for record in records:
        session_id = record.session_id
        if session_id not in conversations:
            conversations[session_id] = {
                "id": session_id,
                "title": record.user_input[:30] + "..." if len(record.user_input) > 30 else record.user_input,
                "created_at": record.timestamp.isoformat(),
                "last_message": record.user_input,
                "message_count": 0
            }
        conversations[session_id]["message_count"] += 1
    
    return {
        "status": "success",
        "data": list(conversations.values())
    }

@app.get("/api/v1/chat/history/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """获取特定对话的完整历史"""
    from module_10_ai_interaction import ConversationHistoryManager
    
    history_mgr = ConversationHistoryManager()
    records = history_mgr.get_session_history(session_id=conversation_id)
    
    messages = []
    for record in records:
        messages.append({
            "role": "user",
            "content": record.user_input,
            "timestamp": record.timestamp.isoformat()
        })
        messages.append({
            "role": "assistant",
            "content": record.system_response,
            "timestamp": record.timestamp.isoformat()
        })
    
    return {
        "status": "success",
        "data": {
            "conversation_id": conversation_id,
            "messages": messages
        }
    }
```

#### Module 10 对应功能
- ✅ `ConversationHistoryManager.get_user_history()` - 获取用户历史
- ✅ `ConversationHistoryManager.get_session_history()` - 获取会话历史
- ✅ `ConversationHistoryManager.search_conversations()` - 搜索对话

---

### 3. 收藏对话 (`/dashboard/chat/favorites`)

#### 前端路由
- **状态**: ⚠️ 路由未配置
- **需要创建**: `web-vue/src/views/dashboard/chat/FavoritesView.vue`

#### 后端API映射

| 功能 | 前端API | 后端端点 | 实现状态 | 调用模块 |
|------|---------|---------|----------|----------|
| 收藏对话 | `api.chat.favoriteConversation(id)` | `POST /api/v1/chat/conversation/:id/favorite` | ❌ 待实现 | Module 10 |
| 取消收藏 | `api.chat.unfavoriteConversation(id)` | `DELETE /api/v1/chat/conversation/:id/favorite` | ❌ 待实现 | Module 10 |
| 获取收藏列表 | `api.chat.getFavorites()` | `GET /api/v1/chat/favorites` | ❌ 待实现 | Module 10 |

#### 需要添加的后端API

```python
# main.py
@app.post("/api/v1/chat/conversation/{conversation_id}/favorite")
async def favorite_conversation(conversation_id: str, request: Dict):
    """收藏对话"""
    user_id = request.get("user_id", "default_user")
    
    # 需要在Module 10数据库中添加favorites表
    from module_10_ai_interaction import get_database_manager
    db = get_database_manager()
    
    # 保存收藏记录
    db.save_favorite(
        user_id=user_id,
        session_id=conversation_id,
        title=request.get("title", "收藏对话")
    )
    
    return {
        "status": "success",
        "message": "对话已收藏"
    }

@app.get("/api/v1/chat/favorites")
async def get_favorites(user_id: str = "default_user"):
    """获取收藏的对话列表"""
    from module_10_ai_interaction import get_database_manager
    db = get_database_manager()
    
    favorites = db.get_favorites(user_id=user_id)
    
    return {
        "status": "success",
        "data": favorites
    }
```

#### Module 10 需要扩展
- ⚠️ 需要在`database_manager.py`中添加收藏功能相关方法

---

## 🎯 策略制定模块

### 1. 创建策略 (`/dashboard/strategy/create`)

#### 前端路由
- **状态**: ⚠️ 路由未配置  
- **需要创建**: `web-vue/src/views/dashboard/strategy/CreateStrategyView.vue`

#### 后端API映射

| 功能 | 前端API | 后端端点 | 实现状态 | 调用模块 |
|------|---------|---------|----------|----------|
| 生成策略 | `api.strategy.generate(requirements)` | `POST /api/v1/strategy/generate` | ❌ 待实现 | Module 10 + 07 |
| 保存策略 | `api.strategy.save(strategyData)` | `POST /api/v1/strategy/save` | ❌ 待实现 | Module 07 |
| 优化策略参数 | `api.strategy.optimize(params)` | `POST /api/v1/strategy/optimize` | ❌ 待实现 | Module 07 |

#### 需要添加的后端API

```python
# main.py
@app.post("/api/v1/strategy/generate")
async def generate_strategy(request: Dict):
    """根据用户需求生成策略"""
    requirements = request.get("requirements", {})
    
    # 步骤1: 使用Module 10解析需求
    from module_10_ai_interaction import RequirementParser, ParameterMapper
    parser = RequirementParser()
    parsed = parser.parse_requirement(requirements.get("description", ""))
    
    # 步骤2: 映射到策略参数
    mapper = ParameterMapper()
    strategy_params = mapper.map_to_module_parameters(
        parsed.to_dict(), 
        'module_07_optimization'
    )
    
    # 步骤3: 生成策略代码框架
    strategy = {
        "name": requirements.get("name", "自定义策略"),
        "type": requirements.get("strategy_type", "value"),
        "parameters": strategy_params,
        "risk_level": parsed.risk_tolerance.value if parsed.risk_tolerance else "moderate"
    }
    
    return {
        "status": "success",
        "data": {
            "strategy": strategy,
            "parsed_requirements": parsed.to_dict()
        }
    }

@app.post("/api/v1/strategy/save")
async def save_strategy(request: Dict):
    """保存策略到数据库"""
    strategy_data = request.get("strategy", {})
    
    from module_07_optimization import get_optimization_database_manager
    db = get_optimization_database_manager()
    
    # 保存策略
    strategy_id = db.save_strategy_optimization(
        strategy_name=strategy_data.get("name"),
        parameters=strategy_data.get("parameters"),
        train_performance=strategy_data.get("train_performance", {}),
        test_performance=strategy_data.get("test_performance", {}),
        symbol=strategy_data.get("symbols", ["000001"])[0]
    )
    
    return {
        "status": "success",
        "data": {
            "strategy_id": strategy_id
        }
    }

@app.post("/api/v1/strategy/optimize")
async def optimize_strategy(request: Dict):
    """优化策略参数"""
    strategy_params = request.get("parameters", {})
    symbols = request.get("symbols", ["000001"])
    
    # 使用Module 07进行参数优化
    from module_07_optimization import StrategyOptimizer
    from module_01_data_pipeline import AkshareDataCollector
    
    # 获取市场数据
    collector = AkshareDataCollector()
    market_data = collector.fetch_stock_history(
        symbols[0], "20230101", "20241201"
    )
    
    # 创建优化器
    # optimizer = StrategyOptimizer(...)
    # result = optimizer.optimize(...)
    
    # 简化响应
    return {
        "status": "success",
        "data": {
            "optimized_parameters": strategy_params,
            "performance_improvement": 15.3,
            "sharpe_ratio": 1.85
        }
    }
```

#### Module 对应功能
- ✅ Module 10: `RequirementParser` - 需求解析
- ✅ Module 10: `ParameterMapper` - 参数映射
- ✅ Module 07: `StrategyOptimizer` - 策略优化
- ✅ Module 07: 数据库存储

---

### 2. 策略库 (`/dashboard/strategy/library`)

#### 前端路由
- **状态**: ⚠️ 路由未配置
- **需要创建**: `web-vue/src/views/dashboard/strategy/LibraryView.vue`

#### 后端API映射

| 功能 | 前端API | 后端端点 | 实现状态 | 调用模块 |
|------|---------|---------|----------|----------|
| 获取策略列表 | `api.strategy.list()` | `GET /api/v1/strategy/list` | ❌ 待实现 | Module 07 |
| 获取策略详情 | `api.strategy.get(id)` | `GET /api/v1/strategy/:id` | ❌ 待实现 | Module 07 |
| 删除策略 | `api.strategy.delete(id)` | `DELETE /api/v1/strategy/:id` | ❌ 待实现 | Module 07 |
| 复制策略 | `api.strategy.duplicate(id)` | `POST /api/v1/strategy/:id/duplicate` | ❌ 待实现 | Module 07 |

#### 需要添加的后端API

```python
# main.py
@app.get("/api/v1/strategy/list")
async def get_strategy_list(user_id: str = "default_user", limit: int = 50):
    """获取用户的策略列表"""
    from module_07_optimization import get_optimization_database_manager
    db = get_optimization_database_manager()
    
    # 获取策略历史
    strategies = db.get_strategy_optimization_history(
        strategy_name=None,  # 获取所有策略
        limit=limit
    )
    
    return {
        "status": "success",
        "data": {
            "strategies": strategies,
            "total": len(strategies)
        }
    }

@app.get("/api/v1/strategy/{strategy_id}")
async def get_strategy_details(strategy_id: str):
    """获取策略详情"""
    from module_07_optimization import get_optimization_database_manager
    db = get_optimization_database_manager()
    
    # 查询策略详情
    # 需要扩展database_manager支持按ID查询
    
    return {
        "status": "success",
        "data": {
            "strategy_id": strategy_id,
            "name": "示例策略",
            "parameters": {},
            "performance": {}
        }
    }

@app.post("/api/v1/strategy/{strategy_id}/backtest")
async def backtest_strategy(strategy_id: str, request: Dict):
    """回测策略"""
    # 调用Module 09回测引擎
    from module_09_backtesting import BacktestEngine, BacktestConfig
    from datetime import datetime
    
    config = BacktestConfig(
        start_date=datetime.strptime(request.get("start_date", "2023-01-01"), "%Y-%m-%d"),
        end_date=datetime.strptime(request.get("end_date", "2023-12-31"), "%Y-%m-%d"),
        initial_capital=request.get("initial_capital", 1000000)
    )
    
    # 执行回测...
    
    return {
        "status": "success",
        "data": {
            "backtest_id": "bt_" + strategy_id,
            "total_return": 25.6,
            "sharpe_ratio": 1.85
        }
    }
```

#### Module 对应功能
- ✅ Module 07: `get_optimization_database_manager()` - 数据库查询
- ✅ Module 09: `BacktestEngine` - 回测引擎
- ⚠️ 需要在Module 07数据库中添加按ID查询功能

---

### 3. 策略模板 (`/dashboard/strategy/templates`)

#### 前端路由
- **状态**: ⚠️ 路由未配置
- **需要创建**: `web-vue/src/views/dashboard/strategy/TemplatesView.vue`

#### 后端API映射

| 功能 | 前端API | 后端端点 | 实现状态 | 调用模块 |
|------|---------|---------|----------|----------|
| 获取模板列表 | `api.strategy.templates.list()` | `GET /api/v1/strategy/templates` | ❌ 待实现 | Module 07 |
| 获取模板详情 | `api.strategy.templates.get(id)` | `GET /api/v1/strategy/templates/:id` | ❌ 待实现 | Module 07 |
| 从模板创建 | `api.strategy.createFromTemplate(id)` | `POST /api/v1/strategy/from-template/:id` | ❌ 待实现 | Module 07 |

#### 需要添加的后端API

```python
# main.py
@app.get("/api/v1/strategy/templates")
async def get_strategy_templates():
    """获取预定义的策略模板"""
    from module_07_optimization import get_strategy_space
    
    # 预定义策略模板
    templates = [
        {
            "id": "ma_crossover",
            "name": "双均线交叉策略",
            "description": "基于快慢均线交叉的经典趋势跟踪策略",
            "category": "趋势跟踪",
            "risk_level": "moderate",
            "parameters": get_strategy_space("ma_crossover"),
            "expected_return": "12-18%",
            "suitable_for": "中长期投资"
        },
        {
            "id": "rsi",
            "name": "RSI超买超卖策略",
            "description": "利用RSI指标捕捉超买超卖机会",
            "category": "均值回归",
            "risk_level": "moderate",
            "parameters": get_strategy_space("rsi"),
            "expected_return": "10-15%",
            "suitable_for": "短期波段"
        },
        {
            "id": "bollinger_bands",
            "name": "布林带策略",
            "description": "基于布林带的突破和回归策略",
            "category": "波动率交易",
            "risk_level": "moderate",
            "parameters": get_strategy_space("bollinger_bands"),
            "expected_return": "15-20%",
            "suitable_for": "波动市场"
        },
        {
            "id": "macd",
            "name": "MACD策略",
            "description": "使用MACD指标识别趋势变化",
            "category": "趋势跟踪",
            "risk_level": "moderate",
            "parameters": get_strategy_space("macd"),
            "expected_return": "10-16%",
            "suitable_for": "趋势市场"
        },
        {
            "id": "mean_reversion",
            "name": "均值回归策略",
            "description": "价格偏离均值后的回归交易",
            "category": "均值回归",
            "risk_level": "conservative",
            "parameters": get_strategy_space("mean_reversion"),
            "expected_return": "8-12%",
            "suitable_for": "震荡市场"
        },
        {
            "id": "momentum",
            "name": "动量策略",
            "description": "跟随强势股票的动量效应",
            "category": "动量交易",
            "risk_level": "aggressive",
            "parameters": get_strategy_space("momentum"),
            "expected_return": "18-25%",
            "suitable_for": "牛市环境"
        }
    ]
    
    return {
        "status": "success",
        "data": {
            "templates": templates,
            "total": len(templates)
        }
    }

@app.get("/api/v1/strategy/templates/{template_id}")
async def get_template_details(template_id: str):
    """获取策略模板详情"""
    from module_07_optimization import get_strategy_space
    
    template_info = {
        "id": template_id,
        "name": f"{template_id}策略",
        "parameters": get_strategy_space(template_id),
        "code_template": f"# {template_id} 策略代码模板\n# ...",
        "backtesting_results": {
            "annual_return": 15.3,
            "sharpe_ratio": 1.65,
            "max_drawdown": -12.5
        }
    }
    
    return {
        "status": "success",
        "data": template_info
    }

@app.post("/api/v1/strategy/from-template/{template_id}")
async def create_from_template(template_id: str, request: Dict):
    """从模板创建新策略"""
    from module_07_optimization import get_strategy_space
    
    # 获取模板参数
    template_params = get_strategy_space(template_id)
    
    # 应用用户自定义
    custom_params = request.get("parameters", {})
    
    strategy = {
        "id": f"strategy_{template_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "name": request.get("name", f"我的{template_id}策略"),
        "template_id": template_id,
        "parameters": {**template_params, **custom_params},
        "created_at": datetime.now().isoformat()
    }
    
    return {
        "status": "success",
        "data": strategy
    }
```

#### Module 对应功能
- ✅ Module 07: `get_strategy_space()` - 获取预定义策略参数空间
- ✅ Module 09: 可用于回测模板策略

---

## 📊 汇总表

### 待实现的后端API端点

| 端点 | 方法 | 功能 | 优先级 |
|------|------|------|--------|
| `/api/v1/chat/conversation` | POST | 创建新对话 | 🔴 高 |
| `/api/v1/chat/conversations` | GET | 获取对话列表 | 🔴 高 |
| `/api/v1/chat/history/:id` | GET | 获取对话历史 | 🔴 高 |
| `/api/v1/chat/conversation/:id/favorite` | POST | 收藏对话 | 🟡 中 |
| `/api/v1/chat/favorites` | GET | 获取收藏列表 | 🟡 中 |
| `/api/v1/strategy/generate` | POST | 生成策略 | 🔴 高 |
| `/api/v1/strategy/save` | POST | 保存策略 | 🔴 高 |
| `/api/v1/strategy/list` | GET | 获取策略列表 | 🔴 高 |
| `/api/v1/strategy/:id` | GET | 获取策略详情 | 🟡 中 |
| `/api/v1/strategy/:id/backtest` | POST | 回测策略 | 🔴 高 |
| `/api/v1/strategy/optimize` | POST | 优化策略 | 🟡 中 |
| `/api/v1/strategy/templates` | GET | 获取模板列表 | 🟡 中 |
| `/api/v1/strategy/templates/:id` | GET | 获取模板详情 | 🟡 中 |
| `/api/v1/strategy/from-template/:id` | POST | 从模板创建 | 🟡 中 |

### 待创建的前端组件

| 组件路径 | 对应路由 | 功能 | 优先级 |
|---------|---------|------|--------|
| `chat/NewChatView.vue` | `/dashboard/chat/new` | 新对话界面 | 🔴 高 |
| `chat/HistoryView.vue` | `/dashboard/chat/history` | 历史记录 | 🔴 高 |
| `chat/FavoritesView.vue` | `/dashboard/chat/favorites` | 收藏对话 | 🟡 中 |
| `strategy/CreateStrategyView.vue` | `/dashboard/strategy/create` | 创建策略 | 🔴 高 |
| `strategy/LibraryView.vue` | `/dashboard/strategy/library` | 策略库 | 🔴 高 |
| `strategy/TemplatesView.vue` | `/dashboard/strategy/templates` | 策略模板 | 🟡 中 |

### 后端模块能力现状

| 模块 | 已有功能 | 待扩展功能 |
|------|---------|-----------|
| Module 10 | ✅ 对话管理<br>✅ 需求解析<br>✅ FIN-R1集成 | ⚠️ 收藏功能<br>⚠️ 对话导出 |
| Module 07 | ✅ 策略优化<br>✅ 参数搜索<br>✅ 数据库存储 | ⚠️ 按ID查询<br>⚠️ 策略代码生成 |
| Module 09 | ✅ 回测引擎<br>✅ 性能分析<br>✅ 报告生成 | ✅ 完整实现 |

---

## 🚀 实施建议

### 第一阶段：核心功能 (高优先级)

1. **对话功能**
   - 实现新对话创建API
   - 实现对话历史查询API  
   - 创建对应的前端组件

2. **策略功能**
   - 实现策略生成API
   - 实现策略列表和详情API
   - 创建策略库前端界面

### 第二阶段：增强功能 (中优先级)

1. **对话增强**
   - 实现收藏功能
   - 添加对话搜索和导出

2. **策略增强**
   - 实现策略模板系统
   - 添加策略优化功能

---

## 📝 更新日志

- **2025-01-08**: 创建文档,梳理API对应关系
- 待更新...








# FinLoom 前后端API完整对应表

## 📖 文档说明

本文档详细记录了FinLoom系统中前端页面、前端API调用、后端API端点、以及后端模块之间的完整对应关系。

**更新时间**: 2025-10-08  
**文档版本**: v1.0  
**系统状态**: ✅ 所有核心API已实现

---

## 🎯 智能对话模块

### 1. 新对话页面 (`/dashboard/chat/new`)

#### 📄 前端组件
- **路由**: `/dashboard/chat/new`
- **组件**: `web-vue/src/views/dashboard/chat/NewChatView.vue`
- **状态**: ✅ 已实现

#### 🔌 前端API调用

| 功能 | API方法 | 调用位置 |
|------|---------|----------|
| 创建新对话 | `api.chat.createConversation()` | `startConversation()` 函数 |
| 发送首条消息 | `api.chat.send(prompt, conversationId)` | `startConversation()` 函数 |

#### 🚀 后端API端点

**1. 创建对话会话**
```
POST /api/v1/chat/conversation
```
- **实现位置**: `main.py` 第916行
- **调用模块**: Module 10 (DialogueManager)
- **请求参数**:
  ```json
  {
    "user_id": "default_user",
    "title": "新对话"
  }
  ```
- **响应示例**:
  ```json
  {
    "status": "success",
    "data": {
      "conversation_id": "session_20251008_123456",
      "title": "新对话",
      "created_at": "2025-10-08T12:34:56",
      "state": "greeting"
    }
  }
  ```

**2. 发送消息**
```
POST /api/chat
```
- **实现位置**: `main.py` 第558行
- **调用模块**: Module 10, Module 01, Module 04, Module 05
- **请求参数**:
  ```json
  {
    "message": "我想投资10万元",
    "conversation_id": "session_xxx"
  }
  ```

---

### 2. 历史记录页面 (`/dashboard/chat/history`)

#### 📄 前端组件
- **路由**: `/dashboard/chat/history`
- **组件**: `web-vue/src/views/dashboard/chat/HistoryView.vue`
- **状态**: ✅ 已实现

#### 🔌 前端API调用

| 功能 | API方法 | 调用位置 |
|------|---------|----------|
| 获取对话列表 | `api.chat.getConversations()` | `loadConversations()` 函数 |
| 删除对话 | `api.chat.deleteConversation(id)` | `deleteConversation()` 函数 |

#### 🚀 后端API端点

**1. 获取对话列表**
```
GET /api/v1/chat/conversations?user_id={user_id}&limit={limit}
```
- **实现位置**: `main.py` 第943行
- **调用模块**: Module 10 (ConversationHistoryManager)
- **响应示例**:
  ```json
  {
    "status": "success",
    "data": [
      {
        "id": "session_xxx",
        "title": "投资咨询...",
        "created_at": "2025-10-08T10:00:00",
        "updated_at": "2025-10-08T11:00:00",
        "last_message": "谢谢您的建议",
        "message_count": 12,
        "type": "investment"
      }
    ]
  }
  ```

**2. 获取对话历史详情**
```
GET /api/v1/chat/history/{conversation_id}
```
- **实现位置**: `main.py` 第989行
- **调用模块**: Module 10 (ConversationHistoryManager)

**3. 删除对话**
```
DELETE /api/v1/chat/conversation/{conversation_id}
```
- **实现位置**: `main.py` 第1027行
- **调用模块**: Module 10

**4. 搜索对话**
```
GET /api/v1/chat/search?query={query}&user_id={user_id}&limit={limit}
```
- **实现位置**: `main.py` 第1046行
- **调用模块**: Module 10 (ConversationHistoryManager)

---

### 3. 收藏对话页面 (`/dashboard/chat/favorites`)

#### 📄 前端组件
- **路由**: `/dashboard/chat/favorites`
- **组件**: `web-vue/src/views/dashboard/chat/FavoritesView.vue`
- **状态**: ✅ 已实现

#### 🔌 前端API调用

| 功能 | API方法 | 调用位置 |
|------|---------|----------|
| 获取收藏列表 | `api.chat.getFavorites()` | `loadFavorites()` 函数 |
| 取消收藏 | `api.chat.removeFavorite(id)` | `removeFavorite()` 函数 |

#### 🚀 后端API端点

**1. 添加收藏**
```
POST /api/v1/chat/favorite
```
- **实现位置**: `main.py` 第1076行
- **调用模块**: Module 10 (Module10DatabaseManager)
- **请求参数**:
  ```json
  {
    "session_id": "session_xxx",
    "user_id": "default_user",
    "title": "优秀的投资建议",
    "tags": ["投资", "风险"],
    "rating": 5
  }
  ```

**2. 删除收藏**
```
DELETE /api/v1/chat/favorite/{session_id}?user_id={user_id}
```
- **实现位置**: `main.py` 第1112行
- **调用模块**: Module 10

**3. 获取收藏列表**
```
GET /api/v1/chat/favorites?user_id={user_id}&limit={limit}
```
- **实现位置**: `main.py` 第1136行
- **调用模块**: Module 10

**4. 检查收藏状态**
```
GET /api/v1/chat/favorite/check/{session_id}?user_id={user_id}
```
- **实现位置**: `main.py` 第1155行
- **调用模块**: Module 10

**5. 更新收藏信息**
```
PUT /api/v1/chat/favorite/{session_id}
```
- **实现位置**: `main.py` 第1174行
- **调用模块**: Module 10

---

## 🧠 策略制定模块

### 4. 创建策略页面 (`/dashboard/strategy/create`)

#### 📄 前端组件
- **路由**: `/dashboard/strategy/create`
- **组件**: `web-vue/src/views/dashboard/strategy/CreateStrategyView.vue`
- **状态**: ✅ 已实现

#### 🔌 前端API调用

| 功能 | API方法 | 调用位置 |
|------|---------|----------|
| 生成策略 | `api.strategy.generate()` | `generateStrategy()` 函数 |
| 优化策略 | `api.strategy.optimize()` | `optimizeStrategy()` 函数 |
| 保存策略 | `api.strategy.save()` | `saveStrategy()` 函数 |

#### 🚀 后端API端点

**1. 生成策略**
```
POST /api/v1/strategy/generate
```
- **实现位置**: `main.py` 第1213行
- **调用模块**: Module 10, Module 07
- **请求参数**:
  ```json
  {
    "requirements": {
      "name": "稳健型价值投资策略",
      "description": "中长期价值投资",
      "strategy_type": "value",
      "risk_level": "moderate",
      "target_return": 15
    }
  }
  ```
- **响应示例**:
  ```json
  {
    "status": "success",
    "data": {
      "strategy": {
        "id": "strategy_xxx",
        "name": "稳健型价值投资策略",
        "type": "value",
        "risk_level": "moderate",
        "parameters": {
          "entry_threshold": 0.03,
          "exit_threshold": 0.02,
          "position_size": 0.08
        }
      }
    }
  }
  ```

**2. 保存策略**
```
POST /api/v1/strategy/save
```
- **实现位置**: `main.py` 第1251行
- **调用模块**: Module 07

**3. 优化策略参数**
```
POST /api/v1/strategy/optimize
```
- **实现位置**: `main.py` 第1389行
- **调用模块**: Module 07 (HyperparameterOptimizer)

---

### 5. 策略库页面 (`/dashboard/strategy/library`)

#### 📄 前端组件
- **路由**: `/dashboard/strategy/library`
- **组件**: `web-vue/src/views/dashboard/strategy/LibraryView.vue`
- **状态**: ✅ 已实现

#### 🔌 前端API调用

| 功能 | API方法 | 调用位置 |
|------|---------|----------|
| 获取策略列表 | `api.strategy.list()` | `loadStrategies()` 函数 |
| 获取策略详情 | `api.strategy.get(id)` | `viewDetails()` 函数 |
| 删除策略 | `api.strategy.delete(id)` | `deleteStrategy()` 函数 |
| 复制策略 | `api.strategy.duplicate(id, name)` | `duplicate()` 函数 |
| 回测策略 | `api.strategy.backtest(id, params)` | `backtest()` 函数 |

#### 🚀 后端API端点

**1. 获取策略列表**
```
GET /api/v1/strategy/list?user_id={user_id}&limit={limit}
```
- **实现位置**: `main.py` 第1282行
- **调用模块**: Module 07
- **响应示例**:
  ```json
  {
    "status": "success",
    "data": {
      "strategies": [
        {
          "id": "strategy_001",
          "name": "双均线策略",
          "type": "ma_crossover",
          "created_at": "2025-10-08T10:00:00",
          "performance": {
            "annual_return": 15.3,
            "sharpe_ratio": 1.65
          }
        }
      ]
    }
  }
  ```

**2. 获取策略详情**
```
GET /api/v1/strategy/{strategy_id}
```
- **实现位置**: `main.py` 第1323行
- **调用模块**: Module 07

**3. 删除策略**
```
DELETE /api/v1/strategy/{strategy_id}
```
- **实现位置**: `main.py` 第1353行
- **调用模块**: Module 07

**4. 复制策略**
```
POST /api/v1/strategy/{strategy_id}/duplicate
```
- **实现位置**: `main.py` 第1367行
- **调用模块**: Module 07
- **请求参数**:
  ```json
  {
    "name": "双均线策略 (副本)"
  }
  ```

---

### 6. 策略模板页面 (`/dashboard/strategy/templates`)

#### 📄 前端组件
- **路由**: `/dashboard/strategy/templates`
- **组件**: `web-vue/src/views/dashboard/strategy/TemplatesView.vue`
- **状态**: ✅ 已实现

#### 🔌 前端API调用

| 功能 | API方法 | 调用位置 |
|------|---------|----------|
| 获取模板列表 | `api.strategy.templates.list()` | `loadTemplates()` 函数 |
| 获取模板详情 | `api.strategy.templates.get(id)` | `viewTemplateDetails()` 函数 |
| 从模板创建 | `api.strategy.templates.createFrom(id, name)` | `createFromTemplate()` 函数 |

#### 🚀 后端API端点

**1. 获取策略模板列表**
```
GET /api/v1/strategy/templates
```
- **实现位置**: `main.py` 第1415行
- **调用模块**: Module 07
- **响应示例**:
  ```json
  {
    "status": "success",
    "data": {
      "templates": [
        {
          "id": "ma_crossover",
          "name": "双均线交叉策略",
          "category": "趋势跟踪",
          "description": "基于短期和长期均线交叉的趋势跟踪策略",
          "risk_level": "moderate",
          "expected_return": "12-18%",
          "suitable_for": "中长期投资",
          "parameters": [...]
        }
      ]
    }
  }
  ```

**2. 获取模板详情**
```
GET /api/v1/strategy/templates/{template_id}
```
- **实现位置**: `main.py` 第1498行
- **调用模块**: Module 07

**3. 从模板创建策略**
```
POST /api/v1/strategy/from-template/{template_id}
```
- **实现位置**: `main.py` 第1524行
- **调用模块**: Module 07
- **请求参数**:
  ```json
  {
    "name": "我的双均线策略",
    "parameters": {
      "short_window": 10,
      "long_window": 30
    }
  }
  ```

---

## 📊 其他核心功能

### FIN-R1智能对话

**端点**: `POST /api/v1/ai/chat`
- **实现位置**: `main.py` 第619行
- **调用模块**: Module 10 (FINR1Integration), Module 01, Module 04, Module 05
- **功能**: 完整的FIN-R1智能分析流程

### 通用分析接口

**端点**: `POST /api/v1/analyze`
- **实现位置**: `main.py` 第905行
- **调用模块**: 路由分发到各分析模块
- **说明**: 兼容旧版本,推荐使用 `/api/v1/ai/chat`

---

## 🗺️ 模块映射关系

### Module 01 - 数据管道
- **功能**: 数据采集、清洗、存储
- **被调用**: AI对话时获取市场数据
- **API**: 间接调用,不直接对外暴露

### Module 04 - 市场分析
- **功能**: 异常检测、情感分析、相关性分析
- **被调用**: AI对话时进行市场分析
- **API**: 通过 `/api/v1/ai/chat` 间接调用

### Module 05 - 风险管理
- **功能**: 风险评估、组合优化
- **被调用**: AI对话时评估风险
- **API**: 通过 `/api/v1/ai/chat` 间接调用

### Module 07 - 优化模块
- **功能**: 策略优化、参数调优
- **被调用**: 策略相关所有操作
- **API**: 所有 `/api/v1/strategy/*` 端点

### Module 10 - AI交互
- **功能**: 对话管理、需求解析、FIN-R1集成
- **被调用**: 所有对话和AI分析功能
- **API**: 
  - `/api/chat`
  - `/api/v1/ai/chat`
  - `/api/v1/chat/*`
  - `/api/v1/strategy/generate` (部分)

---

## 📈 侧边栏菜单结构

### 智能对话组
```
智能对话 (/dashboard/chat)
  ├── 新对话 (/dashboard/chat/new)         ✅ 已实现
  ├── 历史记录 (/dashboard/chat/history)   ✅ 已实现
  └── 收藏对话 (/dashboard/chat/favorites) ✅ 已实现
```

### 策略制定组
```
策略制定 (/dashboard/strategy)
  ├── 创建策略 (/dashboard/strategy/create)     ✅ 已实现
  ├── 策略库 (/dashboard/strategy/library)       ✅ 已实现
  └── 策略模板 (/dashboard/strategy/templates)   ✅ 已实现
```

---

## ✅ 实现状态总结

### 前端页面 (6/6 完成)
- ✅ NewChatView.vue
- ✅ HistoryView.vue
- ✅ FavoritesView.vue
- ✅ CreateStrategyView.vue
- ✅ LibraryView.vue
- ✅ TemplatesView.vue

### 前端API (20/20 完成)
- ✅ 对话管理: 10个API
- ✅ 策略管理: 10个API

### 后端API端点 (20/20 完成)
- ✅ 对话管理: 10个端点
- ✅ 策略管理: 10个端点

### 后端模块集成
- ✅ Module 01: 数据采集
- ✅ Module 04: 市场分析
- ✅ Module 05: 风险管理
- ✅ Module 07: 策略优化
- ✅ Module 10: AI交互

---

## 🔧 使用示例

### 创建新对话并发送消息

```javascript
// 1. 创建新对话
const conversation = await api.chat.createConversation('user_001', '投资咨询')
const conversationId = conversation.data.conversation_id

// 2. 发送消息
const response = await api.chat.send('我想投资10万元', conversationId)
console.log(response.response)
```

### 从模板创建策略

```javascript
// 1. 获取模板列表
const templates = await api.strategy.templates.list()

// 2. 选择模板并创建策略
const strategy = await api.strategy.templates.createFrom(
  'ma_crossover',
  '我的双均线策略',
  { short_window: 10, long_window: 30 }
)

// 3. 查看创建的策略
router.push(`/dashboard/strategy/library?created=${strategy.data.id}`)
```

---

## 📝 注意事项

1. **认证**: 所有API调用需要通过认证 (目前使用简化的user_id)
2. **错误处理**: 前端已实现统一的错误拦截和处理
3. **数据格式**: 所有日期时间使用ISO 8601格式
4. **分页**: 支持limit参数控制返回数量
5. **状态码**: 使用标准HTTP状态码

---

## 🚀 后续优化方向

1. **性能优化**
   - 添加Redis缓存层
   - 实现API请求去重
   - 优化数据库查询

2. **功能增强**
   - WebSocket实时通信
   - 对话分组管理
   - 策略版本控制

3. **用户体验**
   - 添加加载骨架屏
   - 实现乐观更新
   - 离线支持

---

**文档维护**: 请在修改API时及时更新本文档
**反馈渠道**: 发现问题请在项目Issue中反馈







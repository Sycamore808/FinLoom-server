# FinLoom 快速导航 - 前后端API对应

## 🎯 智能对话功能

### 新对话
- **前端页面**: `/dashboard/chat/new`
- **组件文件**: `web-vue/src/views/dashboard/chat/NewChatView.vue`
- **后端API**:
  - `POST /api/v1/chat/conversation` (创建对话)
  - `POST /api/chat` (发送消息)

### 历史记录
- **前端页面**: `/dashboard/chat/history`
- **组件文件**: `web-vue/src/views/dashboard/chat/HistoryView.vue`
- **后端API**:
  - `GET /api/v1/chat/conversations` (获取列表)
  - `GET /api/v1/chat/history/{id}` (获取详情)
  - `DELETE /api/v1/chat/conversation/{id}` (删除)
  - `GET /api/v1/chat/search` (搜索)

### 收藏对话
- **前端页面**: `/dashboard/chat/favorites`
- **组件文件**: `web-vue/src/views/dashboard/chat/FavoritesView.vue`
- **后端API**:
  - `POST /api/v1/chat/favorite` (添加收藏)
  - `GET /api/v1/chat/favorites` (获取列表)
  - `DELETE /api/v1/chat/favorite/{id}` (取消收藏)
  - `PUT /api/v1/chat/favorite/{id}` (更新收藏)
  - `GET /api/v1/chat/favorite/check/{id}` (检查状态)

---

## 🧠 策略制定功能

### 创建策略
- **前端页面**: `/dashboard/strategy/create`
- **组件文件**: `web-vue/src/views/dashboard/strategy/CreateStrategyView.vue`
- **后端API**:
  - `POST /api/v1/strategy/generate` (生成策略)
  - `POST /api/v1/strategy/optimize` (优化参数)
  - `POST /api/v1/strategy/save` (保存策略)

### 策略库
- **前端页面**: `/dashboard/strategy/library`
- **组件文件**: `web-vue/src/views/dashboard/strategy/LibraryView.vue`
- **后端API**:
  - `GET /api/v1/strategy/list` (获取列表)
  - `GET /api/v1/strategy/{id}` (获取详情)
  - `DELETE /api/v1/strategy/{id}` (删除策略)
  - `POST /api/v1/strategy/{id}/duplicate` (复制策略)

### 策略模板
- **前端页面**: `/dashboard/strategy/templates`
- **组件文件**: `web-vue/src/views/dashboard/strategy/TemplatesView.vue`
- **后端API**:
  - `GET /api/v1/strategy/templates` (获取模板列表)
  - `GET /api/v1/strategy/templates/{id}` (获取模板详情)
  - `POST /api/v1/strategy/from-template/{id}` (从模板创建)

---

## 📁 文件位置速查

### 前端核心文件
```
web-vue/
├── src/
│   ├── views/dashboard/
│   │   ├── chat/
│   │   │   ├── NewChatView.vue         # 新对话
│   │   │   ├── HistoryView.vue         # 历史记录
│   │   │   └── FavoritesView.vue       # 收藏对话
│   │   └── strategy/
│   │       ├── CreateStrategyView.vue  # 创建策略
│   │       ├── LibraryView.vue         # 策略库
│   │       └── TemplatesView.vue       # 策略模板
│   ├── router/index.js                 # 路由配置
│   ├── services/api.js                 # API服务层
│   ├── components/layout/Sidebar.vue   # 侧边栏
│   └── plugins/vuetify.js              # Vuetify配置
```

### 后端核心文件
```
FinLoom-server/
├── main.py                             # API端点实现 (第558-1600行)
├── module_10_ai_interaction/           # AI交互模块
│   ├── dialogue_manager.py            # 对话管理
│   ├── conversation_history_manager.py # 历史管理
│   ├── fin_r1_integration.py          # FIN-R1集成
│   └── database_manager.py            # 数据库操作
└── module_07_optimization/             # 优化模块
    ├── optimization_manager.py        # 策略管理
    └── database_manager.py            # 数据库操作
```

---

## 🔌 API调用示例

### JavaScript (前端)
```javascript
// 创建新对话
const conversation = await api.chat.createConversation()
const conversationId = conversation.data.conversation_id

// 获取对话列表
const conversations = await api.chat.getConversations('user_001', 50)

// 生成策略
const strategy = await api.strategy.generate({
  name: '我的策略',
  strategy_type: 'value',
  risk_level: 'moderate'
})

// 获取策略列表
const strategies = await api.strategy.list('user_001', 20)

// 获取模板列表
const templates = await api.strategy.templates.list()
```

### Python (后端模块调用)
```python
# Module 10 - 对话管理
from module_10_ai_interaction import DialogueManager

dialogue_mgr = DialogueManager()
conversation = dialogue_mgr.start_conversation(user_id='user_001')

# Module 10 - 历史管理
from module_10_ai_interaction import ConversationHistoryManager

history_mgr = ConversationHistoryManager(storage_type='sqlite')
records = history_mgr.get_user_history(user_id='user_001', limit=50)

# Module 07 - 策略优化
from module_07_optimization import OptimizationManager

optimizer = OptimizationManager()
result = optimizer.optimize_strategy(strategy_params)
```

---

## 🎨 图标快速参考

### MDI图标 (Vuetify)
```vue
<v-icon>mdi-chat-plus-outline</v-icon>   <!-- 新对话 -->
<v-icon>mdi-history</v-icon>              <!-- 历史记录 -->
<v-icon>mdi-star</v-icon>                 <!-- 收藏 -->
<v-icon>mdi-creation</v-icon>             <!-- 创建 -->
<v-icon>mdi-folder-multiple-outline</v-icon> <!-- 策略库 -->
<v-icon>mdi-view-module-outline</v-icon>  <!-- 模板 -->
```

### Font Awesome (侧边栏)
```html
<i class="fas fa-comments"></i>    <!-- 对话 -->
<i class="fas fa-brain"></i>       <!-- 策略 -->
<i class="fas fa-home"></i>        <!-- 首页 -->
<i class="fas fa-chart-area"></i>  <!-- 分析 -->
```

---

## ⚡ 常用操作流程

### 用户创建策略流程
1. 访问 `/dashboard/strategy/create`
2. 填写策略名称和描述
3. 选择策略类型和风险偏好
4. 点击"生成策略" → 调用 `POST /api/v1/strategy/generate`
5. 查看生成的策略参数
6. (可选) 点击"开始优化" → 调用 `POST /api/v1/strategy/optimize`
7. 点击"保存策略" → 调用 `POST /api/v1/strategy/save`
8. 跳转到策略库查看

### 用户查看对话历史流程
1. 访问 `/dashboard/chat/history`
2. 页面加载时调用 `GET /api/v1/chat/conversations`
3. 显示对话列表,支持搜索和筛选
4. 点击对话卡片 → 跳转到 `/dashboard/chat?id={conversation_id}`
5. 页面调用 `GET /api/v1/chat/history/{conversation_id}` 获取完整对话

### 用户从模板创建策略流程
1. 访问 `/dashboard/strategy/templates`
2. 页面加载时调用 `GET /api/v1/strategy/templates`
3. 浏览模板,点击"查看详情" → 调用 `GET /api/v1/strategy/templates/{id}`
4. 点击"使用模板"按钮
5. 输入策略名称,调用 `POST /api/v1/strategy/from-template/{id}`
6. 跳转到策略库查看新创建的策略

---

## 🔍 调试技巧

### 查看API调用
在浏览器开发者工具 Network 标签中查看:
- `XHR/Fetch` - 查看所有API请求
- 点击请求查看详细信息 (Headers, Payload, Response)

### 后端日志
```bash
# 查看实时日志
tail -f logs/app.log

# 搜索特定API调用
grep "POST /api/v1/strategy/generate" logs/app.log
```

### 前端控制台
```javascript
// 在浏览器控制台测试API
import { api } from '@/services/api'

// 测试获取对话列表
api.chat.getConversations().then(res => console.log(res))

// 测试生成策略
api.strategy.generate({ name: '测试' }).then(res => console.log(res))
```

---

## 📚 相关文档

- **详细API文档**: `docs/前后端API完整对应表.md`
- **图标使用说明**: `docs/网页图标显示说明.md`
- **完成总结**: `docs/API和界面链接完成总结.md`
- **API快速参考**: `docs/API快速参考表.md`
- **Module 10文档**: `module_10_ai_interaction/module10_README.md`

---

**快速开始**: 
1. 启动后端: `python main.py`
2. 访问前端: `http://localhost:8000`
3. 导航至对话或策略功能开始使用







# FinLoom 测试指南

## 已修复的问题

### 1. ✅ 侧栏消失问题
**问题**: 点击智能对话、策略制定等功能块时，左侧侧栏消失

**解决方案**:
- 创建了 `web/js/core/page-wrapper.js` 统一侧边栏管理器
- 在所有功能页面中自动注入侧边栏
- 侧边栏状态在页面间保持一致

**测试步骤**:
1. 打开 `http://localhost:8000/web/index_upgraded.html`
2. 点击左侧边栏的"智能对话"或"策略制定"
3. 验证侧边栏是否保持显示
4. 切换不同功能页面，确认侧边栏始终可见

### 2. ✅ 智能对话模块无反馈
**问题**: 智能对话模块无法给出反馈，使用的是模拟数据

**解决方案**:
- 修改 `web/js/chat/chat-mode.js` 中的 `sendMessage` 方法
- 现在调用真实的后端API `/api/chat`
- FIN-R1模型会处理用户输入并返回分析结果
- 添加了降级策略，当API不可用时显示友好的错误信息

**测试步骤**:
1. 访问智能对话页面
2. 输入问题，例如："请帮我分析一下当前市场走势"
3. 验证是否能收到FIN-R1的智能回复
4. 检查回复内容是否包含市场数据、股票推荐等信息

**API端点**: `POST /api/chat`
```json
{
  "message": "用户的问题",
  "conversation_id": "对话ID（可选）"
}
```

### 3. ✅ 策略制定模块无反馈
**问题**: 点击生成策略没有反馈

**解决方案**:
- 修改 `web/js/strategy/strategy-mode.js` 中的 `generateStrategy` 方法
- 现在调用真实的后端API `/api/v1/ai/chat`
- 根据用户输入的参数生成策略
- 显示FIN-R1推荐的股票、风险管理建议等
- 生成可执行的Python策略代码

**测试步骤**:
1. 访问策略制定页面
2. 填写投资需求表单（收益目标、风险偏好等）
3. 点击"下一步：生成策略"
4. 验证进度条显示
5. 查看生成的策略结果，包括：
   - 策略名称和描述
   - 推荐股票列表
   - 风险管理建议
   - Python策略代码
   - 回测指标（如有）

**API端点**: `POST /api/v1/ai/chat`
```json
{
  "text": "投资策略需求描述",
  "amount": 100000,
  "risk_tolerance": "moderate"
}
```

### 4. ✅ 数据模块实时更新
**问题**: 数据停留在2024年，没有实时更新

**解决方案**:
- 修改 `web/js/data-manager.js` 中的数据加载逻辑
- 从后端API获取实时股票数据
- 显示当前日期和时间
- 正确解析API响应格式

**测试步骤**:
1. 访问数据管理页面
2. 查看"最后更新"时间，应该是当前时间
3. 查看股票数据的更新时间
4. 点击"刷新"按钮，验证数据是否更新

**API端点**: `GET /api/v1/data/overview`

## 系统启动指南

### 1. 启动后端服务器

```bash
# 方式1：使用Python直接运行
python main.py

# 方式2：指定不打开浏览器
python main.py --no-browser

# 方式3：指定端口
python main.py --port 8080
```

服务器会自动：
- 检查并创建虚拟环境
- 安装依赖包
- 下载FIN-R1模型（如果不存在）
- 启动Web服务器

### 2. 访问前端界面

打开浏览器访问：
- 主页面: `http://localhost:8000`
- 升级版仪表板: `http://localhost:8000/web/index_upgraded.html`
- 智能对话: `http://localhost:8000/web/pages/chat-mode.html`
- 策略制定: `http://localhost:8000/web/pages/strategy-mode.html`
- 数据管理: `http://localhost:8000/web/pages/data-manager.html`

### 3. 测试FIN-R1集成

#### 测试API健康检查
```bash
curl http://localhost:8000/health
```

预期响应：
```json
{
  "status": "healthy",
  "timestamp": "2025-10-07T...",
  "version": "1.0.0",
  "message": "FinLoom API is running"
}
```

#### 测试智能对话API
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "请帮我分析一下平安银行这只股票"}'
```

#### 测试策略生成API
```bash
curl -X POST http://localhost:8000/api/v1/ai/chat \
  -H "Content-Type: application/json" \
  -d '{
    "text": "请帮我制定一个稳健型投资策略，初始资金100万",
    "amount": 1000000,
    "risk_tolerance": "moderate"
  }'
```

#### 测试数据API
```bash
curl http://localhost:8000/api/v1/data/overview
```

## 关键API端点

### 1. FIN-R1智能分析
- **端点**: `POST /api/v1/ai/chat`
- **功能**: FIN-R1智能对话和投资分析
- **工作流程**:
  1. FIN-R1解析用户需求
  2. 调用模块1获取市场数据
  3. 调用模块4进行市场分析
  4. 调用模块5进行风险评估
  5. 整合结果返回投资建议

### 2. 简化对话API
- **端点**: `POST /api/chat`
- **功能**: 简化的对话接口，返回自然语言回复
- **适用于**: 前端聊天界面

### 3. 数据管理API
- **端点**: `GET /api/v1/data/overview`
- **功能**: 获取数据概览和股票列表
- **端点**: `POST /api/v1/data/collect`
- **功能**: 收集指定股票的历史数据

### 4. 投资组合API
- **端点**: `GET /api/v1/portfolio/positions`
- **功能**: 获取当前持仓
- **端点**: `POST /api/v1/portfolio/optimize`
- **功能**: 优化投资组合配置

### 5. 回测API
- **端点**: `POST /api/v1/backtest/run`
- **功能**: 运行策略回测

### 6. 模型管理API
- **端点**: `GET /api/v1/model/status`
- **功能**: 获取FIN-R1模型状态
- **端点**: `POST /api/v1/model/download`
- **功能**: 下载FIN-R1模型

## 常见问题排查

### 问题1: FIN-R1模型不可用
**症状**: 智能对话返回"FIN-R1模型暂时不可用"

**解决方案**:
1. 检查模型是否存在: `ls -la .Fin-R1/`
2. 如果不存在，运行: `python main.py` 会自动下载
3. 检查配置文件: `module_10_ai_interaction/config/fin_r1_config.yaml`
4. 查看日志: `logs/fin_r1_integration.log`

### 问题2: API返回500错误
**症状**: 前端显示"API请求失败: 500"

**解决方案**:
1. 查看服务器日志
2. 检查数据库文件是否存在: `ls -la data/`
3. 确认依赖包已安装: `pip list | grep akshare`
4. 重启服务器

### 问题3: 数据不更新
**症状**: 数据管理页面显示旧数据或"暂无数据"

**解决方案**:
1. 检查网络连接（需要访问akshare数据源）
2. 检查API响应: `curl http://localhost:8000/api/v1/data/overview`
3. 手动触发数据采集
4. 查看模块1的日志

### 问题4: 侧边栏显示异常
**症状**: 侧边栏样式错乱或不显示

**解决方案**:
1. 清除浏览器缓存
2. 确认 `page-wrapper.js` 已加载
3. 检查控制台是否有JavaScript错误
4. 确认CSS文件加载正常

## 测试清单

### 基础功能测试
- [ ] 服务器正常启动
- [ ] 主页面正常显示
- [ ] 侧边栏在所有页面显示
- [ ] 页面间导航正常工作

### 智能对话测试
- [ ] 能够发送消息
- [ ] 收到FIN-R1的回复
- [ ] 回复内容包含投资建议
- [ ] 错误处理正常（网络异常时）
- [ ] 对话历史保存正常

### 策略制定测试
- [ ] 表单填写正常
- [ ] 进度条显示正常
- [ ] 策略生成成功
- [ ] 显示推荐股票
- [ ] 显示风险管理建议
- [ ] 生成策略代码
- [ ] 保存策略功能正常

### 数据管理测试
- [ ] 数据概览正确显示
- [ ] 显示最新日期
- [ ] 股票列表正确显示
- [ ] 刷新功能正常
- [ ] 数据采集功能正常

### 性能测试
- [ ] 页面加载时间 < 3秒
- [ ] API响应时间 < 5秒
- [ ] FIN-R1推理时间 < 30秒
- [ ] 内存使用正常
- [ ] CPU使用正常

### 兼容性测试
- [ ] Chrome浏览器正常
- [ ] Firefox浏览器正常
- [ ] Edge浏览器正常
- [ ] 移动端响应式布局正常

## 下一步优化建议

1. **添加加载动画**: 在API调用期间显示加载指示器
2. **优化FIN-R1性能**: 使用GPU加速推理
3. **添加缓存机制**: 缓存常见问题的回复
4. **实现实时数据更新**: WebSocket推送实时行情
5. **添加用户认证**: 实现登录和权限管理
6. **完善错误处理**: 更友好的错误提示
7. **添加数据可视化**: 使用图表展示数据
8. **实现策略回测**: 在策略生成后自动运行回测
9. **添加通知系统**: 重要事件的推送通知
10. **性能监控**: 添加APM监控系统

## 联系与支持

如果遇到问题，请：
1. 查看日志文件: `logs/`
2. 检查API响应
3. 查看浏览器控制台
4. 提供详细的错误信息

---

**测试日期**: 2025-10-07
**版本**: 1.0.0
**状态**: 已修复所有核心问题 ✅








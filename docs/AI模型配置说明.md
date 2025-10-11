# FinLoom AI模型配置说明

## 概述

FinLoom 系统采用**主备容错机制**，实现智能AI服务：

- **主服务**: FIN-R1 本地模型（7B参数，性能优秀但可能不稳定）
- **备用服务**: 阿里云通义千问API（稳定可靠，需要API密钥）

系统会自动管理主备切换，确保服务稳定性。

---

## 工作原理

### 1. 主备容错流程

```
用户请求
    |
    v
尝试FIN-R1主服务
    |
    +---> 成功? --> 返回结果 (model_used: fin_r1)
    |
    +---> 失败? --> 切换到阿里云备用服务
                        |
                        +---> 成功? --> 返回结果 (model_used: aliyun)
                        |
                        +---> 失败? --> 返回错误提示
```

### 2. 失败检测机制

系统会在以下情况切换到备用服务：
- FIN-R1响应超时（默认30秒）
- FIN-R1返回错误或异常
- 响应质量不合格（长度过短、包含错误标记等）
- FIN-R1模型未加载或不可用

---

## 配置文件

### 主配置文件: `config/system_config.yaml`

```yaml
# AI模型配置
ai_model:
  provider: aliyun  # 标识配置了阿里云（不影响主备机制）
  
  # 阿里云备用服务配置
  aliyun:
    api_key: "sk-xxxxx"  # 替换为您的API密钥
    model: "qwen-plus"   # 可选: qwen-turbo, qwen-plus, qwen-max
    temperature: 0.7     # 温度参数 (0-1)
    max_tokens: 2000     # 最大生成token数
```

### FIN-R1配置: `module_10_ai_interaction/config/fin_r1_config.yaml`

```yaml
# FIN-R1模型配置
model:
  model_path: ".Fin-R1"
  device: "cpu"  # 或 "cuda" 如果有GPU
  batch_size: 1
  max_length: 2048
  temperature: 0.7
  top_p: 0.9
  top_k: 50
  do_sample: true
  repetition_penalty: 1.1

# 性能配置
performance:
  timeout: 30  # 超时时间（秒）
  max_retries: 3
```

---

## 获取阿里云API密钥

### 步骤

1. 访问阿里云控制台: https://dashscope.console.aliyun.com/
2. 登录您的阿里云账号
3. 进入"API-KEY管理"页面
4. 点击"创建新的API-KEY"
5. 复制生成的API密钥
6. 将密钥填入 `config/system_config.yaml` 的 `api_key` 字段

### 注意事项

- API密钥请妥善保管，不要泄露
- 建议使用环境变量存储敏感信息
- 阿里云模型按调用量计费，请注意费用控制

---

## 使用示例

### Python代码示例

```python
from module_10_ai_interaction import HybridAIService

# 初始化混合AI服务
service = HybridAIService(system_config)

# 对话示例
result = await service.chat(
    user_message="帮我分析一下当前市场适合什么投资策略",
    conversation_history=[],
    system_prompt="你是一个专业的投资顾问"
)

print(f"回复: {result['response']}")
print(f"使用的模型: {result['model_used']}")  # 'fin_r1' 或 'aliyun'
print(f"是否成功: {result['success']}")

# 策略生成示例
strategy_result = await service.generate_strategy(
    investment_requirement={
        'investment_amount': 100000,
        'risk_tolerance': 'moderate',
        'investment_horizon': 'medium_term',
        'investment_goals': ['wealth_growth']
    }
)

print(f"策略: {strategy_result['strategy']}")
print(f"使用的模型: {strategy_result['model_used']}")
```

### API调用示例

```bash
# 对话接口
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "帮我分析一下市场",
    "user_id": "user123"
  }'

# 响应示例
{
  "response": "当前市场...",
  "model_used": "fin_r1",
  "success": true,
  "conversation_id": "conv_xxx"
}
```

---

## 服务监控

### 查看统计信息

```python
# 获取服务统计
stats = service.get_stats()

print(f"总请求数: {stats['total_requests']}")
print(f"FIN-R1成功: {stats['fin_r1_success']}")
print(f"FIN-R1失败: {stats['fin_r1_failure']}")
print(f"阿里云兜底: {stats['aliyun_fallback']}")
print(f"FIN-R1成功率: {stats['fin_r1_success_rate']}")
print(f"阿里云兜底率: {stats['aliyun_fallback_rate']}")

# 重置统计
service.reset_stats()
```

### 日志监控

系统会自动记录详细日志：

```
[INFO] 尝试使用FIN-R1主服务...
[INFO] FIN-R1主服务响应成功
[INFO] 使用模型: fin_r1

# 或当FIN-R1失败时：
[WARNING] FIN-R1主服务失败: timeout
[INFO] 切换到阿里云备用服务...
[INFO] 阿里云备用服务响应成功
[INFO] 使用模型: aliyun
```

---

## 性能优化建议

### 1. FIN-R1优化

- **使用GPU**: 修改配置 `device: "cuda"` 可大幅提升速度
- **减少max_length**: 降低到1024或512可加快响应
- **调整temperature**: 降低可提高响应稳定性

### 2. 阿里云优化

- **选择合适的模型**:
  - `qwen-turbo`: 速度快，成本低，适合简单对话
  - `qwen-plus`: 平衡性能和成本（推荐）
  - `qwen-max`: 最强性能，成本较高

- **控制token数**: 根据需求调整max_tokens，避免浪费

### 3. 系统优化

- **调整超时时间**: 根据实际情况调整timeout参数
- **缓存常见问题**: 对重复问题使用缓存机制
- **负载均衡**: 高并发时考虑部署多个实例

---

## 常见问题

### Q1: FIN-R1加载很慢怎么办？

**A**: FIN-R1是7B参数的大模型，首次加载需要1-2分钟。建议：
- 启动时自动加载，保持在内存中
- 使用GPU加速（需要CUDA环境）
- 如果硬件受限，可以只使用阿里云模式

### Q2: 如何强制使用阿里云模式？

**A**: 有两种方法：
1. 不下载FIN-R1模型，系统会自动使用阿里云
2. 修改代码，直接初始化AliyunAIService

### Q3: 备用切换会影响用户体验吗？

**A**: 切换过程是自动的，用户无感知。唯一区别是：
- FIN-R1: 响应时间约2-5秒
- 阿里云: 响应时间约1-3秒

### Q4: 如何降低阿里云费用？

**A**: 建议：
- 优化系统提示词，减少不必要的token
- 使用qwen-turbo替代qwen-plus
- 实现问题缓存，避免重复调用
- 设置合理的max_tokens限制

### Q5: FIN-R1和阿里云哪个效果更好？

**A**: 各有优势：
- **FIN-R1**: 金融领域专业，离线可用，无API费用
- **阿里云**: 响应稳定，更新及时，支持更大上下文

系统的主备机制确保您能享受两者的优势。

---

## 技术架构

```
┌─────────────────────────────────────────────┐
│           HybridAIService                   │
│         (混合AI服务-主备容错)                │
└────────────┬────────────────────────────────┘
             |
      ┌──────┴──────┐
      |             |
      v             v
┌──────────┐  ┌──────────┐
│ FIN-R1   │  │ Aliyun   │
│ (主服务)  │  │ (备用)    │
└──────────┘  └──────────┘
      |             |
      v             v
  本地模型       API调用
  (7B参数)      (通义千问)
```

---

## 更新日志

### v1.0.0 (2025-10-10)
- 实现FIN-R1主服务 + 阿里云备用服务的主备容错机制
- 添加响应质量检测和自动切换
- 支持对话和策略生成两种模式
- 添加详细的统计和监控功能

---

## 支持

如有问题，请查看：
- 项目README: `README.md`
- API文档: http://localhost:8000/docs
- 阿里云文档: https://help.aliyun.com/document_detail/2400395.html

---

**最后更新**: 2025-10-10










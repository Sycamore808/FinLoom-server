# FinLoom API文档索引

这里是FinLoom系统的API文档中心,包含前后端API对应关系、使用指南和完整的技术文档。

**最后更新**: 2025-10-08  
**文档版本**: v2.0

---

## 🚀 快速开始

### 新手推荐阅读顺序
1. **快速导航** → 了解功能和API对应关系
2. **完成总结** → 了解系统实现状态
3. **API完整对应表** → 查看详细的API文档
4. **图标显示说明** → 了解图标使用规范

---

## 📚 核心文档

### 1. 快速导航 - 前后端API对应 ⭐ NEW
**文件**: `快速导航-前后端API对应.md`

**内容**:
- 智能对话功能的前后端对应
- 策略制定功能的前后端对应
- 文件位置速查表
- API调用示例
- 常用操作流程
- 调试技巧

**适合**: 需要快速查找某个功能对应的API端点

---

### 2. API和界面链接完成总结 ⭐ NEW
**文件**: `API和界面链接完成总结.md`

**内容**:
- 完成的工作清单 (6个页面 + 20个API端点)
- 侧边栏导航结构
- 图标显示优化说明
- 技术架构详解
- 测试建议和检查清单
- 下一步优化建议

**适合**: 了解项目整体进展和实现状态

---

### 3. 前后端API完整对应表 ⭐ NEW
**文件**: `前后端API完整对应表.md`

**内容**:
- 每个页面的详细API文档
- 请求/响应示例
- 模块调用关系图
- 实现状态标记
- 使用示例代码

**适合**: 需要详细了解某个API的请求参数和响应格式

---

### 4. API快速参考表
**文件**: `API快速参考表.md`

**内容**:
- 前端API → 后端模块映射表
- HTTP方法和路由速查
- 主要调用模块标注

**适合**: 快速查询API的基本信息

---

### 5. 网页图标显示说明 ⭐ NEW
**文件**: `网页图标显示说明.md`

**内容**:
- MDI和Font Awesome配置说明
- 侧边栏图标配置
- 图标使用示例
- 常见问题解决方案
- 最佳实践

**适合**: 需要添加或修改界面图标

---

### 6. API对应关系文档
**文件**: `API对应关系文档.md`

**内容**:
- 详细的前后端功能映射
- 实现状态标记
- 代码示例

**适合**: 深入了解某个功能模块

---

### 7. 前后端API功能对应表
**文件**: `前后端API功能对应表.md`

**内容**:
- 每个页面的API映射关系
- 待实现功能清单
- 后端实现代码示例

**适合**: 检查功能实现进度

---

## 🎯 按功能查找文档

### 智能对话功能
- **新对话页面**:
  - 快速导航 → 智能对话功能 → 新对话
  - API完整对应表 → 新对话页面
  
- **历史记录页面**:
  - 快速导航 → 智能对话功能 → 历史记录
  - API完整对应表 → 历史记录页面
  
- **收藏对话页面**:
  - 快速导航 → 智能对话功能 → 收藏对话
  - API完整对应表 → 收藏对话页面

### 策略制定功能
- **创建策略页面**:
  - 快速导航 → 策略制定功能 → 创建策略
  - API完整对应表 → 创建策略页面
  
- **策略库页面**:
  - 快速导航 → 策略制定功能 → 策略库
  - API完整对应表 → 策略库页面
  
- **策略模板页面**:
  - 快速导航 → 策略制定功能 → 策略模板
  - API完整对应表 → 策略模板页面

---

## 🔧 按任务查找文档

### 我想... 查看某个功能对应的API
→ 阅读 **快速导航-前后端API对应.md**

### 我想... 了解API的详细参数和返回值
→ 阅读 **前后端API完整对应表.md**

### 我想... 快速查询某个API端点
→ 阅读 **API快速参考表.md**

### 我想... 添加或修改界面图标
→ 阅读 **网页图标显示说明.md**

### 我想... 了解项目整体实现进度
→ 阅读 **API和界面链接完成总结.md**

### 我想... 测试某个API功能
→ 阅读 **API和界面链接完成总结.md** → 测试建议章节

---

## 📊 模块文档

### Module 01 - 数据管道
**文件**: `../module_01_data_pipeline/module01_README.md`
- 数据采集 (AkShare)
- 数据清洗
- 数据存储

### Module 04 - 市场分析
**文件**: `../module_04_market_analysis/module04_README.md`
- 异常检测
- 情感分析
- 相关性分析
- 市场状态检测

### Module 05 - 风险管理
**文件**: `../module_05_risk_management/module05_README.md`
- 风险计算
- 组合优化
- 止损策略

### Module 07 - 优化模块
**文件**: `../module_07_optimization/module07_README.md`
- 超参数调优
- 策略优化
- 多目标优化

### Module 10 - AI交互 ⭐ 核心
**文件**: `../module_10_ai_interaction/module10_README.md`
- 对话管理
- FIN-R1集成
- 需求解析
- 参数映射
- 推荐引擎

---

## 🛠️ 开发指南

### 添加新的API端点

1. **后端实现** (main.py):
```python
@app.post("/api/v1/your-endpoint")
async def your_endpoint(request: Dict):
    """端点说明"""
    # 实现逻辑
    return {"status": "success", "data": {...}}
```

2. **前端API服务** (web-vue/src/services/api.js):
```javascript
export const api = {
  yourModule: {
    yourMethod: () => apiClient.post('/v1/your-endpoint')
  }
}
```

3. **更新文档**:
   - 在 `API快速参考表.md` 中添加条目
   - 在 `前后端API完整对应表.md` 中添加详细说明

### 添加新的页面

1. **创建Vue组件** (web-vue/src/views/):
```vue
<template>
  <v-container>
    <!-- 页面内容 -->
  </v-container>
</template>

<script setup>
import { api } from '@/services/api'
// 组件逻辑
</script>
```

2. **添加路由** (web-vue/src/router/index.js):
```javascript
{
  path: '/dashboard/your-page',
  name: 'dashboard-your-page',
  component: () => import('@/views/dashboard/YourPageView.vue'),
  meta: { title: '页面标题' }
}
```

3. **更新侧边栏** (web-vue/src/components/layout/Sidebar.vue):
```javascript
{ 
  path: '/dashboard/your-page', 
  icon: 'fas fa-icon', 
  label: '页面名称' 
}
```

4. **更新文档**:
   - 在 `快速导航-前后端API对应.md` 中添加条目
   - 在 `API和界面链接完成总结.md` 中更新计数

---

## 📖 相关资源

### 外部文档
- **Vue 3**: https://cn.vuejs.org/
- **Vuetify 3**: https://vuetifyjs.com/
- **FastAPI**: https://fastapi.tiangolo.com/zh/
- **MDI图标**: https://pictogrammers.com/library/mdi/
- **Font Awesome**: https://fontawesome.com/icons

### 项目文档
- **项目README**: `../README.md`
- **Vue前端README**: `../web-vue/README.md`
- **开发者指南**: `developer_guide/`

---

## 🔄 文档更新记录

| 日期 | 文档 | 变更说明 |
|------|------|----------|
| 2025-10-08 | 快速导航-前后端API对应.md | 新增,提供快速查询功能 |
| 2025-10-08 | API和界面链接完成总结.md | 新增,记录完成情况 |
| 2025-10-08 | 前后端API完整对应表.md | 新增,详细API文档 |
| 2025-10-08 | 网页图标显示说明.md | 新增,图标使用指南 |
| 2025-10-08 | API文档索引.md | 更新,v2.0版本 |

---

## ❓ 常见问题

**Q: 找不到某个API的文档?**  
A: 先查看"快速导航",如果没有再查看"API完整对应表"

**Q: 如何知道某个功能是否已实现?**  
A: 查看"API和界面链接完成总结"中的实现状态总结

**Q: 图标显示不正常怎么办?**  
A: 查看"网页图标显示说明"中的常见问题章节

**Q: 如何测试某个API?**  
A: 查看"API和界面链接完成总结"中的测试建议章节

---

**文档维护**: 添加新功能或修改API时请及时更新相关文档  
**反馈建议**: 发现文档问题请在项目Issue中反馈

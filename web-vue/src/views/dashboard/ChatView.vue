<template>
  <!-- 子路由视图 -->
  <router-view v-if="$route.path !== '/dashboard/chat'" />
  
  <!-- 主聊天界面 -->
  <v-container v-else fluid class="chat-view pa-0" style="height: calc(100vh - 70px - 4rem);">
    <v-row no-gutters style="height: 100%; flex: 1;">
      <!-- 侧边栏 -->
      <v-col cols="auto" class="sidebar-col">
        <v-card 
          variant="flat" 
          class="sidebar-card h-100"
          rounded="0"
        >
          <!-- 头部 -->
          <v-card-text class="pa-4">
            <div class="d-flex justify-space-between align-center mb-4">
              <div class="d-flex align-center">
                <v-avatar color="primary" variant="tonal" size="32" class="mr-3">
                  <v-icon size="18">mdi-chat-outline</v-icon>
                </v-avatar>
                <div>
                  <div class="text-subtitle-1 font-weight-medium">对话历史</div>
                  <div class="text-caption text-medium-emphasis">{{ filteredConversations.length }} 个对话</div>
                </div>
              </div>
              <div class="d-flex gap-1">
                <v-btn 
                  icon="mdi-plus" 
                  variant="text" 
                  size="small" 
                  @click="newConv"
                  color="primary"
                  rounded="lg"
                ></v-btn>
                <v-btn 
                  icon="mdi-cog-outline" 
                  variant="text" 
                  size="small" 
                  @click="openSettings"
                  color="primary"
                  rounded="lg"
                ></v-btn>
              </div>
            </div>
            
            <!-- 搜索框 -->
            <v-text-field
              v-model="searchQuery"
              placeholder="搜索对话..."
              prepend-inner-icon="mdi-magnify"
              variant="outlined"
              density="comfortable"
              hide-details
              rounded="lg"
              class="mb-3"
              bg-color="surface-variant"
            ></v-text-field>
            
            <!-- 筛选按钮 -->
            <div class="d-flex gap-1 flex-wrap">
              <v-chip
                v-for="filter in conversationFilters"
                :key="filter.value"
                :color="activeFilter === filter.value ? 'primary' : 'surface-variant'"
                :variant="activeFilter === filter.value ? 'flat' : 'outlined'"
                size="small"
                @click="activeFilter = filter.value"
                class="cursor-pointer"
                rounded="lg"
              >
                {{ filter.label }}
              </v-chip>
            </div>
          </v-card-text>
          
          <!-- 对话列表 -->
          <v-card-text class="pa-2 pt-0">
            <div
              v-for="conv in filteredConversations"
              :key="conv.id"
              :class="[
                'conversation-item pa-3 mb-2 rounded-xl cursor-pointer transition-all',
                conv.id === chatStore.conversationId 
                  ? 'bg-primary-container text-primary-on-container elevation-1' 
                  : 'hover:bg-surface-variant hover:elevation-1'
              ]"
              @click="switchConv(conv.id)"
            >
              <div class="d-flex justify-space-between align-center">
                <div class="flex-1 min-width-0">
                  <div class="d-flex align-center mb-1">
                    <v-icon 
                      :icon="getConversationIcon(conv.type)" 
                      size="16" 
                      class="mr-2"
                      :class="'text-' + getConversationColor(conv.type)"
                    ></v-icon>
                    <div class="text-body-2 font-weight-medium text-truncate">{{ conv.title }}</div>
                    <v-chip 
                      v-if="conv.isPinned" 
                      size="x-small" 
                      color="warning" 
                      variant="tonal" 
                      class="ml-2"
                    >
                      置顶
                    </v-chip>
                  </div>
                  <div class="text-caption text-medium-emphasis">{{ formatTime(conv.updatedAt || conv.createdAt) }}</div>
                  <div v-if="conv.lastMessage" class="text-caption text-medium-emphasis mt-1 text-truncate">
                    {{ conv.lastMessage }}
                  </div>
                </div>
                <v-menu>
                  <template v-slot:activator="{ props }">
                    <v-btn 
                      icon="mdi-dots-vertical" 
                      variant="text" 
                      size="x-small" 
                      v-bind="props"
                      class="text-medium-emphasis ml-2"
                    ></v-btn>
                  </template>
                  <v-list>
                    <v-list-item @click="pinConversation(conv.id)">
                      <v-list-item-title>
                        <v-icon start size="16">{{ conv.isPinned ? 'mdi-pin-off' : 'mdi-pin' }}</v-icon>
                        {{ conv.isPinned ? '取消置顶' : '置顶' }}
                      </v-list-item-title>
                    </v-list-item>
                    <v-list-item @click="renameConversation(conv.id)">
                      <v-list-item-title>
                        <v-icon start size="16">mdi-pencil</v-icon>
                        重命名
                      </v-list-item-title>
                    </v-list-item>
                    <v-list-item @click="exportConversation(conv.id)">
                      <v-list-item-title>
                        <v-icon start size="16">mdi-download</v-icon>
                        导出
                      </v-list-item-title>
                    </v-list-item>
                    <v-divider></v-divider>
                    <v-list-item @click="deleteConv(conv.id)" class="text-error">
                      <v-list-item-title>
                        <v-icon start size="16">mdi-delete</v-icon>
                        删除
                      </v-list-item-title>
                    </v-list-item>
                  </v-list>
                </v-menu>
              </div>
            </div>
            
            <!-- 空状态 -->
            <div v-if="filteredConversations.length === 0" class="text-center py-12">
              <v-avatar color="surface-variant" variant="tonal" size="64" class="mb-4">
                <v-icon size="32" color="medium-emphasis">mdi-chat-outline</v-icon>
              </v-avatar>
              <div class="text-body-1 text-medium-emphasis">暂无对话</div>
              <div class="text-caption text-medium-emphasis mt-1">开始新的对话吧</div>
            </div>
          </v-card-text>
        </v-card>
      </v-col>

      <!-- 聊天区域 -->
      <v-col class="chat-area-col">
        <v-card variant="flat" class="chat-area-card h-100" rounded="0">
          <!-- 头部 -->
          <v-card-text class="pa-4">
            <div class="d-flex justify-space-between align-center">
              <div class="d-flex align-center">
                <v-avatar color="primary" variant="tonal" size="40" class="mr-3">
                  <v-icon size="24">mdi-robot-outline</v-icon>
                </v-avatar>
                <div>
                  <div class="text-h6 font-weight-medium">FIN-R1 AI助手</div>
                  <div class="text-caption text-medium-emphasis">
                    <v-chip 
                      size="x-small" 
                      color="success" 
                      variant="tonal" 
                      class="mr-2"
                      rounded="lg"
                    >
                      <v-icon start size="12">mdi-circle</v-icon>
                      在线
                    </v-chip>
                    智能投资顾问
                  </div>
                </div>
              </div>
              <div class="d-flex gap-1">
                <v-btn 
                  icon="mdi-refresh" 
                  variant="text" 
                  size="small" 
                  @click="refreshChat" 
                  color="primary"
                  :loading="chatStore.loading"
                  rounded="lg"
                ></v-btn>
                <v-btn 
                  icon="mdi-content-copy" 
                  variant="text" 
                  size="small" 
                  @click="copyConversation" 
                  color="primary"
                  rounded="lg"
                ></v-btn>
                <v-btn 
                  icon="mdi-delete-outline" 
                  variant="text" 
                  size="small" 
                  @click="clearChat" 
                  color="primary"
                  rounded="lg"
                ></v-btn>
              </div>
            </div>
          </v-card-text>

          <!-- 消息区域 -->
          <div ref="messagesRef" class="messages-container">
            <div v-if="chatStore.messages.length === 0" class="empty-state">
              <div class="welcome-section">
                <v-avatar color="primary" variant="tonal" size="96" class="mb-6">
                  <v-icon size="48">mdi-chat-outline</v-icon>
                </v-avatar>
                <h2 class="text-h4 font-weight-bold mb-3">欢迎使用 FIN-R1 AI助手</h2>
                <p class="text-h6 text-medium-emphasis mb-8">向AI助手描述您的投资需求，获取个性化建议</p>
                
                <!-- 功能特色 -->
                <!-- <div class="feature-highlights mb-8">
                  <v-row class="justify-center">
                    <v-col cols="12" sm="4" class="text-center">
                      <v-icon size="32" color="primary" class="mb-2">mdi-brain</v-icon>
                      <div class="text-body-1 font-weight-medium">智能分析</div>
                      <div class="text-caption text-medium-emphasis">深度理解投资需求</div>
                    </v-col>
                    <v-col cols="12" sm="4" class="text-center">
                      <v-icon size="32" color="success" class="mb-2">mdi-chart-line</v-icon>
                      <div class="text-body-1 font-weight-medium">实时数据</div>
                      <div class="text-caption text-medium-emphasis">基于最新市场信息</div>
                    </v-col>
                    <v-col cols="12" sm="4" class="text-center">
                      <v-icon size="32" color="info" class="mb-2">mdi-shield-check</v-icon>
                      <div class="text-body-1 font-weight-medium">风险控制</div>
                      <div class="text-caption text-medium-emphasis">专业风险评估</div>
                    </v-col>
                  </v-row>
                </div> -->
                
                <!-- 快速开始卡片 -->
                <div class="quick-start-section">
                  <h3 class="text-h6 font-weight-medium mb-4 text-center">快速开始</h3>
                  <v-row class="justify-center">
                    <v-col 
                      v-for="q in quickCardItems" 
                      :key="q.text" 
                      cols="12" 
                      sm="6" 
                      md="4"
                      class="d-flex justify-center"
                    >
                      <v-card 
                        variant="elevated" 
                        class="quick-card pa-6 cursor-pointer w-100"
                        rounded="xl"
                        elevation="2"
                        @click="askQuestion(q.text)"
                      >
                        <div class="text-center">
                          <v-avatar :color="q.color" variant="tonal" size="56" class="mb-4">
                            <v-icon :icon="q.icon" size="28"></v-icon>
                          </v-avatar>
                          <div class="text-body-1 font-weight-medium mb-2">{{ q.text }}</div>
                          <div class="text-caption text-medium-emphasis">{{ getCardDescription(q.text) }}</div>
                        </div>
                      </v-card>
                    </v-col>
                  </v-row>
                </div>
              </div>
            </div>

            <div v-for="message in chatStore.messages" :key="message.id" class="message-wrapper mb-4">
              <div class="d-flex" :class="message.role === 'user' ? 'justify-end' : 'justify-start'">
                <v-avatar v-if="message.role === 'assistant'" class="mr-3" size="36" color="primary" variant="tonal">
                  <v-icon size="20">mdi-robot-outline</v-icon>
                </v-avatar>
                <div
                  :class="[
                    'message-bubble pa-4 rounded-xl max-width-70',
                    message.role === 'user' 
                      ? 'bg-primary text-primary-on-surface ml-3 elevation-1' 
                      : 'bg-surface-variant mr-3 elevation-1'
                  ]"
                >
                  <div class="text-body-1">
                    {{ typeof message.content === 'string' ? message.content : JSON.stringify(message.content) }}
                  </div>
                  <div class="text-caption mt-2 opacity-70">
                    {{ formatTime(message.timestamp) }}
                  </div>
                </div>
                <v-avatar v-if="message.role === 'user'" class="ml-3" size="36" color="secondary" variant="tonal">
                  <v-icon size="20">mdi-account-outline</v-icon>
                </v-avatar>
              </div>
            </div>

            <div v-if="chatStore.loading" class="d-flex justify-start mb-4">
              <v-avatar class="mr-3" size="36" color="primary" variant="tonal">
                <v-icon size="20">mdi-robot-outline</v-icon>
              </v-avatar>
              <div class="bg-surface-variant pa-4 rounded-xl mr-3 elevation-1">
                <div class="d-flex align-center gap-3">
                  <v-progress-circular indeterminate size="20" width="2" color="primary"></v-progress-circular>
                  <span class="text-body-1 font-weight-medium">AI正在思考...</span>
                </div>
              </div>
            </div>
          </div>

          <!-- 输入区域 -->
          <div class="input-container">
            <v-card 
              variant="outlined" 
              class="input-card"
              rounded="xl"
              elevation="2"
            >
              <v-card-text class="pa-0">
                <div class="d-flex align-center">
                  <v-textarea
                    v-model="inputMessage"
                    placeholder="描述您的投资需求..."
                    variant="plain"
                    rows="1"
                    auto-grow
                    hide-details
                    density="comfortable"
                    class="input-textarea"
                    @keydown.enter.exact.prevent="sendMessage"
                    @keydown.enter.shift.exact="inputMessage += '\n'"
                  ></v-textarea>
                  <div class="input-actions">
                    <v-btn
                      icon="mdi-send"
                      color="primary"
                      variant="flat"
                      :disabled="!inputMessage.trim() || chatStore.loading"
                      @click="sendMessage"
                      size="small"
                      rounded="lg"
                      class="send-btn"
                    >
                      <v-icon size="18">mdi-send</v-icon>
                    </v-btn>
                  </div>
                </div>
              </v-card-text>
            </v-card>
            <div class="input-hint">
              <span class="text-caption text-medium-emphasis">
                按 Enter 发送，Shift + Enter 换行
              </span>
            </div>
          </div>
        </v-card>
      </v-col>
    </v-row>

    <!-- 设置对话框 -->
    <v-dialog v-model="settingsOpen" max-width="500">
      <v-card rounded="xl">
        <v-card-title class="d-flex align-center">
          <v-icon start>mdi-cog-outline</v-icon>
          AI 设置
        </v-card-title>
        <v-divider></v-divider>
        <v-card-text class="pa-6">
          <v-select
            v-model="localSettings.model"
            :items="[
              { title: 'FIN-R1', value: 'fin-r1' },
              { title: 'FIN-R1 Pro', value: 'fin-r1-pro' }
            ]"
            label="模型"
            variant="filled"
            density="comfortable"
            rounded="lg"
            class="mb-4"
          ></v-select>

          <div class="mb-4">
            <label class="text-body-2 mb-2 d-block">温度: {{ localSettings.temperature.toFixed(2) }}</label>
            <v-slider
              v-model="localSettings.temperature"
              min="0"
              max="1"
              step="0.05"
              color="primary"
            ></v-slider>
          </div>

          <v-select
            v-model="localSettings.riskTolerance"
            :items="[
              { title: '保守', value: 'low' },
              { title: '中等', value: 'medium' },
              { title: '激进', value: 'high' }
            ]"
            label="风险偏好"
            variant="filled"
            density="comfortable"
            rounded="lg"
          ></v-select>
        </v-card-text>
        <v-card-actions class="pa-6 pt-0">
          <v-spacer></v-spacer>
          <v-btn variant="text" @click="settingsOpen = false" rounded="pill">取消</v-btn>
          <v-btn color="primary" @click="saveSettings" rounded="pill" variant="flat">保存</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-container>
</template>

<script setup>
import { ref, nextTick, computed } from 'vue'
import { useChatStore } from '@/stores/chat'

const chatStore = useChatStore()
const inputMessage = ref('')
const messagesRef = ref(null)
const settingsOpen = ref(false)
const searchQuery = ref('')
const activeFilter = ref('all')

const localSettings = ref({
  model: 'fin-r1',
  temperature: 0.7,
  riskTolerance: 'medium'
})

// 对话筛选器
const conversationFilters = [
  { label: '全部', value: 'all' },
  { label: '置顶', value: 'pinned' },
  { label: '最近', value: 'recent' },
  { label: '投资', value: 'investment' },
  { label: '风险', value: 'risk' }
]

// 快速问题卡片
const quickCardItems = [
  { text: '推荐一些稳健的投资标的', icon: 'mdi-chart-line', color: 'primary' },
  { text: '分析当前市场趋势', icon: 'mdi-trending-up', color: 'success' },
  { text: '帮我优化投资组合', icon: 'mdi-chart-pie', color: 'secondary' },
  { text: '评估投资风险', icon: 'mdi-shield-alert', color: 'warning' },
  { text: '解释技术指标', icon: 'mdi-chart-areaspline', color: 'info' },
  { text: '制定交易策略', icon: 'mdi-strategy', color: 'tertiary' }
]

// 计算属性
const filteredConversations = computed(() => {
  let conversations = [...chatStore.conversations]
  
  // 搜索过滤
  if (searchQuery.value) {
    conversations = conversations.filter(conv => 
      conv.title.toLowerCase().includes(searchQuery.value.toLowerCase()) ||
      conv.lastMessage?.toLowerCase().includes(searchQuery.value.toLowerCase())
    )
  }
  
  // 类型过滤
  if (activeFilter.value === 'pinned') {
    conversations = conversations.filter(conv => conv.isPinned)
  } else if (activeFilter.value === 'recent') {
    // 最近7天
    const sevenDaysAgo = new Date()
    sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7)
    conversations = conversations.filter(conv => 
      new Date(conv.updatedAt || conv.createdAt) >= sevenDaysAgo
    )
  } else if (activeFilter.value === 'investment') {
    conversations = conversations.filter(conv => conv.type === 'investment')
  } else if (activeFilter.value === 'risk') {
    conversations = conversations.filter(conv => conv.type === 'risk')
  }
  
  return conversations
})

async function sendMessage() {
  if (!inputMessage.value.trim() || chatStore.loading) return
  
  await chatStore.sendMessage(inputMessage.value)
  inputMessage.value = ''
  await nextTick()
  scrollToBottom()
}

function scrollToBottom() {
  if (messagesRef.value) {
    messagesRef.value.scrollTop = messagesRef.value.scrollHeight
  }
}

function askQuestion(text) {
  inputMessage.value = text
  sendMessage()
}

function newConv() {
  chatStore.createConversation()
}

function switchConv(id) {
  chatStore.switchConversation(id)
}

function deleteConv(id) {
  chatStore.deleteConversation(id)
}

function clearChat() {
  chatStore.clearMessages()
}

function openSettings() {
  settingsOpen.value = true
}

function saveSettings() {
  settingsOpen.value = false
}

function formatTime(timestamp) {
  return new Date(timestamp).toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  })
}

// 新增工具函数
function getConversationIcon(type) {
  const icons = {
    'investment': 'mdi-chart-line',
    'risk': 'mdi-shield-alert',
    'strategy': 'mdi-strategy',
    'general': 'mdi-chat-outline',
    'analysis': 'mdi-chart-areaspline'
  }
  return icons[type] || 'mdi-chat-outline'
}

function getConversationColor(type) {
  const colors = {
    'investment': 'primary',
    'risk': 'error',
    'strategy': 'secondary',
    'general': 'default',
    'analysis': 'info'
  }
  return colors[type] || 'default'
}

// 新增事件处理
function refreshChat() {
  // 刷新聊天状态
  console.log('刷新聊天')
}

function copyConversation() {
  // 复制当前对话
  const conversationText = chatStore.messages.map(msg => 
    `${msg.role === 'user' ? '用户' : 'AI'}: ${msg.content}`
  ).join('\n')
  
  navigator.clipboard.writeText(conversationText).then(() => {
    console.log('对话已复制到剪贴板')
  })
}

function pinConversation(id) {
  // 置顶/取消置顶对话
  const conv = chatStore.conversations.find(c => c.id === id)
  if (conv) {
    conv.isPinned = !conv.isPinned
  }
}

function renameConversation(id) {
  // 重命名对话
  const newName = prompt('请输入新的对话名称:')
  if (newName) {
    const conv = chatStore.conversations.find(c => c.id === id)
    if (conv) {
      conv.title = newName
    }
  }
}

function exportConversation(id) {
  // 导出对话
  const conv = chatStore.conversations.find(c => c.id === id)
  if (conv) {
    console.log('导出对话:', conv.title)
  }
}

// 获取卡片描述
function getCardDescription(text) {
  const descriptions = {
    '推荐一些稳健的投资标的': '基于风险偏好推荐优质股票',
    '分析当前市场趋势': '深度解读市场动态和机会',
    '帮我优化投资组合': '智能调整资产配置比例',
    '评估投资风险': '全面分析投资风险因素',
    '解释技术指标': '详细解读各类技术分析指标',
    '制定交易策略': '量身定制个性化交易方案'
  }
  return descriptions[text] || '点击开始对话'
}
</script>

<style lang="scss" scoped>
.chat-view {
  height: calc(100vh - 70px - 4rem);
  background: rgb(var(--v-theme-surface));
  border-radius: 16px;
  overflow: visible; // 改为visible允许滚动
  box-shadow: 0 4px 12px rgba(var(--v-theme-shadow), 0.1);
  margin: -2rem; // 抵消父容器的padding
  width: calc(100% + 4rem); // 抵消父容器的padding
  display: flex;
  flex-direction: column;
}

// 侧边栏样式
.sidebar-col {
  width: 320px;
  min-width: 320px;
  max-width: 320px;
}

.sidebar-card {
  background: rgb(var(--v-theme-surface-container-lowest));
  border-right: 1px solid rgba(var(--v-theme-outline), 0.08);
  position: relative;
  border-radius: 0 0 0 16px; // 左侧圆角
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    bottom: 0;
    width: 1px;
    background: linear-gradient(180deg, 
      transparent 0%, 
      rgba(var(--v-theme-primary), 0.15) 50%, 
      transparent 100%
    );
  }
}

// 聊天区域样式
.chat-area-col {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
}

.chat-area-card {
  background: rgb(var(--v-theme-surface));
  display: flex;
  flex-direction: column;
  position: relative;
  border-radius: 0 0 16px 0; // 右侧圆角
  flex: 1; // 使用flex: 1而不是height: 100%
  min-height: 0; // 允许内容收缩
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, 
      transparent 0%, 
      rgba(var(--v-theme-primary), 0.15) 50%, 
      transparent 100%
    );
  }
}

// 消息容器
.messages-container {
  flex: 1;
  padding: 0 24px 24px 24px; // 移除顶部padding
  background: rgb(var(--v-theme-surface));
  min-height: 0; // 确保flex容器可以收缩
  overflow-y: auto; // 启用滚动
  height: calc(100vh - 70px - 4rem - 140px); // 设置固定高度，减去TopNavbar、content-wrapper padding和输入框高度
}

// 空状态
.empty-state {
  text-align: center;
  padding: 0;
  display: block; // 改为block，去除flex布局
  position: relative;
  width: 100%;
}

.welcome-section {
  max-width: 800px;
  margin: 0 auto;
  width: 100%;
  padding: 32px 24px 40px 24px; // 从顶部开始的小间距
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start; // 改为顶部对齐
}

.feature-highlights {
  background: linear-gradient(135deg, 
    rgba(var(--v-theme-primary), 0.02) 0%, 
    rgba(var(--v-theme-secondary), 0.02) 100%
  );
  border-radius: 24px;
  padding: 32px 24px;
  border: 1px solid rgba(var(--v-theme-outline), 0.08);
}

.quick-start-section {
  margin-top: 32px;
}

// 快速卡片
.quick-card {
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  background: rgb(var(--v-theme-surface-container-lowest));
  border: 1px solid rgba(var(--v-theme-outline), 0.12);
  position: relative;
  overflow: hidden;
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, 
      rgba(var(--v-theme-primary), 0.8) 0%, 
      rgba(var(--v-theme-secondary), 0.8) 100%
    );
    transform: scaleX(0);
    transition: transform 0.3s ease;
  }
  
  &:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(var(--v-theme-primary), 0.15);
    border-color: rgba(var(--v-theme-primary), 0.3);
    
    &::before {
      transform: scaleX(1);
    }
  }
  
  &:active {
    transform: translateY(-2px);
  }
}

// 对话项
.conversation-item {
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  border: 1px solid transparent;
  position: relative;
  overflow: hidden;
  
  &::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, 
      rgba(var(--v-theme-primary), 0.8) 0%, 
      rgba(var(--v-theme-secondary), 0.8) 100%
    );
    transform: scaleY(0);
    transition: transform 0.3s ease;
  }
  
  &:hover {
    transform: translateX(4px);
    border-color: rgba(var(--v-theme-outline), 0.2);
    box-shadow: 0 4px 12px rgba(var(--v-theme-shadow), 0.1);
    
    &::before {
      transform: scaleY(1);
    }
  }
}

// 消息气泡
.message-bubble {
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  word-wrap: break-word;
  border: 1px solid rgba(var(--v-theme-outline), 0.12);
  position: relative;
  overflow: hidden;
  
  &::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(135deg, 
      rgba(var(--v-theme-primary), 0.02) 0%, 
      transparent 50%
    );
    opacity: 0;
    transition: opacity 0.3s ease;
  }
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(var(--v-theme-shadow), 0.2);
    border-color: rgba(var(--v-theme-outline), 0.2);
    
    &::after {
      opacity: 1;
    }
  }
}

.message-wrapper {
  animation: fadeInUp 0.3s ease-out;
  
  &:first-child {
    margin-top: 16px; // 第一条消息的顶部间距
  }
}

.cursor-pointer {
  cursor: pointer;
}

// OpenWebUI 风格输入框
.input-container {
  position: relative;
  background: rgba(var(--v-theme-surface), 0.98);
  padding: 16px 24px 24px 24px;
  border-top: 1px solid rgba(var(--v-theme-outline), 0.12);
  backdrop-filter: blur(16px);
  z-index: 100;
  border-radius: 0 0 16px 0; // 底部右侧圆角
  box-shadow: 0 -4px 12px rgba(var(--v-theme-shadow), 0.08);
  flex-shrink: 0; // 防止输入框被压缩
  margin-top: auto; // 推到底部
  min-height: 80px; // 确保输入框有最小高度
}

.input-card {
  background: rgb(var(--v-theme-surface-container-lowest));
  border: 1px solid rgba(var(--v-theme-outline), 0.08);
  box-shadow: 0 2px 8px rgba(var(--v-theme-shadow), 0.06);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  overflow: hidden;
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, 
      rgba(var(--v-theme-primary), 0.6) 0%, 
      rgba(var(--v-theme-secondary), 0.6) 100%
    );
    transform: scaleX(0);
    transition: transform 0.3s ease;
  }
  
  &:hover {
    border-color: rgba(var(--v-theme-outline), 0.15);
    box-shadow: 0 4px 16px rgba(var(--v-theme-shadow), 0.1);
  }
  
  &:focus-within {
    border-color: rgba(var(--v-theme-primary), 0.3);
    box-shadow: 0 6px 20px rgba(var(--v-theme-primary), 0.12);
    
    &::before {
      transform: scaleX(1);
    }
  }
}

.input-textarea {
  flex: 1;
  min-height: 48px;
  max-height: 120px;
  
  :deep(.v-field) {
    box-shadow: none;
    border: none;
    background: transparent;
  }
  
  :deep(.v-field__input) {
    padding: 12px 16px;
    font-size: 14px;
    line-height: 1.5;
    resize: none;
  }
  
  :deep(.v-field__outline) {
    display: none;
  }
}

.input-actions {
  padding: 8px 12px 8px 0;
  display: flex;
  align-items: center;
}

.send-btn {
  min-width: 44px;
  height: 44px;
  box-shadow: 0 2px 8px rgba(var(--v-theme-primary), 0.15);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  overflow: hidden;
  
  &::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    background: rgba(255, 255, 255, 0.3);
    border-radius: 50%;
    transform: translate(-50%, -50%);
    transition: all 0.3s ease;
  }
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(var(--v-theme-primary), 0.25);
    
    &::before {
      width: 100%;
      height: 100%;
    }
  }
  
  &:active {
    transform: translateY(-1px);
    box-shadow: 0 3px 10px rgba(var(--v-theme-primary), 0.2);
  }
}

.input-hint {
  text-align: center;
  margin-top: 8px;
  opacity: 0.7;
}

// 自定义滚动条
.messages-container {
  &::-webkit-scrollbar {
    width: 6px;
  }
  
  &::-webkit-scrollbar-track {
    background: transparent;
  }
  
  &::-webkit-scrollbar-thumb {
    background: rgba(var(--v-theme-outline), 0.2);
    border-radius: 3px;
    
    &:hover {
      background: rgba(var(--v-theme-outline), 0.3);
    }
  }
}

// 响应式调整
@media (max-width: 960px) {
  .chat-view {
    height: calc(100vh - 70px - 2rem);
    margin: -1rem;
    width: calc(100% + 2rem);
  }
  
  .sidebar-col {
    width: 280px;
    min-width: 280px;
    max-width: 280px;
  }
  
  .conversation-item {
    padding: 12px !important;
  }
  
  .messages-container {
    padding: 0 16px 16px 16px; // 移除顶部padding
    height: calc(100vh - 70px - 2rem - 120px); // 移动端固定高度
  }
  
  .empty-state {
    padding: 0;
  }
  
  .welcome-section {
    padding: 24px 16px 20px 16px;
  }
  
  .input-container {
    padding: 12px 16px 16px 16px;
  }
}

// 动画效果
@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes slideInLeft {
  from {
    opacity: 0;
    transform: translateX(-20px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

@keyframes pulse {
  0%, 100% {
    transform: scale(1);
  }
  50% {
    transform: scale(1.05);
  }
}

.conversation-item {
  animation: slideInLeft 0.4s ease-out;
}

.quick-card {
  animation: fadeInUp 0.5s ease-out;
}

.message-wrapper {
  animation: fadeInUp 0.3s ease-out;
}

// 延迟动画
.conversation-item:nth-child(1) { animation-delay: 0.1s; }
.conversation-item:nth-child(2) { animation-delay: 0.2s; }
.conversation-item:nth-child(3) { animation-delay: 0.3s; }
.conversation-item:nth-child(4) { animation-delay: 0.4s; }
.conversation-item:nth-child(5) { animation-delay: 0.5s; }

.quick-card:nth-child(1) { animation-delay: 0.1s; }
.quick-card:nth-child(2) { animation-delay: 0.2s; }
.quick-card:nth-child(3) { animation-delay: 0.3s; }
.quick-card:nth-child(4) { animation-delay: 0.4s; }
.quick-card:nth-child(5) { animation-delay: 0.5s; }
.quick-card:nth-child(6) { animation-delay: 0.6s; }

// Material 3 颜色调整
:deep(.v-card) {
  box-shadow: 0 1px 3px rgba(var(--v-theme-shadow), 0.12), 0 1px 2px rgba(var(--v-theme-shadow), 0.24);
}

:deep(.v-btn) {
  text-transform: none;
  font-weight: 500;
  letter-spacing: 0.01em;
}

:deep(.v-chip) {
  font-weight: 500;
  letter-spacing: 0.01em;
}

:deep(.v-text-field) {
  .v-field__outline {
    --v-field-border-opacity: 0.08;
  }
  
  .v-field--focused .v-field__outline {
    --v-field-border-opacity: 0.3;
  }
}

:deep(.v-list-item) {
  border-radius: 12px;
  margin: 2px 0;
  
  &:hover {
    background: rgba(var(--v-theme-primary), 0.04);
  }
  
  &.v-list-item--active {
    background: rgba(var(--v-theme-primary), 0.08);
  }
}

</style>

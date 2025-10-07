<template>
  <v-container fluid class="chat-view pa-0" style="height: calc(100vh - 120px);">
    <v-row no-gutters style="height: 100%;">
      <!-- 侧边栏 -->
      <v-col cols="auto" style="border-right: 1px solid rgb(var(--v-border-color)); height: 100%;">
        <v-card variant="flat" style="width: 280px; height: 100%;" rounded="0">
          <v-card-title class="d-flex justify-space-between align-center pa-4">
            <div class="d-flex align-center">
              <v-avatar color="primary" variant="tonal" size="36" class="mr-2">
                <v-icon size="20">mdi-history</v-icon>
              </v-avatar>
              <span class="text-subtitle-1 font-weight-bold">历史对话</span>
            </div>
            <v-btn icon="mdi-plus" color="primary" variant="tonal" size="small" @click="newConv"></v-btn>
          </v-card-title>
          <v-list density="compact">
            <v-list-item
              v-for="conv in chatStore.conversations"
              :key="conv.id"
              :active="conv.id === chatStore.conversationId"
              @click="switchConv(conv.id)"
            >
              <v-list-item-title>{{ conv.title }}</v-list-item-title>
              <template v-slot:append>
                <v-btn icon="mdi-delete" variant="text" size="x-small" @click.stop="deleteConv(conv.id)"></v-btn>
              </template>
            </v-list-item>
          </v-list>
        </v-card>
      </v-col>

      <!-- 聊天区域 -->
      <v-col style="height: 100%; display: flex; flex-direction: column;">
        <v-card variant="flat" style="flex: 1; display: flex; flex-direction: column; height: 100%;">
          <v-card-title class="d-flex justify-space-between align-center pa-6">
            <div class="d-flex align-center">
              <v-avatar color="primary" variant="tonal" size="48" class="mr-3">
                <v-icon>mdi-robot</v-icon>
              </v-avatar>
              <div>
                <div class="text-h6 font-weight-bold">FIN-R1 AI助手</div>
                <div class="text-caption text-medium-emphasis">智能投资顾问</div>
              </div>
            </div>
            <div>
              <v-btn icon="mdi-cog" variant="text" size="small" @click="openSettings"></v-btn>
              <v-btn icon="mdi-delete-sweep" variant="text" size="small" @click="clearChat"></v-btn>
            </div>
          </v-card-title>

          <!-- 消息区域 -->
          <v-card-text ref="messagesRef" style="flex: 1; overflow-y: auto; padding: 24px;">
            <div v-if="chatStore.messages.length === 0" class="text-center py-10">
              <v-avatar color="primary" variant="tonal" size="96" class="mb-6">
                <v-icon size="56">mdi-message-text-outline</v-icon>
              </v-avatar>
              <h3 class="text-h4 font-weight-bold mb-2">开始对话</h3>
              <p class="text-body-1 text-medium-emphasis mb-8">向AI助手描述您的投资需求，获取个性化建议</p>
              <v-row>
                <v-col v-for="q in quickCardItems" :key="q.text" cols="12" sm="6">
                  <v-card variant="flat" class="bg-primary-container" hover @click="askQuestion(q.text)">
                    <v-card-text class="pa-6 text-center">
                      <v-avatar :color="q.color" variant="tonal" size="56" class="mb-4">
                        <v-icon :icon="q.icon" size="32"></v-icon>
                      </v-avatar>
                      <div class="text-body-1 font-weight-medium">{{ q.text }}</div>
                    </v-card-text>
                  </v-card>
                </v-col>
              </v-row>
            </div>

            <div v-for="message in chatStore.messages" :key="message.id" class="mb-6">
              <div class="d-flex" :class="message.role === 'user' ? 'justify-end' : 'justify-start'">
                <v-avatar v-if="message.role === 'assistant'" color="primary" variant="tonal" class="mr-3" size="40">
                  <v-icon>mdi-robot</v-icon>
                </v-avatar>
                <v-card
                  :color="message.role === 'user' ? 'primary' : undefined"
                  :variant="message.role === 'user' ? 'flat' : 'flat'"
                  :class="message.role === 'user' ? '' : 'bg-surface-variant'"
                  max-width="70%"
                  elevation="0"
                >
                  <v-card-text class="pa-4">
                    <div :class="message.role === 'user' ? 'text-white' : ''">
                      {{ typeof message.content === 'string' ? message.content : JSON.stringify(message.content) }}
                    </div>
                    <div class="text-caption mt-2" :class="message.role === 'user' ? 'text-white opacity-75' : 'text-medium-emphasis'">
                      {{ formatTime(message.timestamp) }}
                    </div>
                  </v-card-text>
                </v-card>
                <v-avatar v-if="message.role === 'user'" color="secondary" variant="tonal" class="ml-3" size="40">
                  <v-icon>mdi-account</v-icon>
                </v-avatar>
              </div>
            </div>

            <div v-if="chatStore.loading" class="d-flex justify-start mb-6">
              <v-avatar color="primary" variant="tonal" class="mr-3" size="40">
                <v-icon>mdi-robot</v-icon>
              </v-avatar>
              <v-card variant="flat" class="bg-surface-variant">
                <v-card-text class="pa-4">
                  <div class="d-flex align-center gap-3">
                    <v-progress-circular indeterminate size="24" width="3" color="primary"></v-progress-circular>
                    <span class="font-weight-medium">AI正在思考...</span>
                  </div>
                </v-card-text>
              </v-card>
            </div>
          </v-card-text>

          <!-- 输入区域 -->
          <v-divider></v-divider>
          <v-card-actions class="pa-4">
            <v-textarea
              v-model="inputMessage"
              placeholder="描述您的投资需求..."
              variant="outlined"
              rows="2"
              auto-grow
              hide-details
              @keydown.enter.exact.prevent="sendMessage"
            >
              <template v-slot:append-inner>
                <v-btn
                  icon="mdi-send"
                  color="primary"
                  :disabled="!inputMessage.trim() || chatStore.loading"
                  @click="sendMessage"
                ></v-btn>
              </template>
            </v-textarea>
          </v-card-actions>
        </v-card>
      </v-col>
    </v-row>

    <!-- 设置对话框 -->
    <v-dialog v-model="settingsOpen" max-width="500">
      <v-card>
        <v-card-title>
          <v-icon start>mdi-cog</v-icon>
          AI 设置
        </v-card-title>
        <v-divider></v-divider>
        <v-card-text>
          <v-select
            v-model="localSettings.model"
            :items="[
              { title: 'FIN-R1', value: 'fin-r1' },
              { title: 'FIN-R1 Pro', value: 'fin-r1-pro' }
            ]"
            label="模型"
            variant="outlined"
            density="comfortable"
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
            variant="outlined"
            density="comfortable"
          ></v-select>
        </v-card-text>
        <v-card-actions>
          <v-spacer></v-spacer>
          <v-btn variant="text" @click="settingsOpen = false">取消</v-btn>
          <v-btn color="primary" @click="saveSettings">保存</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-container>
</template>

<script setup>
import { ref, nextTick } from 'vue'
import { useChatStore } from '@/stores/chat'

const chatStore = useChatStore()
const inputMessage = ref('')
const messagesRef = ref(null)
const settingsOpen = ref(false)

const localSettings = ref({
  model: 'fin-r1',
  temperature: 0.7,
  riskTolerance: 'medium'
})

const quickCardItems = [
  { text: '推荐一些稳健的投资标的', icon: 'mdi-chart-line', color: 'primary' },
  { text: '分析当前市场趋势', icon: 'mdi-trending-up', color: 'success' },
  { text: '帮我优化投资组合', icon: 'mdi-chart-pie', color: 'secondary' },
  { text: '评估投资风险', icon: 'mdi-shield-alert', color: 'warning' }
]

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
</script>

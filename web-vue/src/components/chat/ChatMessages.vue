<template>
  <div ref="messagesRef" class="messages-container">
    <!-- 空状态 -->
    <div v-if="messages.length === 0" class="empty-state">
      <div class="welcome-section">
        <v-avatar color="primary" variant="tonal" size="96" class="mb-6">
          <v-icon size="48">mdi-chat-outline</v-icon>
        </v-avatar>
        <h2 class="text-h4 font-weight-bold mb-3">欢迎使用 FIN-R1 AI助手</h2>
        <p class="text-h6 text-medium-emphasis mb-8">向AI助手描述您的投资需求，获取个性化建议</p>
        
        <!-- 快速开始卡片 -->
        <div class="quick-start-section">
          <h3 class="text-h6 font-weight-medium mb-4 text-center">快速开始</h3>
          <v-row class="justify-center">
            <v-col 
              v-for="q in quickQuestions" 
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
                @click="$emit('ask-question', q.text)"
              >
                <div class="text-center">
                  <v-avatar :color="q.color" variant="tonal" size="56" class="mb-4">
                    <v-icon :icon="q.icon" size="28"></v-icon>
                  </v-avatar>
                  <div class="text-body-1 font-weight-medium mb-2">{{ q.text }}</div>
                  <div class="text-caption text-medium-emphasis">{{ q.description }}</div>
                </div>
              </v-card>
            </v-col>
          </v-row>
        </div>
      </div>
    </div>

    <!-- 消息列表 -->
    <div v-for="message in messages" :key="message.id" class="message-wrapper mb-4">
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

    <!-- 加载状态 -->
    <div v-if="loading" class="d-flex justify-start mb-4">
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
</template>

<script setup>
import { ref } from 'vue'
import { formatTime } from '@/utils/date'

defineProps({
  messages: {
    type: Array,
    required: true
  },
  loading: {
    type: Boolean,
    default: false
  },
  quickQuestions: {
    type: Array,
    required: true
  }
})

defineEmits(['ask-question'])

const messagesRef = ref(null)

defineExpose({
  messagesRef
})
</script>


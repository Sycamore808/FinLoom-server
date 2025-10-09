<template>
  <!-- 子路由视图 -->
  <router-view v-if="$route.path !== '/dashboard/chat'" />
  
  <!-- 主聊天界面 -->
  <v-container v-else fluid class="chat-view pa-0" style="height: calc(100vh - 70px - 4rem);">
    <v-row no-gutters style="height: 100%; flex: 1;">
      <!-- 侧边栏 -->
      <v-col cols="auto" class="sidebar-col">
        <ChatSidebar
          :conversations="filteredConversations"
          :current-conversation-id="conversationId"
          v-model:search-query="searchQuery"
          v-model:active-filter="activeFilter"
          :filters="conversationFilters"
          @new-conversation="newConversation"
          @open-settings="openSettings"
          @switch-conversation="switchConversation"
          @delete-conversation="deleteConversation"
          @pin-conversation="pinConversation"
          @rename-conversation="renameConversation"
          @export-conversation="exportConversationHandler"
        />
      </v-col>

      <!-- 聊天区域 -->
      <v-col class="chat-area-col">
        <v-card variant="flat" class="chat-area-card h-100" rounded="0">
          <!-- 头部 -->
          <ChatHeader
            :loading="loading"
            @refresh="refreshChat"
            @copy="copyConversation"
            @clear="clearMessages"
          />

          <!-- 消息区域 -->
          <ChatMessages
            ref="messagesComponent"
            :messages="messages"
            :loading="loading"
            :quick-questions="quickQuestions"
            @ask-question="askQuestion"
          />

          <!-- 输入区域 -->
          <ChatInput
            v-model="inputMessage"
            :loading="loading"
            @send="sendMessage"
          />
        </v-card>
      </v-col>
    </v-row>

    <!-- 设置对话框 -->
    <AISettingsDialog
      v-model="settingsOpen"
      :settings="localSettings"
      @save="saveSettings"
      @cancel="settingsOpen = false"
      @update-setting="updateSettingValue"
    />
  </v-container>
</template>

<script setup>
import { ref, computed } from 'vue'
import ChatSidebar from '@/components/chat/ChatSidebar.vue'
import ChatHeader from '@/components/chat/ChatHeader.vue'
import ChatMessages from '@/components/chat/ChatMessages.vue'
import ChatInput from '@/components/chat/ChatInput.vue'
import AISettingsDialog from '@/components/chat/AISettingsDialog.vue'
import { useChat } from '@/composables/useChat'
import { useConversationActions } from '@/composables/useConversationActions'
import { useAISettings } from '@/composables/useAISettings'
import { CONVERSATION_FILTERS, QUICK_QUESTIONS } from '@/constants/chat'

// 使用组合式函数
const {
  inputMessage,
  messagesRef,
  searchQuery,
  activeFilter,
  filteredConversations,
  messages,
  conversations,
  conversationId,
  loading,
  error,
  sendMessage,
  scrollToBottom,
  askQuestion,
  newConversation,
  switchConversation,
  deleteConversation,
  clearMessages,
  refreshChat
} = useChat()

const {
  pinConversation,
  renameConversation,
  exportConversation: exportConversationHandler,
  copyConversation
} = useConversationActions()

const {
  settingsOpen,
  localSettings,
  openSettings,
  saveSettings
} = useAISettings()

// 常量
const conversationFilters = CONVERSATION_FILTERS
const quickQuestions = QUICK_QUESTIONS

// 消息组件引用
const messagesComponent = ref(null)

// 更新设置值
function updateSettingValue(key, value) {
  localSettings.value[key] = value
}
</script>

<style lang="scss" scoped>
@import '@/assets/styles/views/chat-view.scss';
</style>


import { defineStore } from 'pinia'
import { ref } from 'vue'
import { api } from '@/services/api'
import { nanoid } from 'nanoid'

export const useChatStore = defineStore('chat', () => {
  // 状态
  const messages = ref([])
  const conversationId = ref('')
  const loading = ref(false)
  const error = ref(null)
  const conversations = ref([]) // 历史会话列表
  const settings = ref({        // AI 设置
    model: 'fin-r1',
    temperature: 0.7,
    riskTolerance: 'balanced'
  })
  const initialized = ref(false)
  
  // 操作
  function initializeConversation(title = '') {
    ensureLoaded()
    const id = nanoid()
    conversationId.value = id
    messages.value = []
    conversations.value.unshift({
      id,
      title: title || '新对话',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      messages: []
    })
    persistConversations()
  }
  
  function addMessage(message) {
    messages.value.push({
      id: nanoid(),
      timestamp: new Date(),
      ...message
    })
    // 同步到当前会话并更新标题/时间
    syncCurrentConversation()
    if (message.role === 'user') {
      maybeSetConversationTitleFrom(message.content)
    }
  }
  
  async function sendMessage(text, amount = null, riskTolerance = null, attachments = null) {
    try {
      loading.value = true
      error.value = null
      
      // 添加用户消息
      addMessage({
        role: 'user',
        content: text,
        attachments: Array.isArray(attachments) ? attachments : null
      })
      
      // 调用AI对话API
      const response = await api.chat.aiChat(text, amount, riskTolerance)
      
      if (response.status === 'success') {
        // 添加AI回复
        addMessage({
          role: 'assistant',
          content: response.data,
          raw: response
        })
      } else {
        throw new Error(response.message || '分析失败')
      }
      
      return response
    } catch (err) {
      console.error('发送消息失败:', err)
      error.value = err.message || '发送失败'
      
      // 添加错误消息
      addMessage({
        role: 'assistant',
        content: '抱歉，我现在遇到了一些问题，请稍后再试。',
        error: true
      })
      
      throw err
    } finally {
      loading.value = false
    }
  }
  
  async function sendSimpleMessage(text) {
    try {
      loading.value = true
      error.value = null
      
      // 添加用户消息
      addMessage({
        role: 'user',
        content: text
      })
      
      // 调用简化对话API
      const response = await api.chat.send(text, conversationId.value)
      
      if (response.status === 'success') {
        // 添加AI回复
        addMessage({
          role: 'assistant',
          content: response.response,
          detailedData: response.detailed_data
        })
      } else {
        throw new Error(response.response || '发送失败')
      }
      
      return response
    } catch (err) {
      console.error('发送消息失败:', err)
      error.value = err.message || '发送失败'
      
      addMessage({
        role: 'assistant',
        content: '抱歉，我现在遇到了一些问题，请稍后再试。',
        error: true
      })
      
      throw err
    } finally {
      loading.value = false
    }
  }
  
  function clearMessages() {
    messages.value = []
    error.value = null
    syncCurrentConversation()
  }

  // 会话历史与设置
  function ensureLoaded() {
    if (initialized.value) return
    try {
      const convRaw = localStorage.getItem('finloom_conversations')
      const cfgRaw = localStorage.getItem('finloom_ai_settings')
      conversations.value = convRaw ? JSON.parse(convRaw) : []
      if (cfgRaw) {
        const cfg = JSON.parse(cfgRaw)
        settings.value = {
          ...settings.value,
          ...cfg
        }
      }
    } catch (e) {
      console.warn('加载历史/设置失败', e)
    } finally {
      initialized.value = true
    }
    // 若无会话则初始化一个
    if (!conversations.value.length) {
      initializeConversation()
    } else {
      // 进入最近的会话
      const latest = conversations.value[0]
      conversationId.value = latest.id
      messages.value = Array.isArray(latest.messages) ? latest.messages.map(m => ({ ...m, timestamp: new Date(m.timestamp) })) : []
    }
  }

  function persistConversations() {
    try {
      localStorage.setItem('finloom_conversations', JSON.stringify(conversations.value))
    } catch {}
  }

  function persistSettings() {
    try {
      localStorage.setItem('finloom_ai_settings', JSON.stringify(settings.value))
    } catch {}
  }

  function syncCurrentConversation() {
    const idx = conversations.value.findIndex(c => c.id === conversationId.value)
    if (idx >= 0) {
      conversations.value[idx] = {
        ...conversations.value[idx],
        messages: messages.value,
        updatedAt: new Date().toISOString()
      }
      persistConversations()
    }
  }

  function maybeSetConversationTitleFrom(text) {
    const idx = conversations.value.findIndex(c => c.id === conversationId.value)
    if (idx >= 0 && conversations.value[idx].title === '新对话') {
      const cleaned = String(text || '').replace(/\s+/g, ' ').trim()
      conversations.value[idx].title = cleaned ? cleaned.slice(0, 20) : '新对话'
      persistConversations()
    }
  }

  function newConversation(title = '') {
    initializeConversation(title)
  }

  function switchConversation(id) {
    ensureLoaded()
    const conv = conversations.value.find(c => c.id === id)
    if (!conv) return
    conversationId.value = id
    messages.value = Array.isArray(conv.messages) ? conv.messages.map(m => ({ ...m, timestamp: new Date(m.timestamp) })) : []
  }

  function deleteConversation(id) {
    const idx = conversations.value.findIndex(c => c.id === id)
    if (idx < 0) return
    conversations.value.splice(idx, 1)
    persistConversations()
    if (conversationId.value === id) {
      if (conversations.value.length) {
        switchConversation(conversations.value[0].id)
      } else {
        initializeConversation()
      }
    }
  }

  function renameConversation(id, title) {
    const idx = conversations.value.findIndex(c => c.id === id)
    if (idx < 0) return
    conversations.value[idx].title = String(title || '新对话').slice(0, 60)
    persistConversations()
  }

  function updateSettings(partial) {
    settings.value = { ...settings.value, ...partial }
    persistSettings()
  }
  
  return {
    // 状态
    messages,
    conversationId,
    loading,
    error,
    conversations,
    settings,
    
    // 操作
    initializeConversation,
    addMessage,
    sendMessage,
    sendSimpleMessage,
    clearMessages,
    // 历史与设置
    ensureLoaded,
    newConversation,
    switchConversation,
    deleteConversation,
    renameConversation,
    updateSettings
  }
})


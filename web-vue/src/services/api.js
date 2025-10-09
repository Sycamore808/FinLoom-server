import axios from 'axios'

// 创建axios实例
const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截器
apiClient.interceptors.request.use(
  config => {
    // 可以在这里添加token等认证信息
    const token = localStorage.getItem('finloom_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  error => {
    return Promise.reject(error)
  }
)

// 响应拦截器
apiClient.interceptors.response.use(
  response => {
    return response.data
  },
  error => {
    console.error('API错误:', error)
    
    if (error.response) {
      // 服务器返回错误状态码
      const { status, data } = error.response
      
      if (status === 401) {
        // 未授权，跳转登录
        localStorage.removeItem('finloom_auth')
        localStorage.removeItem('finloom_token')
        window.location.href = '/login'
      }
      
      return Promise.reject(data || error.message)
    } else if (error.request) {
      // 请求已发送但没有收到响应
      return Promise.reject({ message: '网络错误，请检查连接' })
    } else {
      // 请求配置出错
      return Promise.reject({ message: error.message })
    }
  }
)

// API服务
export const api = {
  // 健康检查
  health: {
    check: () => axios.get('/health'),
    ready: () => apiClient.get('/v1/ready')
  },
  
  // AI对话
  chat: {
    send: (message, conversationId = '') => 
      apiClient.post('/chat', { message, conversation_id: conversationId }),
    
    aiChat: (text, amount = null, riskTolerance = null) =>
      apiClient.post('/v1/ai/chat', {
        text,
        amount,
        risk_tolerance: riskTolerance
      }),
    
    // 对话管理
    createConversation: (userId = 'default_user', title = '新对话') =>
      apiClient.post('/v1/chat/conversation', { user_id: userId, title }),
    
    getConversations: (userId = 'default_user', limit = 50) =>
      apiClient.get('/v1/chat/conversations', { params: { user_id: userId, limit } }),
    
    getHistory: (conversationId) =>
      apiClient.get(`/v1/chat/history/${conversationId}`),
    
    deleteConversation: (conversationId) =>
      apiClient.delete(`/v1/chat/conversation/${conversationId}`),
    
    searchConversations: (query, userId = 'default_user', limit = 20) =>
      apiClient.get('/v1/chat/search', { params: { query, user_id: userId, limit } }),
    
    // 收藏对话
    addFavorite: (sessionId, data) =>
      apiClient.post('/v1/chat/favorite', { session_id: sessionId, ...data }),
    
    removeFavorite: (sessionId, userId = 'default_user') =>
      apiClient.delete(`/v1/chat/favorite/${sessionId}`, { params: { user_id: userId } }),
    
    getFavorites: (userId = 'default_user', limit = 50) =>
      apiClient.get('/v1/chat/favorites', { params: { user_id: userId, limit } }),
    
    checkFavorite: (sessionId, userId = 'default_user') =>
      apiClient.get(`/v1/chat/favorite/check/${sessionId}`, { params: { user_id: userId } }),
    
    updateFavorite: (sessionId, data) =>
      apiClient.put(`/v1/chat/favorite/${sessionId}`, data)
  },
  
  // 仪表盘
  dashboard: {
    getMetrics: () => apiClient.get('/v1/dashboard/metrics')
  },
  
  // 投资组合
  portfolio: {
    getPositions: () => apiClient.get('/v1/portfolio/positions')
  },
  
  // 交易
  trades: {
    getRecent: () => apiClient.get('/v1/trades/recent')
  },
  
  // 回测
  backtest: {
    run: (params) => apiClient.post('/v1/backtest/run', params)
  },
  
  // 数据管理
  data: {
    collect: (params) => apiClient.post('/v1/data/collect', params),
    getOverview: () => apiClient.get('/v1/data/overview')
  },
  
  // 市场分析
  market: {
    getOverview: () => apiClient.get('/v1/market/overview'),
    
    // 综合分析API
    analysis: {
      detectAnomaly: (params) => 
        apiClient.post('/v1/analysis/anomaly/detect', params),
      
      analyzeCorrelation: (params) =>
        apiClient.post('/v1/analysis/correlation/analyze', params),
      
      detectRegime: (params) =>
        apiClient.post('/v1/analysis/regime/detect', params),
      
      analyzeSentiment: (params) =>
        apiClient.post('/v1/analysis/sentiment/analyze', params),
      
      aggregateSentiment: (params) =>
        apiClient.post('/v1/analysis/sentiment/aggregate', params)
    }
  },
  
  // 策略管理
  strategy: {
    // 生成策略
    generate: (requirements) =>
      apiClient.post('/v1/strategy/generate', { requirements }),
    
    // 保存策略
    save: (strategy) =>
      apiClient.post('/v1/strategy/save', { strategy }),
    
    // 获取策略列表
    list: (userId = 'default_user', limit = 50) =>
      apiClient.get('/v1/strategy/list', { params: { user_id: userId, limit } }),
    
    // 获取策略详情
    get: (strategyId) =>
      apiClient.get(`/v1/strategy/${strategyId}`),
    
    // 删除策略
    delete: (strategyId) =>
      apiClient.delete(`/v1/strategy/${strategyId}`),
    
    // 复制策略
    duplicate: (strategyId, name) =>
      apiClient.post(`/v1/strategy/${strategyId}/duplicate`, { name }),
    
    // 优化策略
    optimize: (parameters, symbols = ['000001']) =>
      apiClient.post('/v1/strategy/optimize', { parameters, symbols }),
    
    // 回测策略
    backtest: (strategyId, params) =>
      apiClient.post(`/v1/strategy/${strategyId}/backtest`, params),
    
    // 策略模板
    templates: {
      list: () =>
        apiClient.get('/v1/strategy/templates'),
      
      get: (templateId) =>
        apiClient.get(`/v1/strategy/templates/${templateId}`),
      
      createFrom: (templateId, name, parameters = {}) =>
        apiClient.post(`/v1/strategy/from-template/${templateId}`, { name, parameters })
    }
  },
  
  // 分析接口（兼容旧版本）
  analyze: (params) => apiClient.post('/v1/analyze', params)
}

export default apiClient


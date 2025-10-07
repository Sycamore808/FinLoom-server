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
      })
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
  
  // 分析接口（兼容旧版本）
  analyze: (params) => apiClient.post('/v1/analyze', params)
}

export default apiClient


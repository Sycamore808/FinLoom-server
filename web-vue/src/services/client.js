/**
 * Axios客户端配置
 */

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
    // 添加认证token
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

export default apiClient


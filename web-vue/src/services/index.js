/**
 * API服务统一导出
 */

import { healthApi } from './modules/health'
import { chatApi } from './modules/chat'
import { strategyApi } from './modules/strategy'
import { marketApi } from './modules/market'
import { dashboardApi } from './modules/dashboard'
import { portfolioApi } from './modules/portfolio'
import { tradesApi } from './modules/trades'
import { backtestApi } from './modules/backtest'
import { dataApi } from './modules/data'
import apiClient from './client'

// 统一导出API对象（兼容旧版本）
export const api = {
  health: healthApi,
  chat: chatApi,
  strategy: strategyApi,
  market: marketApi,
  dashboard: dashboardApi,
  portfolio: portfolioApi,
  trades: tradesApi,
  backtest: backtestApi,
  data: dataApi,
  
  // 兼容旧版本的analyze接口
  analyze: (params) => apiClient.post('/v1/analyze', params)
}

// 导出各个模块（供按需引入）
export {
  healthApi,
  chatApi,
  strategyApi,
  marketApi,
  dashboardApi,
  portfolioApi,
  tradesApi,
  backtestApi,
  dataApi
}

// 导出axios实例
export { default as apiClient } from './client'

// 默认导出
export default api


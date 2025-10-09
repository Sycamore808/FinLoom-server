/**
 * 市场分析相关API服务
 */

import apiClient from '../client'

export const marketApi = {
  /**
   * 获取市场概览（包含指数和热门股票）
   */
  getOverview: () => apiClient.get('/v1/market/overview'),
  
  /**
   * 获取市场指数（专门为OverviewView优化）
   */
  getIndices: () => apiClient.get('/v1/market/indices'),
  
  /**
   * 获取热门股票（专门为MarketView优化）
   */
  getHotStocks: () => apiClient.get('/v1/market/hot-stocks'),
  
  /**
   * 市场分析子模块
   */
  analysis: {
    /**
     * 异常检测
     */
    detectAnomaly: (params) => 
      apiClient.post('/v1/analysis/anomaly/detect', params),
    
    /**
     * 相关性分析
     */
    analyzeCorrelation: (params) =>
      apiClient.post('/v1/analysis/correlation/analyze', params),
    
    /**
     * 市场状态检测
     */
    detectRegime: (params) =>
      apiClient.post('/v1/analysis/regime/detect', params),
    
    /**
     * 情绪分析
     */
    analyzeSentiment: (params) =>
      apiClient.post('/v1/analysis/sentiment/analyze', params),
    
    /**
     * 聚合情绪分析
     */
    aggregateSentiment: (params) =>
      apiClient.post('/v1/analysis/sentiment/aggregate', params)
  }
}


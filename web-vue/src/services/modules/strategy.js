/**
 * 策略相关API服务
 */

import apiClient from '../client'

export const strategyApi = {
  /**
   * 生成策略
   */
  generate: (requirements) =>
    apiClient.post('/v1/strategy/generate', { requirements }),
  
  /**
   * 保存策略
   */
  save: (strategy) =>
    apiClient.post('/v1/strategy/save', { strategy }),
  
  /**
   * 获取策略列表
   */
  list: (userId = 'default_user', limit = 50) =>
    apiClient.get('/v1/strategy/list', { params: { user_id: userId, limit } }),
  
  /**
   * 获取策略详情
   */
  get: (strategyId) =>
    apiClient.get(`/v1/strategy/${strategyId}`),
  
  /**
   * 删除策略
   */
  delete: (strategyId) =>
    apiClient.delete(`/v1/strategy/${strategyId}`),
  
  /**
   * 复制策略
   */
  duplicate: (strategyId, name) =>
    apiClient.post(`/v1/strategy/${strategyId}/duplicate`, { name }),
  
  /**
   * 优化策略
   */
  optimize: (parameters, symbols = ['000001']) =>
    apiClient.post('/v1/strategy/optimize', { parameters, symbols }),
  
  /**
   * 回测策略
   */
  backtest: (strategyId, params) =>
    apiClient.post(`/v1/strategy/${strategyId}/backtest`, params),
  
  /**
   * 策略模板
   */
  templates: {
    /**
     * 获取模板列表
     */
    list: () =>
      apiClient.get('/v1/strategy/templates'),
    
    /**
     * 获取模板详情
     */
    get: (templateId) =>
      apiClient.get(`/v1/strategy/templates/${templateId}`),
    
    /**
     * 从模板创建策略
     */
    createFrom: (templateId, name, parameters = {}) =>
      apiClient.post(`/v1/strategy/from-template/${templateId}`, { name, parameters })
  }
}


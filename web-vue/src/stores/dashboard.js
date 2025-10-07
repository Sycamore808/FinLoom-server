import { defineStore } from 'pinia'
import { ref } from 'vue'
import { api } from '@/services/api'

export const useDashboardStore = defineStore('dashboard', () => {
  // 状态
  const metrics = ref({
    total_assets: 0,
    daily_return: 0,
    sharpe_ratio: 0,
    max_drawdown: 0,
    win_rate: 0,
    total_trades: 0
  })
  
  const positions = ref([])
  const recentTrades = ref([])
  const loading = ref(false)
  const error = ref(null)
  
  // 操作
  async function fetchMetrics() {
    try {
      loading.value = true
      const response = await api.dashboard.getMetrics()
      metrics.value = response.data || {}
      error.value = null
    } catch (err) {
      console.error('获取仪表盘指标失败:', err)
      error.value = err.message || '获取数据失败'
    } finally {
      loading.value = false
    }
  }
  
  async function fetchPositions() {
    try {
      loading.value = true
      const response = await api.portfolio.getPositions()
      positions.value = response.data?.positions || []
      error.value = null
    } catch (err) {
      console.error('获取持仓数据失败:', err)
      error.value = err.message || '获取持仓失败'
    } finally {
      loading.value = false
    }
  }
  
  async function fetchRecentTrades() {
    try {
      loading.value = true
      const response = await api.trades.getRecent()
      recentTrades.value = response.data?.trades || []
      error.value = null
    } catch (err) {
      console.error('获取交易记录失败:', err)
      error.value = err.message || '获取交易记录失败'
    } finally {
      loading.value = false
    }
  }
  
  async function refreshAll() {
    await Promise.all([
      fetchMetrics(),
      fetchPositions(),
      fetchRecentTrades()
    ])
  }
  
  return {
    // 状态
    metrics,
    positions,
    recentTrades,
    loading,
    error,
    
    // 操作
    fetchMetrics,
    fetchPositions,
    fetchRecentTrades,
    refreshAll
  }
})


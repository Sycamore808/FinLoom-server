<template>
  <v-container fluid class="overview-view pa-6">
    <!-- 页面头部 -->
    <div class="mb-6">
      <div class="d-flex justify-space-between align-center mb-4">
        <div>
          <h1 class="text-h3 font-weight-bold mb-2">仪表盘概览</h1>
        </div>
        <div class="d-flex gap-2">
          <v-alert
            type="success"
            variant="tonal"
            class="mb-0"
            rounded="lg"
            density="compact"
          >
            <template v-slot:prepend>
              <v-icon>mdi-check-circle</v-icon>
            </template>
            <span class="text-body-2 font-weight-medium">市场正常运行</span>
          </v-alert>
        </div>
      </div>
    </div>

    <div v-if="dashboardStore.loading && !metrics" class="text-center py-10">
      <v-progress-circular indeterminate color="primary" size="64"></v-progress-circular>
      <p class="mt-4 text-body-1">加载数据中...</p>
    </div>

    <div v-else>
      <!-- 关键指标卡片 - Material 3 风格 -->
      <v-row class="mb-6">
        <v-col cols="12" sm="6" md="3">
          <v-card variant="elevated" class="metric-card bg-primary-container" hover>
            <v-card-text class="pa-6">
              <div class="d-flex justify-space-between align-start mb-4">
                <v-icon size="48" color="primary">mdi-wallet</v-icon>
                <v-menu>
                  <template v-slot:activator="{ props }">
                    <v-btn icon="mdi-dots-vertical" variant="text" size="small" v-bind="props"></v-btn>
                  </template>
                  <v-list>
                    <v-list-item @click="viewDetails('assets')">
                      <v-list-item-title>查看详情</v-list-item-title>
                    </v-list-item>
                    <v-list-item @click="setAlert('assets')">
                      <v-list-item-title>设置预警</v-list-item-title>
                    </v-list-item>
                  </v-list>
                </v-menu>
              </div>
              <div class="text-caption mb-1 text-primary">总资产</div>
              <div class="text-h4 font-weight-bold mb-3">
                ¥{{ formatNumber(metrics.total_assets) }}
              </div>
              <div class="d-flex align-center justify-space-between">
                <v-chip 
                  size="small" 
                  :color="metrics.assets_change >= 0 ? 'success' : 'error'" 
                  variant="flat" 
                  :class="metrics.assets_change >= 0 ? 'bg-success-lighten-4' : 'bg-error-lighten-4'"
                >
                  <v-icon start size="16">{{ metrics.assets_change >= 0 ? 'mdi-trending-up' : 'mdi-trending-down' }}</v-icon>
                  {{ formatPercent(metrics.assets_change) }}
                </v-chip>
                <span class="text-caption text-medium-emphasis">vs 昨日</span>
              </div>
            </v-card-text>
          </v-card>
        </v-col>

        <v-col cols="12" sm="6" md="3">
          <v-card variant="elevated" class="metric-card bg-secondary-container" hover>
            <v-card-text class="pa-6">
              <div class="d-flex justify-space-between align-start mb-4">
                <v-icon size="48" color="secondary">mdi-trending-up</v-icon>
                <v-menu>
                  <template v-slot:activator="{ props }">
                    <v-btn icon="mdi-dots-vertical" variant="text" size="small" v-bind="props"></v-btn>
                  </template>
                  <v-list>
                    <v-list-item @click="viewDetails('returns')">
                      <v-list-item-title>查看详情</v-list-item-title>
                    </v-list-item>
                    <v-list-item @click="setAlert('returns')">
                      <v-list-item-title>设置预警</v-list-item-title>
                    </v-list-item>
                  </v-list>
                </v-menu>
              </div>
              <div class="text-caption mb-1 text-secondary">日收益</div>
              <div class="text-h4 font-weight-bold mb-3">
                ¥{{ formatNumber(metrics.daily_return) }}
              </div>
              <div class="d-flex align-center justify-space-between">
                <v-chip 
                  size="small" 
                  :color="metrics.daily_return >= 0 ? 'success' : 'error'" 
                  variant="flat" 
                  :class="metrics.daily_return >= 0 ? 'bg-success-lighten-4' : 'bg-error-lighten-4'"
                >
                  <v-icon start size="16">{{ metrics.daily_return >= 0 ? 'mdi-trending-up' : 'mdi-trending-down' }}</v-icon>
                  {{ formatPercent(metrics.daily_return_pct) }}
                </v-chip>
                <span class="text-caption text-medium-emphasis">今日</span>
              </div>
            </v-card-text>
          </v-card>
        </v-col>

        <v-col cols="12" sm="6" md="3">
          <v-card variant="elevated" class="metric-card bg-tertiary-container" hover>
            <v-card-text class="pa-6">
              <div class="d-flex justify-space-between align-start mb-4">
                <v-icon size="48" color="tertiary">mdi-chart-line</v-icon>
                <v-menu>
                  <template v-slot:activator="{ props }">
                    <v-btn icon="mdi-dots-vertical" variant="text" size="small" v-bind="props"></v-btn>
                  </template>
                  <v-list>
                    <v-list-item @click="viewDetails('sharpe')">
                      <v-list-item-title>查看详情</v-list-item-title>
                    </v-list-item>
                    <v-list-item @click="setAlert('sharpe')">
                      <v-list-item-title>设置预警</v-list-item-title>
                    </v-list-item>
                  </v-list>
                </v-menu>
              </div>
              <div class="text-caption mb-1 text-tertiary">夏普比率</div>
              <div class="text-h4 font-weight-bold mb-3">
                {{ formatNumber(metrics.sharpe_ratio, 2) }}
              </div>
              <div class="d-flex align-center justify-space-between">
                <v-chip 
                  size="small" 
                  :color="getSharpeColor(metrics.sharpe_ratio)" 
                  variant="flat" 
                  :class="getSharpeColor(metrics.sharpe_ratio) + '-lighten-4'"
                >
                  <v-icon start size="16">{{ getSharpeIcon(metrics.sharpe_ratio) }}</v-icon>
                  {{ getSharpeLabel(metrics.sharpe_ratio) }}
                </v-chip>
                <span class="text-caption text-medium-emphasis">风险调整</span>
              </div>
            </v-card-text>
          </v-card>
        </v-col>

        <v-col cols="12" sm="6" md="3">
          <v-card variant="elevated" class="metric-card bg-error-container" hover>
            <v-card-text class="pa-6">
              <div class="d-flex justify-space-between align-start mb-4">
                <v-icon size="48" color="error">mdi-arrow-down</v-icon>
                <v-menu>
                  <template v-slot:activator="{ props }">
                    <v-btn icon="mdi-dots-vertical" variant="text" size="small" v-bind="props"></v-btn>
                  </template>
                  <v-list>
                    <v-list-item @click="viewDetails('drawdown')">
                      <v-list-item-title>查看详情</v-list-item-title>
                    </v-list-item>
                    <v-list-item @click="setAlert('drawdown')">
                      <v-list-item-title>设置预警</v-list-item-title>
                    </v-list-item>
                  </v-list>
                </v-menu>
              </div>
              <div class="text-caption mb-1 text-error">最大回撤</div>
              <div class="text-h4 font-weight-bold mb-3">
                {{ formatNumber(metrics.max_drawdown, 2) }}%
              </div>
              <div class="d-flex align-center justify-space-between">
                <v-chip 
                  size="small" 
                  :color="getDrawdownColor(metrics.max_drawdown)" 
                  variant="flat" 
                  :class="getDrawdownColor(metrics.max_drawdown) + '-lighten-4'"
                >
                  <v-icon start size="16">{{ getDrawdownIcon(metrics.max_drawdown) }}</v-icon>
                  {{ getDrawdownLabel(metrics.max_drawdown) }}
                </v-chip>
                <span class="text-caption text-medium-emphasis">历史最大</span>
              </div>
            </v-card-text>
          </v-card>
        </v-col>
      </v-row>

      <!-- 新增：市场概览卡片 -->
      <v-row class="mb-6">
        <v-col cols="12">
          <v-card variant="elevated" class="market-overview-card">
            <v-card-title class="d-flex align-center pa-6">
              <v-avatar color="info" variant="tonal" size="40" class="mr-3">
                <v-icon>mdi-chart-multiple</v-icon>
              </v-avatar>
              <div>
                <div class="text-h6 font-weight-bold">市场概览</div>
                <div class="text-caption text-medium-emphasis">主要指数实时表现</div>
              </div>
              <v-spacer></v-spacer>
              <v-btn
                color="primary"
                variant="text"
                size="small"
                @click="refreshMarketData"
                :loading="marketLoading"
                prepend-icon="mdi-refresh"
              >
                刷新
              </v-btn>
            </v-card-title>
            <v-card-text class="pa-6 pt-0">
              <v-row>
                <v-col v-for="index in marketIndices" :key="index.symbol" cols="12" sm="6" md="3">
                  <v-card variant="outlined" class="index-card" hover>
                    <v-card-text class="pa-4">
                      <div class="d-flex justify-space-between align-center mb-2">
                        <div class="text-subtitle-2 font-weight-bold">{{ index.name }}</div>
                        <v-chip size="x-small" :color="index.change >= 0 ? 'success' : 'error'" variant="tonal">
                          {{ index.change >= 0 ? '+' : '' }}{{ formatPercent(index.change_pct) }}
                        </v-chip>
                      </div>
                      <div class="text-h6 font-weight-bold mb-1">{{ formatNumber(index.value, 2) }}</div>
                      <div class="d-flex align-center">
                        <v-icon 
                          :color="index.change >= 0 ? 'success' : 'error'" 
                          size="16" 
                          class="mr-1"
                        >
                          {{ index.change >= 0 ? 'mdi-trending-up' : 'mdi-trending-down' }}
                        </v-icon>
                        <span 
                          :class="index.change >= 0 ? 'text-success' : 'text-error'" 
                          class="text-caption font-weight-medium"
                        >
                          {{ index.change >= 0 ? '+' : '' }}{{ formatNumber(index.change, 2) }}
                        </span>
                      </div>
                    </v-card-text>
                  </v-card>
                </v-col>
              </v-row>
            </v-card-text>
          </v-card>
        </v-col>
      </v-row>

      <!-- 图表区域 - Material 3 风格 -->
      <v-row>
        <!-- 投资组合分布 -->
        <v-col cols="12" md="6">
          <v-card variant="elevated" class="chart-card">
            <v-card-title class="d-flex align-center justify-space-between pa-6">
              <div class="d-flex align-center">
                <v-avatar color="primary" variant="tonal" size="40" class="mr-3">
                  <v-icon>mdi-chart-donut</v-icon>
                </v-avatar>
                <div>
                  <div class="text-h6 font-weight-bold">投资组合分布</div>
                  <div class="text-caption text-medium-emphasis">按市值占比</div>
                </div>
              </div>
              <v-menu>
                <template v-slot:activator="{ props }">
                  <v-btn icon="mdi-dots-vertical" variant="text" size="small" v-bind="props"></v-btn>
                </template>
                <v-list>
                  <v-list-item @click="exportChart('portfolio')">
                    <v-list-item-title>导出图表</v-list-item-title>
                  </v-list-item>
                  <v-list-item @click="viewFullChart('portfolio')">
                    <v-list-item-title>全屏查看</v-list-item-title>
                  </v-list-item>
                </v-list>
              </v-menu>
            </v-card-title>
            <v-card-text class="px-6 pb-6">
              <div class="chart-container">
                <canvas ref="portfolioChartRef"></canvas>
              </div>
              <div v-if="positions.length === 0" class="text-center py-8">
                <v-icon size="48" class="text-medium-emphasis mb-4">mdi-chart-donut-variant</v-icon>
                <p class="text-body-2 text-medium-emphasis">暂无持仓数据</p>
              </div>
            </v-card-text>
          </v-card>
        </v-col>

        <!-- 收益曲线 -->
        <v-col cols="12" md="6">
          <v-card variant="elevated" class="chart-card">
            <v-card-title class="d-flex align-center justify-space-between pa-6">
              <div class="d-flex align-center">
                <v-avatar color="success" variant="tonal" size="40" class="mr-3">
                  <v-icon>mdi-chart-areaspline</v-icon>
                </v-avatar>
                <div>
                  <div class="text-h6 font-weight-bold">收益曲线</div>
                  <div class="text-caption text-medium-emphasis">资产净值变化</div>
                </div>
              </div>
              <div class="d-flex gap-2">
                <v-btn-toggle v-model="chartPeriod" mandatory>
                  <v-btn value="1M" size="small">1M</v-btn>
                  <v-btn value="3M" size="small">3M</v-btn>
                  <v-btn value="1Y" size="small">1Y</v-btn>
                </v-btn-toggle>
                <v-menu>
                  <template v-slot:activator="{ props }">
                    <v-btn icon="mdi-dots-vertical" variant="text" size="small" v-bind="props"></v-btn>
                  </template>
                  <v-list>
                    <v-list-item @click="exportChart('equity')">
                      <v-list-item-title>导出图表</v-list-item-title>
                    </v-list-item>
                    <v-list-item @click="viewFullChart('equity')">
                      <v-list-item-title>全屏查看</v-list-item-title>
                    </v-list-item>
                  </v-list>
                </v-menu>
              </div>
            </v-card-title>
            <v-card-text class="px-6 pb-6">
              <div class="chart-container">
                <canvas ref="equityChartRef"></canvas>
              </div>
            </v-card-text>
          </v-card>
        </v-col>

        <!-- 风险指标仪表盘 -->
        <v-col cols="12" md="6">
          <v-card variant="elevated" class="chart-card">
            <v-card-title class="d-flex align-center pa-6">
              <v-avatar color="warning" variant="tonal" size="40" class="mr-3">
                <v-icon>mdi-gauge</v-icon>
              </v-avatar>
              <div>
                <div class="text-h6 font-weight-bold">风险指标</div>
                <div class="text-caption text-medium-emphasis">实时风险评估</div>
              </div>
            </v-card-title>
            <v-card-text class="pa-6">
              <v-row>
                <v-col cols="6">
                  <div class="text-center">
                    <div class="text-h4 font-weight-bold mb-2" :class="getRiskColor(riskMetrics.var_95)">
                      {{ formatNumber(riskMetrics.var_95, 2) }}%
                    </div>
                    <div class="text-caption text-medium-emphasis">VaR (95%)</div>
                  </div>
                </v-col>
                <v-col cols="6">
                  <div class="text-center">
                    <div class="text-h4 font-weight-bold mb-2" :class="getRiskColor(riskMetrics.beta)">
                      {{ formatNumber(riskMetrics.beta, 2) }}
                    </div>
                    <div class="text-caption text-medium-emphasis">Beta系数</div>
                  </div>
                </v-col>
              </v-row>
              <v-divider class="my-4"></v-divider>
              <div class="d-flex justify-space-between align-center mb-2">
                <span class="text-body-2">波动率</span>
                <span class="font-weight-bold">{{ formatNumber(riskMetrics.volatility, 2) }}%</span>
              </div>
              <v-progress-linear
                :model-value="Math.min(riskMetrics.volatility, 50)"
                :color="getVolatilityColor(riskMetrics.volatility)"
                height="8"
                rounded
              ></v-progress-linear>
            </v-card-text>
          </v-card>
        </v-col>

        <!-- 最近交易 -->
        <v-col cols="12" md="6">
          <v-card variant="elevated" class="chart-card">
            <v-card-title class="d-flex align-center justify-space-between pa-6">
              <div class="d-flex align-center">
                <v-avatar color="info" variant="tonal" size="40" class="mr-3">
                  <v-icon>mdi-history</v-icon>
                </v-avatar>
                <div>
                  <div class="text-h6 font-weight-bold">最近交易</div>
                  <div class="text-caption text-medium-emphasis">最新交易记录</div>
                </div>
              </div>
              <v-btn
                color="primary"
                variant="text"
                size="small"
                @click="viewAllTrades"
                prepend-icon="mdi-arrow-right"
              >
                查看全部
              </v-btn>
            </v-card-title>
            <v-card-text class="pa-0">
              <v-list density="compact">
                <v-list-item
                  v-for="trade in recentTrades.slice(0, 5)"
                  :key="trade.time"
                  class="px-6"
                >
                  <template v-slot:prepend>
                    <v-avatar
                      :color="trade.action === 'BUY' ? 'success' : 'error'"
                      size="32"
                      variant="tonal"
                    >
                      <v-icon size="16">
                        {{ trade.action === 'BUY' ? 'mdi-arrow-up' : 'mdi-arrow-down' }}
                      </v-icon>
                    </v-avatar>
                  </template>
                  <v-list-item-title class="font-weight-medium">{{ trade.name }}</v-list-item-title>
                  <v-list-item-subtitle>
                    {{ trade.action === 'BUY' ? '买入' : '卖出' }} {{ trade.quantity }}股
                  </v-list-item-subtitle>
                  <template v-slot:append>
                    <div class="text-end">
                      <div class="font-weight-bold">¥{{ formatNumber(trade.price, 2) }}</div>
                      <div class="text-caption text-medium-emphasis">{{ formatTime(trade.time) }}</div>
                    </div>
                  </template>
                </v-list-item>
              </v-list>
              <div v-if="recentTrades.length === 0" class="text-center py-8">
                <v-icon size="48" class="text-medium-emphasis mb-4">mdi-history</v-icon>
                <p class="text-body-2 text-medium-emphasis">暂无交易记录</p>
              </div>
            </v-card-text>
          </v-card>
        </v-col>
      </v-row>
    </div>
  </v-container>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { useDashboardStore } from '@/stores/dashboard'
import { useRouter } from 'vue-router'
import Chart from 'chart.js/auto'

const dashboardStore = useDashboardStore()
const router = useRouter()

const portfolioChartRef = ref(null)
const equityChartRef = ref(null)
const chartPeriod = ref('3M')
const marketLoading = ref(false)

const metrics = computed(() => dashboardStore.metrics)
const positions = computed(() => dashboardStore.positions)
const recentTrades = computed(() => dashboardStore.recentTrades)


// 市场指数数据
const marketIndices = ref([
  { symbol: 'SH000001', name: '上证指数', value: 3000.25, change: 15.32, change_pct: 0.51 },
  { symbol: 'SZ399001', name: '深证成指', value: 9500.15, change: -25.18, change_pct: -0.26 },
  { symbol: 'SZ399006', name: '创业板指', value: 1850.45, change: 8.75, change_pct: 0.48 },
  { symbol: 'HKHSI', name: '恒生指数', value: 16500.30, change: 120.45, change_pct: 0.73 }
])

// 风险指标
const riskMetrics = ref({
  var_95: 2.5,
  beta: 1.2,
  volatility: 18.5
})


let portfolioChart = null
let equityChart = null

onMounted(async () => {
  await dashboardStore.refreshAll()
  await loadMarketData()
  initCharts()
})

watch(positions, () => {
  updatePortfolioChart()
})

watch(chartPeriod, () => {
  updateEquityChart()
})

// 初始化图表
function initCharts() {
  initPortfolioChart()
  initEquityChart()
}

function initPortfolioChart() {
  if (portfolioChartRef.value) {
    portfolioChart = new Chart(portfolioChartRef.value, {
      type: 'doughnut',
      data: {
        labels: [],
        datasets: [{
          data: [],
          backgroundColor: [
            '#3b82f6',
            '#8b5cf6',
            '#ec4899',
            '#10b981',
            '#f59e0b',
            '#f97316',
            '#84cc16',
            '#06b6d4'
          ],
          borderWidth: 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              padding: 20,
              usePointStyle: true
            }
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                const total = context.dataset.data.reduce((a, b) => a + b, 0)
                const percentage = ((context.parsed / total) * 100).toFixed(1)
                return `${context.label}: ¥${context.parsed.toLocaleString()} (${percentage}%)`
              }
            }
          }
        }
      }
    })
    updatePortfolioChart()
  }
}

function initEquityChart() {
  if (equityChartRef.value) {
    equityChart = new Chart(equityChartRef.value, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: '资产净值',
          data: [],
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          fill: true,
          tension: 0.4,
          pointRadius: 0,
          pointHoverRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: false
          },
          tooltip: {
            mode: 'index',
            intersect: false,
            callbacks: {
              label: function(context) {
                return `资产净值: ¥${context.parsed.y.toLocaleString()}`
              }
            }
          }
        },
        scales: {
          x: {
            grid: {
              display: false
            }
          },
          y: {
            beginAtZero: false,
            grid: {
              color: 'rgba(0, 0, 0, 0.1)'
            },
            ticks: {
              callback: function(value) {
                return '¥' + (value / 10000).toFixed(0) + '万'
              }
            }
          }
        },
        interaction: {
          intersect: false,
          mode: 'index'
        }
      }
    })
    updateEquityChart()
  }
}

function updatePortfolioChart() {
  if (!portfolioChart || positions.value.length === 0) return

  portfolioChart.data.labels = positions.value.map(p => p.name)
  portfolioChart.data.datasets[0].data = positions.value.map(p => p.market_value)
  portfolioChart.update()
}

function updateEquityChart() {
  if (!equityChart) return

  // 根据选择的时间周期生成模拟数据
  const data = generateEquityData(chartPeriod.value)
  equityChart.data.labels = data.labels
  equityChart.data.datasets[0].data = data.values
  equityChart.update()
}

function generateEquityData(period) {
  const baseValue = 1000000
  const days = period === '1M' ? 30 : period === '3M' ? 90 : 365
  const labels = []
  const values = []
  
  for (let i = 0; i < days; i++) {
    const date = new Date()
    date.setDate(date.getDate() - (days - i))
    labels.push(date.toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit' }))
    
    // 模拟价格波动
    const randomChange = (Math.random() - 0.5) * 0.02
    const value = baseValue * (1 + randomChange * (i + 1) / days)
    values.push(Math.max(value, baseValue * 0.8))
  }
  
  return { labels, values }
}

// 工具函数
function formatNumber(value, decimals = 0) {
  if (value === null || value === undefined) return '0'
  return Number(value).toLocaleString('zh-CN', { 
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals 
  })
}

function formatPercent(value) {
  if (value === null || value === undefined) return '0.00%'
  return (Number(value) * 100).toFixed(2) + '%'
}

function formatTime(time) {
  return new Date(time).toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  })
}

// 风险指标颜色
function getSharpeColor(value) {
  if (value >= 2) return 'success'
  if (value >= 1) return 'primary'
  if (value >= 0) return 'warning'
  return 'error'
}

function getSharpeIcon(value) {
  if (value >= 2) return 'mdi-trending-up'
  if (value >= 1) return 'mdi-trending-neutral'
  if (value >= 0) return 'mdi-trending-down'
  return 'mdi-alert'
}

function getSharpeLabel(value) {
  if (value >= 2) return '优秀'
  if (value >= 1) return '良好'
  if (value >= 0) return '一般'
  return '较差'
}

function getDrawdownColor(value) {
  if (value <= 5) return 'success'
  if (value <= 10) return 'warning'
  if (value <= 20) return 'error'
  return 'error'
}

function getDrawdownIcon(value) {
  if (value <= 5) return 'mdi-check'
  if (value <= 10) return 'mdi-alert'
  return 'mdi-alert-circle'
}

function getDrawdownLabel(value) {
  if (value <= 5) return '低风险'
  if (value <= 10) return '中风险'
  if (value <= 20) return '高风险'
  return '极高风险'
}

function getRiskColor(value) {
  if (value <= 2) return 'text-success'
  if (value <= 5) return 'text-warning'
  return 'text-error'
}

function getVolatilityColor(value) {
  if (value <= 10) return 'success'
  if (value <= 20) return 'warning'
  if (value <= 30) return 'error'
  return 'error'
}

// 事件处理

async function loadMarketData() {
  marketLoading.value = true
  try {
    // 模拟API调用
    await new Promise(resolve => setTimeout(resolve, 1000))
    // 这里应该调用实际的API
  } catch (error) {
    console.error('加载市场数据失败:', error)
  } finally {
    marketLoading.value = false
  }
}

async function refreshMarketData() {
  await loadMarketData()
}


function exportChart(type) {
  // 实现图表导出功能
  console.log('导出图表:', type)
}

function viewFullChart(type) {
  // 实现全屏查看图表功能
  console.log('全屏查看图表:', type)
}

function viewDetails(type) {
  // 实现查看详情功能
  console.log('查看详情:', type)
}

function setAlert(type) {
  // 实现设置预警功能
  console.log('设置预警:', type)
}

function viewAllTrades() {
  router.push('/dashboard/trades')
}
</script>

<style lang="scss" scoped>
.overview-view {
  max-width: 1600px;
  margin: 0 auto;
}

.metric-card {
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  cursor: pointer;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12) !important;
  }
}

.chart-card {
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  
  &:hover {
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12) !important;
  }
}

.chart-container {
  height: 300px;
  position: relative;
}

.market-overview-card {
  .index-card {
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    
    &:hover {
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
    }
  }
}

// 响应式调整
@media (max-width: 960px) {
  .overview-view {
    padding: 1rem !important;
  }
  
  .chart-container {
    height: 250px;
  }
}

@media (max-width: 600px) {
  .chart-container {
    height: 200px;
  }
}

// 自定义滚动条
:deep(.v-list) {
  &::-webkit-scrollbar {
    width: 4px;
  }
  
  &::-webkit-scrollbar-track {
    background: transparent;
  }
  
  &::-webkit-scrollbar-thumb {
    background: rgba(0, 0, 0, 0.2);
    border-radius: 2px;
  }
  
  &::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 0, 0, 0.3);
  }
}

// 动画效果
@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.metric-card,
.chart-card {
  animation: fadeInUp 0.6s ease-out;
}

// 延迟动画
.metric-card:nth-child(1) { animation-delay: 0.1s; }
.metric-card:nth-child(2) { animation-delay: 0.2s; }
.metric-card:nth-child(3) { animation-delay: 0.3s; }
.metric-card:nth-child(4) { animation-delay: 0.4s; }
</style>

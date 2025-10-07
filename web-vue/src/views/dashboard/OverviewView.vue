<template>
  <v-container fluid class="overview-view pa-6">
    <div class="mb-6">
      <h1 class="text-h3 font-weight-bold mb-2">仪表盘概览</h1>
      <p class="text-body-1 text-medium-emphasis">实时监控您的投资组合表现</p>
    </div>

    <div v-if="dashboardStore.loading && !metrics" class="text-center py-10">
      <v-progress-circular indeterminate color="primary" size="64"></v-progress-circular>
      <p class="mt-4 text-body-1">加载数据中...</p>
    </div>

    <div v-else>
      <!-- 关键指标卡片 - Material 3 风格 -->
      <v-row class="mb-6">
        <v-col cols="12" sm="6" md="3">
          <v-card variant="flat" class="bg-primary-container">
            <v-card-text class="pa-6">
              <v-icon size="48" color="primary" class="mb-4">mdi-wallet</v-icon>
              <div class="text-caption mb-1">总资产</div>
              <div class="text-h4 font-weight-bold mb-3">
                ¥{{ metrics.total_assets?.toLocaleString() || 0 }}
              </div>
              <v-chip size="small" color="success" variant="flat" class="bg-success-lighten-4">
                <v-icon start size="16">mdi-trending-up</v-icon>
                +2.5%
              </v-chip>
            </v-card-text>
          </v-card>
        </v-col>

        <v-col cols="12" sm="6" md="3">
          <v-card variant="flat" class="bg-secondary-container">
            <v-card-text class="pa-6">
              <v-icon size="48" color="secondary" class="mb-4">mdi-trending-up</v-icon>
              <div class="text-caption mb-1">日收益</div>
              <div class="text-h4 font-weight-bold mb-3">
                ¥{{ metrics.daily_return?.toLocaleString() || 0 }}
              </div>
              <v-chip size="small" color="success" variant="flat" class="bg-success-lighten-4">
                <v-icon start size="16">mdi-trending-up</v-icon>
                +5.2%
              </v-chip>
            </v-card-text>
          </v-card>
        </v-col>

        <v-col cols="12" sm="6" md="3">
          <v-card variant="flat" class="bg-tertiary-container">
            <v-card-text class="pa-6">
              <v-icon size="48" color="tertiary" class="mb-4">mdi-chart-line</v-icon>
              <div class="text-caption mb-1">夏普比率</div>
              <div class="text-h4 font-weight-bold mb-3">
                {{ metrics.sharpe_ratio?.toFixed(2) || '0.00' }}
              </div>
              <v-chip size="small" color="tertiary" variant="flat" class="bg-tertiary-lighten-4" style="visibility: hidden;">
                <v-icon start size="16">mdi-trending-neutral</v-icon>
                placeholder
              </v-chip>
            </v-card-text>
          </v-card>
        </v-col>

        <v-col cols="12" sm="6" md="3">
          <v-card variant="flat" class="bg-error-container">
            <v-card-text class="pa-6">
              <v-icon size="48" color="error" class="mb-4">mdi-arrow-down</v-icon>
              <div class="text-caption mb-1">最大回撤</div>
              <div class="text-h4 font-weight-bold mb-3">
                {{ metrics.max_drawdown?.toFixed(2) || '0.00' }}%
              </div>
              <v-chip size="small" color="error" variant="flat" class="bg-error-lighten-4">
                <v-icon start size="16">mdi-trending-down</v-icon>
                -1.2%
              </v-chip>
            </v-card-text>
          </v-card>
        </v-col>
      </v-row>

      <!-- 图表区域 - Material 3 风格 -->
      <v-row>
        <!-- 投资组合分布 -->
        <v-col cols="12" md="6">
          <v-card variant="elevated">
            <v-card-title class="text-h6 font-weight-bold d-flex align-center pa-6">
              <v-avatar color="primary" variant="tonal" size="40" class="mr-3">
                <v-icon>mdi-chart-donut</v-icon>
              </v-avatar>
              投资组合分布
            </v-card-title>
            <v-card-text class="px-6 pb-6">
              <div class="chart-container">
                <canvas ref="portfolioChartRef"></canvas>
              </div>
            </v-card-text>
          </v-card>
        </v-col>

        <!-- 收益曲线 -->
        <v-col cols="12" md="6">
          <v-card variant="elevated">
            <v-card-title class="text-h6 font-weight-bold d-flex align-center pa-6">
              <v-avatar color="success" variant="tonal" size="40" class="mr-3">
                <v-icon>mdi-chart-areaspline</v-icon>
              </v-avatar>
              收益曲线
            </v-card-title>
            <v-card-text class="px-6 pb-6">
              <div class="chart-container">
                <canvas ref="equityChartRef"></canvas>
              </div>
            </v-card-text>
          </v-card>
        </v-col>

        <!-- 最近交易 -->
        <v-col cols="12">
          <v-card variant="elevated">
            <v-card-title class="text-h6 font-weight-bold d-flex align-center pa-6">
              <v-avatar color="primary" variant="tonal" size="40" class="mr-3">
                <v-icon>mdi-history</v-icon>
              </v-avatar>
              最近交易
            </v-card-title>
            <v-card-text class="pa-0">
              <v-table>
                <thead>
                  <tr>
                    <th class="text-subtitle-2">时间</th>
                    <th class="text-subtitle-2">股票</th>
                    <th class="text-subtitle-2">操作</th>
                    <th class="text-subtitle-2">价格</th>
                    <th class="text-subtitle-2">数量</th>
                    <th class="text-subtitle-2">状态</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="trade in recentTrades.slice(0, 5)" :key="trade.time">
                    <td>{{ formatTime(trade.time) }}</td>
                    <td class="font-weight-medium">{{ trade.name }}</td>
                    <td>
                      <v-chip 
                        :color="trade.action === 'BUY' ? 'success' : 'error'" 
                        size="small"
                        variant="tonal"
                      >
                        {{ trade.action === 'BUY' ? '买入' : '卖出' }}
                      </v-chip>
                    </td>
                    <td class="font-weight-bold">¥{{ trade.price.toFixed(2) }}</td>
                    <td>{{ trade.quantity }}</td>
                    <td>
                      <v-chip color="primary" size="small" variant="tonal">{{ trade.status }}</v-chip>
                    </td>
                  </tr>
                </tbody>
              </v-table>
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
import Chart from 'chart.js/auto'

const dashboardStore = useDashboardStore()

const portfolioChartRef = ref(null)
const equityChartRef = ref(null)

const metrics = computed(() => dashboardStore.metrics)
const positions = computed(() => dashboardStore.positions)
const recentTrades = computed(() => dashboardStore.recentTrades)

let portfolioChart = null
let equityChart = null

onMounted(async () => {
  await dashboardStore.refreshAll()
  initCharts()
})

watch(positions, () => {
  updatePortfolioChart()
})

function initCharts() {
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
            '#f59e0b'
          ]
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom'
          }
        }
      }
    })
    updatePortfolioChart()
  }

  if (equityChartRef.value) {
    equityChart = new Chart(equityChartRef.value, {
      type: 'line',
      data: {
        labels: ['1月', '2月', '3月', '4月', '5月', '6月'],
        datasets: [{
          label: '资产净值',
          data: [1000000, 1050000, 1080000, 1120000, 1150000, 1200000],
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          fill: true,
          tension: 0.4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: false
          }
        },
        scales: {
          y: {
            beginAtZero: false
          }
        }
      }
    })
  }
}

function updatePortfolioChart() {
  if (!portfolioChart || positions.value.length === 0) return

  portfolioChart.data.labels = positions.value.map(p => p.name)
  portfolioChart.data.datasets[0].data = positions.value.map(p => p.market_value)
  portfolioChart.update()
}

function formatTime(time) {
  return new Date(time).toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  })
}
</script>

<style lang="scss" scoped>
.chart-container {
  height: 300px;
  position: relative;
}
</style>

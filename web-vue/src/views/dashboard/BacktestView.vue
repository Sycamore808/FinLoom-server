<template>
  <v-container fluid class="backtest-view pa-6">
    <div class="mb-6">
      <h1 class="text-h3 font-weight-bold mb-2">策略回测</h1>
      <p class="text-body-1 text-medium-emphasis">测试您的交易策略历史表现</p>
    </div>

    <v-row>
      <v-col cols="12" md="4">
        <v-card variant="elevated">
          <v-card-title class="d-flex align-center pa-4">
            <v-avatar color="primary" variant="tonal" size="40" class="mr-3">
              <v-icon>mdi-cog</v-icon>
            </v-avatar>
            <span class="text-h6 font-weight-bold">回测配置</span>
          </v-card-title>
          <v-card-text>
            <v-form @submit.prevent="runBacktest">
              <v-text-field
                v-model="config.symbol"
                label="股票代码"
                prepend-inner-icon="mdi-chart-candlestick"
                variant="outlined"
                density="comfortable"
                class="mb-4"
                placeholder="例如: 000001"
              ></v-text-field>

              <v-select
                v-model="config.strategy"
                :items="[
                  { title: '移动平均线', value: 'sma' },
                  { title: 'RSI策略', value: 'rsi' },
                  { title: '布林带策略', value: 'bollinger' }
                ]"
                label="策略类型"
                prepend-inner-icon="mdi-strategy"
                variant="outlined"
                density="comfortable"
                class="mb-4"
              ></v-select>

              <v-text-field
                v-model="config.start_date"
                label="开始日期"
                type="date"
                prepend-inner-icon="mdi-calendar-start"
                variant="outlined"
                density="comfortable"
                class="mb-4"
              ></v-text-field>

              <v-text-field
                v-model="config.end_date"
                label="结束日期"
                type="date"
                prepend-inner-icon="mdi-calendar-end"
                variant="outlined"
                density="comfortable"
                class="mb-4"
              ></v-text-field>

              <v-text-field
                v-model.number="config.initial_capital"
                label="初始资金"
                type="number"
                prepend-inner-icon="mdi-currency-usd"
                variant="outlined"
                density="comfortable"
                class="mb-4"
                step="1000"
              ></v-text-field>

              <v-btn
                type="submit"
                color="primary"
                size="large"
                block
                :loading="loading"
                prepend-icon="mdi-play"
              >
                开始回测
              </v-btn>
            </v-form>
          </v-card-text>
        </v-card>
      </v-col>

      <v-col cols="12" md="8">
        <v-card v-if="result" variant="elevated">
          <v-card-title class="d-flex align-center pa-4">
            <v-avatar color="success" variant="tonal" size="40" class="mr-3">
              <v-icon>mdi-chart-areaspline</v-icon>
            </v-avatar>
            <span class="text-h6 font-weight-bold">回测结果</span>
          </v-card-title>
          <v-card-text>
            <v-row class="mb-4">
              <v-col cols="12" sm="6" md="3">
                <v-card variant="flat" class="bg-primary-container">
                  <v-card-text class="text-center pa-4">
                    <div class="text-h5 font-weight-bold">{{ result.total_return?.toFixed(2) }}%</div>
                    <div class="text-caption">总收益率</div>
                  </v-card-text>
                </v-card>
              </v-col>
              <v-col cols="12" sm="6" md="3">
                <v-card variant="flat" class="bg-secondary-container">
                  <v-card-text class="text-center pa-4">
                    <div class="text-h5 font-weight-bold">{{ result.annualized_return?.toFixed(2) }}%</div>
                    <div class="text-caption">年化收益</div>
                  </v-card-text>
                </v-card>
              </v-col>
              <v-col cols="12" sm="6" md="3">
                <v-card variant="flat" class="bg-tertiary-container">
                  <v-card-text class="text-center pa-4">
                    <div class="text-h5 font-weight-bold">{{ result.sharpe_ratio?.toFixed(2) }}</div>
                    <div class="text-caption">夏普比率</div>
                  </v-card-text>
                </v-card>
              </v-col>
              <v-col cols="12" sm="6" md="3">
                <v-card variant="flat" class="bg-error-container">
                  <v-card-text class="text-center pa-4">
                    <div class="text-h5 font-weight-bold">{{ result.max_drawdown?.toFixed(2) }}%</div>
                    <div class="text-caption">最大回撤</div>
                  </v-card-text>
                </v-card>
              </v-col>
            </v-row>

            <v-row class="mb-4">
              <v-col cols="6" md="3">
                <div class="text-center">
                  <div class="text-h6 font-weight-bold">{{ (result.win_rate * 100).toFixed(2) }}%</div>
                  <div class="text-caption text-medium-emphasis">胜率</div>
                </div>
              </v-col>
              <v-col cols="6" md="3">
                <div class="text-center">
                  <div class="text-h6 font-weight-bold">{{ result.total_trades }}</div>
                  <div class="text-caption text-medium-emphasis">总交易次数</div>
                </div>
              </v-col>
              <v-col cols="6" md="3">
                <div class="text-center">
                  <div class="text-h6 font-weight-bold text-success">{{ result.winning_trades }}</div>
                  <div class="text-caption text-medium-emphasis">盈利交易</div>
                </div>
              </v-col>
              <v-col cols="6" md="3">
                <div class="text-center">
                  <div class="text-h6 font-weight-bold text-error">{{ result.losing_trades }}</div>
                  <div class="text-caption text-medium-emphasis">亏损交易</div>
                </div>
              </v-col>
            </v-row>

            <h4 class="text-h6 mb-3">资金曲线</h4>
            <div style="height: 300px; position: relative;">
              <canvas ref="equityChartRef"></canvas>
            </div>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script setup>
import { ref } from 'vue'
import { api } from '@/services/api'
import Chart from 'chart.js/auto'

const config = ref({
  symbol: '000001',
  strategy: 'sma',
  start_date: '2023-01-01',
  end_date: '2023-12-31',
  initial_capital: 1000000
})

const loading = ref(false)
const result = ref(null)
const equityChartRef = ref(null)
let equityChart = null

async function runBacktest() {
  loading.value = true
  result.value = null

  try {
    const response = await api.backtest.run(config.value)
    result.value = response.data
    await new Promise(resolve => setTimeout(resolve, 100))
    renderEquityChart()
  } catch (error) {
    console.error('回测失败:', error)
  } finally {
    loading.value = false
  }
}

function renderEquityChart() {
  if (!equityChartRef.value || !result.value?.equity_curve) return

  if (equityChart) {
    equityChart.destroy()
  }

  const dates = result.value.equity_curve.map(d => d.date)
  const values = result.value.equity_curve.map(d => d.value)

  equityChart = new Chart(equityChartRef.value, {
    type: 'line',
    data: {
      labels: dates,
      datasets: [{
        label: '资金净值',
        data: values,
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
      }
    }
  })
}
</script>

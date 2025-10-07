<template>
  <v-container fluid class="market-view pa-6">
    <div class="mb-6">
      <h1 class="text-h3 font-weight-bold mb-2">市场分析</h1>
      <p class="text-body-1 text-medium-emphasis">实时市场概览与分析</p>
    </div>

    <div v-if="loading && !marketData" class="text-center py-10">
      <v-progress-circular indeterminate color="primary" size="64"></v-progress-circular>
      <p class="mt-4 text-body-1">加载市场数据...</p>
    </div>

    <div v-else>
      <h3 class="text-h5 font-weight-bold mb-4">市场指数</h3>
      <v-row class="mb-6">
        <v-col v-for="index in marketData?.indices" :key="index.symbol" cols="12" sm="6" md="3">
          <v-card variant="elevated" hover>
            <v-card-text class="text-center pa-6">
              <div class="text-caption text-medium-emphasis mb-2">{{ index.name }}</div>
              <div class="text-h4 font-weight-bold mb-3">{{ index.value.toFixed(2) }}</div>
              <v-chip
                :color="index.change > 0 ? 'success' : 'error'"
                size="small"
                variant="flat"
                :class="index.change > 0 ? 'bg-success-lighten-4' : 'bg-error-lighten-4'"
              >
                <v-icon start :icon="index.change > 0 ? 'mdi-arrow-up' : 'mdi-arrow-down'"></v-icon>
                {{ index.change.toFixed(2) }} ({{ (index.change_pct * 100).toFixed(2) }}%)
              </v-chip>
            </v-card-text>
          </v-card>
        </v-col>
      </v-row>

      <v-card variant="elevated" class="mb-6">
        <v-card-title class="d-flex align-center pa-6">
          <v-avatar color="primary" variant="tonal" size="40" class="mr-3">
            <v-icon>mdi-fire</v-icon>
          </v-avatar>
          <span class="text-h6 font-weight-bold">热门股票</span>
        </v-card-title>
        <v-card-text class="pa-0">
          <v-table>
            <thead>
              <tr>
                <th>股票</th>
                <th>价格</th>
                <th>涨跌</th>
                <th>涨跌幅</th>
                <th>成交量</th>
                <th>板块</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="stock in marketData?.hot_stocks" :key="stock.symbol">
                <td>
                  <div>
                    <div class="font-weight-bold">{{ stock.name }}</div>
                    <div class="text-caption text-medium-emphasis">{{ stock.symbol }}</div>
                  </div>
                </td>
                <td class="font-weight-bold">¥{{ stock.price.toFixed(2) }}</td>
                <td :class="stock.change > 0 ? 'text-success' : 'text-error'" class="font-weight-bold">
                  {{ stock.change > 0 ? '+' : '' }}{{ stock.change.toFixed(2) }}
                </td>
                <td :class="stock.change_pct > 0 ? 'text-success' : 'text-error'" class="font-weight-bold">
                  {{ stock.change_pct > 0 ? '+' : '' }}{{ (stock.change_pct * 100).toFixed(2) }}%
                </td>
                <td>{{ (stock.volume / 10000).toFixed(0) }}万</td>
                <td>
                  <v-chip size="small" color="primary" variant="tonal">{{ stock.sector }}</v-chip>
                </td>
              </tr>
            </tbody>
          </v-table>
        </v-card-text>
      </v-card>

      <v-card variant="elevated">
        <v-card-title class="d-flex align-center pa-6">
          <v-avatar color="warning" variant="tonal" size="40" class="mr-3">
            <v-icon>mdi-emoticon</v-icon>
          </v-avatar>
          <span class="text-h6 font-weight-bold">市场情绪</span>
        </v-card-title>
        <v-card-text>
          <div class="mb-6">
            <div class="d-flex justify-space-between align-center mb-2">
              <span>恐慌</span>
              <span class="font-weight-bold">{{ marketData?.market_sentiment?.fear_greed_index }}</span>
              <span>贪婪</span>
            </div>
            <v-progress-linear
              :model-value="marketData?.market_sentiment?.fear_greed_index"
              height="20"
              color="primary"
              rounded
            ></v-progress-linear>
          </div>

          <v-row>
            <v-col cols="4">
              <div class="text-center">
                <div class="text-h5 font-weight-bold text-success">{{ marketData?.market_sentiment?.advancing_stocks }}</div>
                <div class="text-caption">上涨股票</div>
              </div>
            </v-col>
            <v-col cols="4">
              <div class="text-center">
                <div class="text-h5 font-weight-bold text-error">{{ marketData?.market_sentiment?.declining_stocks }}</div>
                <div class="text-caption">下跌股票</div>
              </div>
            </v-col>
            <v-col cols="4">
              <div class="text-center">
                <div class="text-h5 font-weight-bold">{{ marketData?.market_sentiment?.vix }}</div>
                <div class="text-caption">波动率指数</div>
              </div>
            </v-col>
          </v-row>
        </v-card-text>
      </v-card>
    </div>
  </v-container>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { api } from '@/services/api'

const loading = ref(false)
const marketData = ref(null)

onMounted(() => {
  loadMarketData()
})

async function loadMarketData() {
  loading.value = true
  try {
    const response = await api.market.getOverview()
    marketData.value = response.data
  } catch (error) {
    console.error('加载市场数据失败:', error)
  } finally {
    loading.value = false
  }
}
</script>

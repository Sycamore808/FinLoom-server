<template>
  <v-container fluid class="data-view pa-6">
    <div class="mb-6">
      <h1 class="text-h3 font-weight-bold mb-2">数据管理</h1>
      <p class="text-body-1 text-medium-emphasis">管理和收集市场数据</p>
    </div>

    <v-row>
      <v-col cols="12" md="4">
        <v-card variant="elevated">
          <v-card-title class="d-flex align-center pa-4">
            <v-avatar color="primary" variant="tonal" size="40" class="mr-3">
              <v-icon>mdi-download</v-icon>
            </v-avatar>
            <span class="text-h6 font-weight-bold">数据采集</span>
          </v-card-title>
          <v-card-text>
            <v-form @submit.prevent="collectData">
              <v-text-field
                v-model="collectConfig.symbol"
                label="股票代码"
                prepend-inner-icon="mdi-chart-candlestick"
                variant="outlined"
                density="comfortable"
                class="mb-4"
                placeholder="例如: 000001"
              ></v-text-field>

              <v-select
                v-model="collectConfig.period"
                :items="[
                  { title: '1年', value: '1y' },
                  { title: '2年', value: '2y' },
                  { title: '5年', value: '5y' },
                  { title: '10年', value: '10y' }
                ]"
                label="时间周期"
                prepend-inner-icon="mdi-calendar"
                variant="outlined"
                density="comfortable"
                class="mb-4"
              ></v-select>

              <v-select
                v-model="collectConfig.data_type"
                :items="[
                  { title: '日线', value: 'daily' },
                  { title: '周线', value: 'weekly' },
                  { title: '月线', value: 'monthly' }
                ]"
                label="数据类型"
                prepend-inner-icon="mdi-chart-timeline-variant"
                variant="outlined"
                density="comfortable"
                class="mb-4"
              ></v-select>

              <v-btn
                type="submit"
                color="primary"
                size="large"
                block
                :loading="collecting"
                prepend-icon="mdi-download"
              >
                开始采集
              </v-btn>
            </v-form>

            <v-alert v-if="collectResult" type="success" variant="tonal" class="mt-4">
              成功采集 {{ collectResult.records_count }} 条数据
            </v-alert>
          </v-card-text>
        </v-card>
      </v-col>

      <v-col cols="12" md="8">
        <v-card variant="elevated">
          <v-card-title class="d-flex justify-space-between align-center pa-4">
            <div class="d-flex align-center">
              <v-avatar color="primary" variant="tonal" size="40" class="mr-3">
                <v-icon>mdi-database</v-icon>
              </v-avatar>
              <span class="text-h6 font-weight-bold">数据概览</span>
            </div>
            <v-btn
              color="primary"
              prepend-icon="mdi-refresh"
              @click="loadOverview"
              size="small"
            >
              刷新
            </v-btn>
          </v-card-title>
          
          <v-card-text v-if="loadingOverview" class="text-center py-10">
            <v-progress-circular indeterminate color="primary"></v-progress-circular>
            <p class="mt-4">加载中...</p>
          </v-card-text>

          <v-card-text v-else-if="overview">
            <v-row class="mb-4">
              <v-col cols="6">
                <v-card color="primary" dark>
                  <v-card-text>
                    <div class="text-h4 font-weight-bold">{{ overview.total_symbols }}</div>
                    <div class="text-caption">股票数量</div>
                  </v-card-text>
                </v-card>
              </v-col>
              <v-col cols="6">
                <v-card color="success" dark>
                  <v-card-text>
                    <div class="text-h4 font-weight-bold">{{ overview.total_records?.toLocaleString() }}</div>
                    <div class="text-caption">数据记录</div>
                  </v-card-text>
                </v-card>
              </v-col>
            </v-row>

            <h4 class="text-h6 mb-3">最近更新</h4>
            <v-list>
              <v-list-item v-for="symbol in overview.symbols" :key="symbol.symbol">
                <v-list-item-title class="font-weight-bold">{{ symbol.name }}</v-list-item-title>
                <v-list-item-subtitle>{{ symbol.symbol }}</v-list-item-subtitle>
                <template v-slot:append>
                  <div class="text-end">
                    <div class="font-weight-bold">¥{{ symbol.latest_price }}</div>
                    <div class="text-caption">{{ symbol.records_count }} 条</div>
                  </div>
                </template>
              </v-list-item>
            </v-list>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { api } from '@/services/api'

const collectConfig = ref({
  symbol: '000001',
  period: '1y',
  data_type: 'daily'
})

const collecting = ref(false)
const collectResult = ref(null)
const loadingOverview = ref(false)
const overview = ref(null)

onMounted(() => {
  loadOverview()
})

async function collectData() {
  collecting.value = true
  collectResult.value = null

  try {
    const response = await api.data.collect(collectConfig.value)
    collectResult.value = response.data
    setTimeout(() => { loadOverview() }, 1000)
  } catch (error) {
    console.error('数据采集失败:', error)
  } finally {
    collecting.value = false
  }
}

async function loadOverview() {
  loadingOverview.value = true
  try {
    const response = await api.data.getOverview()
    overview.value = response.data
  } catch (error) {
    console.error('加载概览失败:', error)
  } finally {
    loadingOverview.value = false
  }
}
</script>

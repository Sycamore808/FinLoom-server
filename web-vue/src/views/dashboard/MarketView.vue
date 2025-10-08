<template>
  <v-container fluid class="market-view pa-6">
    <!-- 页面头部 -->
    <div class="mb-6">
      <div class="d-flex justify-space-between align-center mb-4">
        <div>
      <h1 class="text-h3 font-weight-bold mb-2">市场分析</h1>
      <p class="text-body-1 text-medium-emphasis">实时市场概览与分析</p>
        </div>
        <div class="d-flex gap-2">
          <!-- 保留空div以维持布局 -->
        </div>
      </div>
      
      <!-- 市场状态指示器 -->
      <v-alert
        v-if="marketStatus"
        :type="marketStatus.type"
        variant="tonal"
        class="mb-4"
        rounded="lg"
      >
        <template v-slot:prepend>
          <v-icon>{{ marketStatus.icon }}</v-icon>
        </template>
        <div class="d-flex justify-space-between align-center">
          <span>{{ marketStatus.message }}</span>
          <span class="text-caption">{{ lastUpdateTime }}</span>
        </div>
      </v-alert>
    </div>

    <div v-if="loading && !marketData" class="text-center py-10">
      <v-progress-circular indeterminate color="primary" size="64"></v-progress-circular>
      <p class="mt-4 text-body-1">加载市场数据...</p>
    </div>

    <div v-else>
      <!-- 市场指数卡片 -->
      <v-card variant="elevated" class="mb-6">
        <v-card-title class="d-flex align-center justify-space-between pa-6">
          <div class="d-flex align-center">
            <v-avatar color="primary" variant="tonal" size="40" class="mr-3">
              <v-icon>mdi-chart-multiple</v-icon>
            </v-avatar>
            <div>
              <div class="text-h6 font-weight-bold">市场指数</div>
              <div class="text-caption text-medium-emphasis">主要指数实时表现</div>
            </div>
          </div>
          <v-btn-toggle v-model="indexPeriod" mandatory>
            <v-btn value="1D" size="small">1日</v-btn>
            <v-btn value="1W" size="small">1周</v-btn>
            <v-btn value="1M" size="small">1月</v-btn>
          </v-btn-toggle>
        </v-card-title>
        <v-card-text class="pa-6 pt-0">
          <v-row>
            <v-col v-for="index in marketIndices" :key="index.symbol" cols="12" sm="6" md="3">
              <v-card variant="outlined" class="index-card" hover @click="viewIndexDetail(index)">
                <v-card-text class="pa-4">
                  <div class="d-flex justify-space-between align-center mb-2">
                    <div class="text-subtitle-2 font-weight-bold">{{ index.name }}</div>
                    <v-chip size="x-small" :color="index.change >= 0 ? 'success' : 'error'" variant="tonal">
                      {{ index.change >= 0 ? '+' : '' }}{{ formatPercent(index.change_pct) }}
                    </v-chip>
                  </div>
                  <div class="text-h6 font-weight-bold mb-1">{{ formatNumber(index.value, 2) }}</div>
                  <div class="d-flex align-center justify-space-between">
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
                    <v-icon size="16" class="text-medium-emphasis">mdi-chevron-right</v-icon>
                  </div>
                </v-card-text>
              </v-card>
            </v-col>
          </v-row>
        </v-card-text>
      </v-card>

      <!-- 热门股票和板块分析 -->
      <v-row class="mb-6">
        <!-- 热门股票 -->
        <v-col cols="12" md="8">
          <v-card variant="elevated" class="h-100">
            <v-card-title class="d-flex align-center justify-space-between pa-6">
              <div class="d-flex align-center">
                <v-avatar color="success" variant="tonal" size="40" class="mr-3">
                  <v-icon>mdi-fire</v-icon>
                </v-avatar>
                <div>
                  <div class="text-h6 font-weight-bold">热门股票</div>
                  <div class="text-caption text-medium-emphasis">涨幅榜前10</div>
                </div>
              </div>
              <v-btn-toggle v-model="stockSort" mandatory>
                <v-btn value="change" size="small">涨跌幅</v-btn>
                <v-btn value="volume" size="small">成交量</v-btn>
                <v-btn value="amount" size="small">成交额</v-btn>
              </v-btn-toggle>
            </v-card-title>
            <v-card-text class="pa-0">
              <v-data-table
                :headers="stockHeaders"
                :items="sortedHotStocks"
                :items-per-page="10"
                class="elevation-0"
                hide-default-footer
              >
                <template v-slot:item.name="{ item }">
                  <div class="d-flex align-center">
                    <v-avatar size="32" class="mr-3">
                      <v-img :src="getStockLogo(item.symbol)" :alt="item.name">
                        <template v-slot:error>
                          <v-icon>mdi-chart-line</v-icon>
                        </template>
                      </v-img>
                    </v-avatar>
                    <div>
                      <div class="font-weight-bold">{{ item.name }}</div>
                      <div class="text-caption text-medium-emphasis">{{ item.symbol }}</div>
                    </div>
                  </div>
                </template>
                
                <template v-slot:item.price="{ item }">
                  <div class="font-weight-bold">¥{{ formatNumber(item.price, 2) }}</div>
                </template>
                
                <template v-slot:item.change="{ item }">
                  <div :class="item.change >= 0 ? 'text-success' : 'text-error'" class="font-weight-bold">
                    {{ item.change >= 0 ? '+' : '' }}{{ formatNumber(item.change, 2) }}
                  </div>
                </template>
                
                <template v-slot:item.change_pct="{ item }">
                  <v-chip 
                    :color="item.change_pct >= 0 ? 'success' : 'error'" 
                    size="small" 
                    variant="tonal"
                  >
                    {{ item.change_pct >= 0 ? '+' : '' }}{{ formatPercent(item.change_pct) }}
                  </v-chip>
                </template>
                
                <template v-slot:item.volume="{ item }">
                  <div class="text-body-2">{{ formatVolume(item.volume) }}</div>
                </template>
                
                <template v-slot:item.sector="{ item }">
                  <v-chip size="small" :color="getSectorColor(item.sector)" variant="tonal">
                    {{ item.sector }}
              </v-chip>
                </template>
                
                <template v-slot:item.actions="{ item }">
                  <v-btn
                    icon="mdi-chart-line"
                    variant="text"
                    size="small"
                    @click="viewStockDetail(item)"
                  ></v-btn>
                </template>
              </v-data-table>
            </v-card-text>
          </v-card>
        </v-col>

        <!-- 板块分析 -->
        <v-col cols="12" md="4">
          <v-card variant="elevated" class="h-100">
            <v-card-title class="d-flex align-center pa-6">
              <v-avatar color="info" variant="tonal" size="40" class="mr-3">
                <v-icon>mdi-chart-pie</v-icon>
              </v-avatar>
              <div>
                <div class="text-h6 font-weight-bold">板块分析</div>
                <div class="text-caption text-medium-emphasis">行业表现</div>
              </div>
            </v-card-title>
            <v-card-text class="pa-6">
              <v-list density="compact">
                <v-list-item
                  v-for="sector in sectorPerformance"
                  :key="sector.name"
                  class="px-0"
                >
                  <template v-slot:prepend>
                    <v-avatar :color="sector.color" size="24" variant="tonal">
                      <v-icon size="14">{{ sector.icon }}</v-icon>
                    </v-avatar>
                  </template>
                  <v-list-item-title class="text-body-2">{{ sector.name }}</v-list-item-title>
                  <template v-slot:append>
                    <div class="text-end">
                      <div :class="sector.change >= 0 ? 'text-success' : 'text-error'" class="font-weight-bold">
                        {{ sector.change >= 0 ? '+' : '' }}{{ formatPercent(sector.change) }}
                      </div>
                      <div class="text-caption text-medium-emphasis">{{ sector.count }}只股票</div>
                    </div>
                  </template>
                </v-list-item>
              </v-list>
            </v-card-text>
          </v-card>
        </v-col>
      </v-row>

      <!-- 市场情绪和技术指标 -->
      <v-row class="mb-6">
        <!-- 市场情绪 -->
        <v-col cols="12" md="6">
          <v-card variant="elevated" class="h-100">
        <v-card-title class="d-flex align-center pa-6">
          <v-avatar color="warning" variant="tonal" size="40" class="mr-3">
            <v-icon>mdi-emoticon</v-icon>
          </v-avatar>
              <div>
                <div class="text-h6 font-weight-bold">市场情绪</div>
                <div class="text-caption text-medium-emphasis">恐慌贪婪指数</div>
              </div>
        </v-card-title>
            <v-card-text class="pa-6">
          <div class="mb-6">
            <div class="d-flex justify-space-between align-center mb-2">
                  <span class="text-body-2">恐慌</span>
                  <span class="font-weight-bold text-h6">{{ marketSentiment.fear_greed_index }}</span>
                  <span class="text-body-2">贪婪</span>
            </div>
            <v-progress-linear
                  :model-value="marketSentiment.fear_greed_index"
              height="20"
                  :color="getSentimentColor(marketSentiment.fear_greed_index)"
              rounded
            ></v-progress-linear>
                <div class="text-center mt-2">
                  <v-chip :color="getSentimentColor(marketSentiment.fear_greed_index)" size="small" variant="tonal">
                    {{ getSentimentLabel(marketSentiment.fear_greed_index) }}
                  </v-chip>
                </div>
          </div>

          <v-row>
                <v-col cols="6">
              <div class="text-center">
                    <div class="text-h5 font-weight-bold text-success">{{ marketSentiment.advancing_stocks }}</div>
                <div class="text-caption">上涨股票</div>
              </div>
            </v-col>
                <v-col cols="6">
              <div class="text-center">
                    <div class="text-h5 font-weight-bold text-error">{{ marketSentiment.declining_stocks }}</div>
                <div class="text-caption">下跌股票</div>
              </div>
            </v-col>
              </v-row>
            </v-card-text>
          </v-card>
        </v-col>

        <!-- 技术指标 -->
        <v-col cols="12" md="6">
          <v-card variant="elevated" class="h-100">
            <v-card-title class="d-flex align-center pa-6">
              <v-avatar color="info" variant="tonal" size="40" class="mr-3">
                <v-icon>mdi-chart-line-variant</v-icon>
              </v-avatar>
              <div>
                <div class="text-h6 font-weight-bold">技术指标</div>
                <div class="text-caption text-medium-emphasis">市场技术分析</div>
              </div>
            </v-card-title>
            <v-card-text class="pa-6">
              <v-list density="compact">
                <v-list-item
                  v-for="indicator in technicalIndicators"
                  :key="indicator.name"
                  class="px-0"
                >
                  <template v-slot:prepend>
                    <v-avatar :color="indicator.color" size="32" variant="tonal">
                      <v-icon size="16">{{ indicator.icon }}</v-icon>
                    </v-avatar>
                  </template>
                  <v-list-item-title class="text-body-2 font-weight-medium">{{ indicator.name }}</v-list-item-title>
                  <v-list-item-subtitle class="text-caption">{{ indicator.description }}</v-list-item-subtitle>
                  <template v-slot:append>
                    <div class="text-end">
                      <div :class="indicator.value >= 0 ? 'text-success' : 'text-error'" class="font-weight-bold">
                        {{ indicator.value >= 0 ? '+' : '' }}{{ formatNumber(indicator.value, 2) }}
                      </div>
                      <div class="text-caption text-medium-emphasis">{{ indicator.signal }}</div>
                    </div>
                  </template>
                </v-list-item>
              </v-list>
            </v-card-text>
          </v-card>
            </v-col>
          </v-row>

      <!-- 市场新闻和公告 -->
      <v-card variant="elevated" class="mb-6">
        <v-card-title class="d-flex align-center justify-space-between pa-6">
          <div class="d-flex align-center">
            <v-avatar color="primary" variant="tonal" size="40" class="mr-3">
              <v-icon>mdi-newspaper</v-icon>
            </v-avatar>
            <div>
              <div class="text-h6 font-weight-bold">市场资讯</div>
              <div class="text-caption text-medium-emphasis">最新市场动态</div>
            </div>
          </div>
          <v-btn
            color="primary"
            variant="text"
            size="small"
            @click="viewAllNews"
            prepend-icon="mdi-arrow-right"
          >
            查看更多
          </v-btn>
        </v-card-title>
        <v-card-text class="pa-0">
          <v-list>
            <v-list-item
              v-for="news in marketNews"
              :key="news.id"
              class="px-6"
            >
              <template v-slot:prepend>
                <v-avatar :color="news.type === 'important' ? 'error' : 'primary'" size="32" variant="tonal">
                  <v-icon size="16">{{ news.type === 'important' ? 'mdi-alert' : 'mdi-newspaper' }}</v-icon>
                </v-avatar>
              </template>
              <v-list-item-title class="font-weight-medium">{{ news.title }}</v-list-item-title>
              <v-list-item-subtitle>{{ news.summary }}</v-list-item-subtitle>
              <template v-slot:append>
                <div class="text-end">
                  <div class="text-caption text-medium-emphasis">{{ formatTime(news.time) }}</div>
                  <v-chip size="x-small" :color="news.type === 'important' ? 'error' : 'primary'" variant="tonal">
                    {{ news.type === 'important' ? '重要' : '资讯' }}
                  </v-chip>
                </div>
              </template>
            </v-list-item>
          </v-list>
        </v-card-text>
      </v-card>
    </div>
  </v-container>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { api } from '@/services/api'

const router = useRouter()
const loading = ref(false)
const marketData = ref(null)
const indexPeriod = ref('1D')
const stockSort = ref('change')

// 市场状态
const marketStatus = ref({
  type: 'success',
  icon: 'mdi-check-circle',
  message: '市场正常运行'
})

// 市场指数数据
const marketIndices = ref([
  { symbol: 'SH000001', name: '上证指数', value: 3000.25, change: 15.32, change_pct: 0.51 },
  { symbol: 'SZ399001', name: '深证成指', value: 9500.15, change: -25.18, change_pct: -0.26 },
  { symbol: 'SZ399006', name: '创业板指', value: 1850.45, change: 8.75, change_pct: 0.48 },
  { symbol: 'HKHSI', name: '恒生指数', value: 16500.30, change: 120.45, change_pct: 0.73 }
])

// 热门股票数据
const hotStocks = ref([
  { symbol: '000001', name: '平安银行', price: 12.45, change: 0.85, change_pct: 0.73, volume: 12500000, sector: '银行' },
  { symbol: '000002', name: '万科A', price: 18.32, change: -0.45, change_pct: -0.24, volume: 8900000, sector: '房地产' },
  { symbol: '000858', name: '五粮液', price: 156.80, change: 3.20, change_pct: 2.08, volume: 5600000, sector: '食品饮料' },
  { symbol: '002415', name: '海康威视', price: 45.60, change: 1.20, change_pct: 2.70, volume: 7800000, sector: '电子' },
  { symbol: '300059', name: '东方财富', price: 23.15, change: -0.35, change_pct: -1.49, volume: 15200000, sector: '金融科技' }
])

// 板块表现数据
const sectorPerformance = ref([
  { name: '科技', change: 2.5, count: 156, color: 'primary', icon: 'mdi-laptop' },
  { name: '医药', change: 1.8, count: 89, color: 'success', icon: 'mdi-medical-bag' },
  { name: '金融', change: -0.5, count: 45, color: 'info', icon: 'mdi-bank' },
  { name: '消费', change: 1.2, count: 78, color: 'warning', icon: 'mdi-shopping' },
  { name: '能源', change: -1.8, count: 34, color: 'error', icon: 'mdi-lightning-bolt' }
])

// 市场情绪数据
const marketSentiment = ref({
  fear_greed_index: 65,
  advancing_stocks: 1256,
  declining_stocks: 834
})

// 技术指标数据
const technicalIndicators = ref([
  { name: 'RSI', value: 58.5, signal: '中性', color: 'primary', icon: 'mdi-chart-line', description: '相对强弱指数' },
  { name: 'MACD', value: 0.25, signal: '买入', color: 'success', icon: 'mdi-chart-areaspline', description: '移动平均收敛散度' },
  { name: 'KDJ', value: 72.3, signal: '超买', color: 'warning', icon: 'mdi-chart-scatter-plot', description: '随机指标' },
  { name: 'BOLL', value: 1.05, signal: '突破', color: 'info', icon: 'mdi-chart-box', description: '布林带指标' }
])

// 市场新闻数据
const marketNews = ref([
  { id: 1, title: '央行宣布降准0.5个百分点', summary: '为支持实体经济发展，央行决定下调存款准备金率', time: new Date(), type: 'important' },
  { id: 2, title: '科技股集体上涨', summary: '人工智能概念股表现强劲，多只股票涨停', time: new Date(Date.now() - 3600000), type: 'normal' },
  { id: 3, title: '新能源汽车销量创新高', summary: '11月新能源汽车销量同比增长35%', time: new Date(Date.now() - 7200000), type: 'normal' }
])

// 表格头部
const stockHeaders = [
  { title: '股票', key: 'name', sortable: false },
  { title: '价格', key: 'price', sortable: true },
  { title: '涨跌', key: 'change', sortable: true },
  { title: '涨跌幅', key: 'change_pct', sortable: true },
  { title: '成交量', key: 'volume', sortable: true },
  { title: '板块', key: 'sector', sortable: true },
  { title: '操作', key: 'actions', sortable: false }
]

// 计算属性
const sortedHotStocks = computed(() => {
  const stocks = [...hotStocks.value]
  switch (stockSort.value) {
    case 'change':
      return stocks.sort((a, b) => b.change_pct - a.change_pct)
    case 'volume':
      return stocks.sort((a, b) => b.volume - a.volume)
    case 'amount':
      return stocks.sort((a, b) => (b.price * b.volume) - (a.price * a.volume))
    default:
      return stocks
  }
})

const lastUpdateTime = computed(() => {
  return new Date().toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  })
})

onMounted(() => {
  loadMarketData()
})

async function loadMarketData() {
  loading.value = true
  try {
    // 模拟API调用
    await new Promise(resolve => setTimeout(resolve, 1000))
    // const response = await api.market.getOverview()
    // marketData.value = response.data
  } catch (error) {
    console.error('加载市场数据失败:', error)
  } finally {
    loading.value = false
  }
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

function formatVolume(volume) {
  if (volume >= 100000000) {
    return (volume / 100000000).toFixed(1) + '亿'
  } else if (volume >= 10000) {
    return (volume / 10000).toFixed(0) + '万'
  }
  return volume.toString()
}

function formatTime(time) {
  const now = new Date()
  const diff = now - time
  const minutes = Math.floor(diff / 60000)
  const hours = Math.floor(diff / 3600000)
  
  if (minutes < 60) {
    return `${minutes}分钟前`
  } else if (hours < 24) {
    return `${hours}小时前`
  } else {
    return time.toLocaleDateString('zh-CN')
  }
}

// 颜色和标签函数
function getSentimentColor(index) {
  if (index >= 70) return 'error'
  if (index >= 50) return 'warning'
  if (index >= 30) return 'primary'
  return 'success'
}

function getSentimentLabel(index) {
  if (index >= 70) return '极度贪婪'
  if (index >= 50) return '贪婪'
  if (index >= 30) return '中性'
  return '恐慌'
}

function getSectorColor(sector) {
  const colors = {
    '银行': 'primary',
    '房地产': 'warning',
    '食品饮料': 'success',
    '电子': 'info',
    '金融科技': 'secondary'
  }
  return colors[sector] || 'primary'
}

function getStockLogo(symbol) {
  // 这里应该返回实际的股票logo URL
  return `https://logo.clearbit.com/${symbol}.com`
}

// 事件处理
async function refreshData() {
  await loadMarketData()
}

function exportData() {
  // 实现数据导出功能
  console.log('导出市场数据')
}

function viewIndexDetail(index) {
  // 实现查看指数详情功能
  console.log('查看指数详情:', index)
}

function viewStockDetail(stock) {
  // 实现查看股票详情功能
  console.log('查看股票详情:', stock)
}

function viewAllNews() {
  // 实现查看所有新闻功能
  console.log('查看所有新闻')
}
</script>

<style lang="scss" scoped>
.market-view {
  max-width: 1600px;
  margin: 0 auto;
}

.index-card {
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  cursor: pointer;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12) !important;
  }
}

// 响应式调整
@media (max-width: 960px) {
  .market-view {
    padding: 1rem !important;
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

.index-card,
.v-card {
  animation: fadeInUp 0.6s ease-out;
}

// 延迟动画
.index-card:nth-child(1) { animation-delay: 0.1s; }
.index-card:nth-child(2) { animation-delay: 0.2s; }
.index-card:nth-child(3) { animation-delay: 0.3s; }
.index-card:nth-child(4) { animation-delay: 0.4s; }

// 表格样式优化
:deep(.v-data-table) {
  .v-data-table__wrapper {
    border-radius: 8px;
  }
  
  .v-data-table__tr {
    &:hover {
      background-color: rgba(var(--v-theme-primary), 0.04) !important;
    }
  }
}

// 进度条样式
:deep(.v-progress-linear) {
  border-radius: 10px;
  overflow: hidden;
}
</style>

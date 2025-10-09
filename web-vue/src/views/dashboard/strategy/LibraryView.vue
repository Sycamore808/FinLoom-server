<template>
  <v-container fluid class="library-view pa-6">
    <v-card rounded="xl">
      <!-- 头部 -->
      <v-card-title class="pa-6 d-flex align-center justify-space-between">
        <div class="d-flex align-center">
          <v-icon start color="primary" size="32">mdi-folder-multiple-outline</v-icon>
          <div>
            <div class="text-h4 font-weight-bold">策略库</div>
            <div class="text-subtitle-1 text-medium-emphasis">
              共 {{ strategies.length }} 个策略
            </div>
          </div>
        </div>
        
        <div class="d-flex align-center gap-2">
          <v-btn
            color="secondary"
            prepend-icon="mdi-view-module"
            rounded="pill"
            variant="outlined"
            @click="$router.push('/dashboard/strategy/templates')"
          >
            浏览模板
          </v-btn>
          
          <v-btn
            color="primary"
            prepend-icon="mdi-plus"
            rounded="pill"
            variant="flat"
            @click="$router.push('/dashboard/strategy/create')"
          >
            创建策略
          </v-btn>
        </div>
      </v-card-title>
      
      <v-divider></v-divider>
      
      <!-- 筛选和搜索 -->
      <v-card-text class="pa-6">
        <v-row>
          <v-col cols="12" md="8">
            <v-chip-group v-model="selectedFilter" mandatory>
              <v-chip
                v-for="filter in filters"
                :key="filter.value"
                :value="filter.value"
                rounded="lg"
              >
                <v-icon start size="18">{{ filter.icon }}</v-icon>
                {{ filter.label }}
              </v-chip>
            </v-chip-group>
          </v-col>
          
          <v-col cols="12" md="4">
            <v-text-field
              v-model="searchQuery"
              placeholder="搜索策略..."
              prepend-inner-icon="mdi-magnify"
              variant="outlined"
              density="comfortable"
              hide-details
              rounded="lg"
            ></v-text-field>
          </v-col>
        </v-row>
      </v-card-text>
      
      <v-divider></v-divider>
      
      <!-- 策略列表 -->
      <v-card-text class="pa-6">
        <v-row v-if="!loading && filteredStrategies.length > 0">
          <v-col
            v-for="strategy in filteredStrategies"
            :key="strategy.id"
            cols="12"
            md="6"
            lg="4"
          >
            <v-card
              variant="outlined"
              rounded="xl"
              class="strategy-card h-100"
            >
              <v-card-text class="pa-6">
                <div class="d-flex justify-space-between align-start mb-3">
                  <v-chip
                    size="small"
                    :color="getTypeColor(strategy.type)"
                    variant="tonal"
                    rounded="lg"
                  >
                    {{ getTypeLabel(strategy.type) }}
                  </v-chip>
                  
                  <v-menu>
                    <template v-slot:activator="{ props }">
                      <v-btn
                        icon="mdi-dots-vertical"
                        variant="text"
                        size="small"
                        v-bind="props"
                      ></v-btn>
                    </template>
                    <v-list>
                      <v-list-item @click="viewDetails(strategy)">
                        <v-list-item-title>
                          <v-icon start size="18">mdi-eye</v-icon>
                          查看详情
                        </v-list-item-title>
                      </v-list-item>
                      <v-list-item @click="backtest(strategy)">
                        <v-list-item-title>
                          <v-icon start size="18">mdi-chart-timeline-variant</v-icon>
                          回测策略
                        </v-list-item-title>
                      </v-list-item>
                      <v-list-item @click="duplicate(strategy)">
                        <v-list-item-title>
                          <v-icon start size="18">mdi-content-copy</v-icon>
                          复制策略
                        </v-list-item-title>
                      </v-list-item>
                      <v-divider></v-divider>
                      <v-list-item @click="deleteStrategy(strategy.id)" class="text-error">
                        <v-list-item-title>
                          <v-icon start size="18">mdi-delete</v-icon>
                          删除策略
                        </v-list-item-title>
                      </v-list-item>
                    </v-list>
                  </v-menu>
                </div>
                
                <div class="text-h6 font-weight-medium mb-3">
                  {{ strategy.name }}
                </div>
                
                <!-- 性能指标 -->
                <v-row v-if="strategy.performance" dense class="mb-3">
                  <v-col cols="6">
                    <div class="text-caption text-medium-emphasis">年化收益</div>
                    <div class="text-h6 text-success">
                      {{ strategy.performance.train?.sharpe_ratio || 'N/A' }}
                    </div>
                  </v-col>
                  <v-col cols="6">
                    <div class="text-caption text-medium-emphasis">夏普比率</div>
                    <div class="text-h6">1.85</div>
                  </v-col>
                </v-row>
                
                <div class="text-caption text-medium-emphasis">
                  <v-icon start size="16">mdi-clock-outline</v-icon>
                  创建于 {{ formatDate(strategy.created_at) }}
                </div>
              </v-card-text>
              
              <v-divider></v-divider>
              
              <v-card-actions class="pa-4">
                <v-btn
                  variant="tonal"
                  color="primary"
                  block
                  rounded="lg"
                  prepend-icon="mdi-play"
                  @click="runStrategy(strategy)"
                >
                  运行策略
                </v-btn>
              </v-card-actions>
            </v-card>
          </v-col>
        </v-row>
        
        <!-- 加载状态 -->
        <div v-if="loading" class="text-center py-12">
          <v-progress-circular indeterminate color="primary" size="64"></v-progress-circular>
          <div class="text-h6 mt-4">加载中...</div>
        </div>
        
        <!-- 空状态 -->
        <div v-if="!loading && filteredStrategies.length === 0" class="text-center py-12">
          <v-avatar color="surface-variant" size="96" class="mb-4">
            <v-icon size="48" color="medium-emphasis">mdi-folder-outline</v-icon>
          </v-avatar>
          <div class="text-h5 mb-2">还没有策略</div>
          <div class="text-body-1 text-medium-emphasis mb-4">
            {{ searchQuery ? '没有找到匹配的策略' : '创建您的第一个策略吧' }}
          </div>
          <v-btn
            color="primary"
            prepend-icon="mdi-plus"
            rounded="pill"
            variant="flat"
            size="large"
            @click="$router.push('/dashboard/strategy/create')"
          >
            创建策略
          </v-btn>
        </div>
      </v-card-text>
    </v-card>
    
    <!-- 详情对话框 -->
    <v-dialog v-model="detailsDialog" max-width="800">
      <v-card v-if="selectedStrategy" rounded="xl">
        <v-card-title class="pa-6">
          <v-icon start color="primary">mdi-file-document-outline</v-icon>
          {{ selectedStrategy.name }}
        </v-card-title>
        <v-divider></v-divider>
        <v-card-text class="pa-6">
          <div class="mb-4">
            <div class="text-subtitle-2 mb-2">策略类型</div>
            <v-chip :color="getTypeColor(selectedStrategy.type)" variant="tonal">
              {{ getTypeLabel(selectedStrategy.type) }}
            </v-chip>
          </div>
          
          <div class="mb-4">
            <div class="text-subtitle-2 mb-2">策略参数</div>
            <pre class="pa-4 rounded" style="background: rgba(var(--v-theme-surface-variant), 0.5); overflow-x: auto">{{ JSON.stringify(selectedStrategy.parameters, null, 2) }}</pre>
          </div>
          
          <div v-if="selectedStrategy.performance">
            <div class="text-subtitle-2 mb-2">性能指标</div>
            <v-row dense>
              <v-col cols="6">
                <v-card variant="tonal" color="success">
                  <v-card-text>
                    <div class="text-caption">训练集表现</div>
                    <div class="text-h6">优秀</div>
                  </v-card-text>
                </v-card>
              </v-col>
              <v-col cols="6">
                <v-card variant="tonal" color="info">
                  <v-card-text>
                    <div class="text-caption">测试集表现</div>
                    <div class="text-h6">良好</div>
                  </v-card-text>
                </v-card>
              </v-col>
            </v-row>
          </div>
        </v-card-text>
        <v-card-actions class="pa-6 pt-0">
          <v-spacer></v-spacer>
          <v-btn variant="text" @click="detailsDialog = false" rounded="pill">关闭</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-container>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { api } from '@/services/api'

const router = useRouter()
const strategies = ref([])
const loading = ref(false)
const searchQuery = ref('')
const selectedFilter = ref('all')
const detailsDialog = ref(false)
const selectedStrategy = ref(null)

const filters = [
  { label: '全部', value: 'all', icon: 'mdi-view-grid-outline' },
  { label: '价值投资', value: 'value', icon: 'mdi-chart-line' },
  { label: '成长投资', value: 'growth', icon: 'mdi-trending-up' },
  { label: '动量策略', value: 'momentum', icon: 'mdi-rocket-launch' },
  { label: '均值回归', value: 'mean_reversion', icon: 'mdi-chart-bell-curve' }
]

const filteredStrategies = computed(() => {
  let filtered = [...strategies.value]
  
  // 搜索过滤
  if (searchQuery.value) {
    const query = searchQuery.value.toLowerCase()
    filtered = filtered.filter(s =>
      s.name.toLowerCase().includes(query)
    )
  }
  
  // 类型过滤
  if (selectedFilter.value !== 'all') {
    filtered = filtered.filter(s => s.type === selectedFilter.value)
  }
  
  return filtered
})

onMounted(async () => {
  await loadStrategies()
})

async function loadStrategies() {
  loading.value = true
  try {
    const response = await api.strategy.list()
    strategies.value = response.data.strategies || []
  } catch (error) {
    console.error('加载策略列表失败:', error)
  } finally {
    loading.value = false
  }
}

function viewDetails(strategy) {
  selectedStrategy.value = strategy
  detailsDialog.value = true
}

function backtest(strategy) {
  router.push({
    name: 'dashboard-backtest',
    query: { strategy_id: strategy.id }
  })
}

async function duplicate(strategy) {
  try {
    await api.strategy.duplicate(strategy.id, `${strategy.name} (副本)`)
    await loadStrategies()
  } catch (error) {
    console.error('复制策略失败:', error)
  }
}

async function deleteStrategy(id) {
  if (!confirm('确定要删除这个策略吗?')) return
  
  try {
    await api.strategy.delete(id)
    strategies.value = strategies.value.filter(s => s.id !== id)
  } catch (error) {
    console.error('删除策略失败:', error)
  }
}

function runStrategy(strategy) {
  // 运行策略功能待实现
  console.log('运行策略:', strategy.name)
}

function getTypeColor(type) {
  const colors = {
    value: 'primary',
    growth: 'success',
    momentum: 'warning',
    mean_reversion: 'info',
    custom: 'secondary'
  }
  return colors[type] || 'default'
}

function getTypeLabel(type) {
  const labels = {
    value: '价值投资',
    growth: '成长投资',
    momentum: '动量策略',
    mean_reversion: '均值回归',
    custom: '自定义'
  }
  return labels[type] || type
}

function formatDate(dateString) {
  return new Date(dateString).toLocaleDateString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit'
  })
}
</script>

<style lang="scss" scoped>
.library-view {
  max-width: 1600px;
  margin: 0 auto;
}

.strategy-card {
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  
  &:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(var(--v-theme-primary), 0.12);
  }
}
</style>








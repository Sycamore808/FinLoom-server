<template>
  <v-container fluid class="portfolio-view pa-6">
    <div class="d-flex justify-space-between align-center mb-6">
      <div>
        <h1 class="text-h3 font-weight-bold mb-2">投资组合</h1>
        <p class="text-body-1 text-medium-emphasis">查看您的持仓情况</p>
      </div>
      <v-btn 
        color="primary" 
        prepend-icon="mdi-refresh" 
        size="large"
        @click="dashboardStore.fetchPositions()"
      >
        刷新
      </v-btn>
    </div>

    <v-row>
      <v-col 
        v-for="position in dashboardStore.positions" 
        :key="position.symbol"
        cols="12"
        sm="6"
        md="4"
        lg="3"
      >
        <v-card variant="elevated" hover>
          <v-card-title class="d-flex justify-space-between align-start pa-4 pb-2">
            <div>
              <div class="text-h6 font-weight-bold">{{ position.name }}</div>
              <div class="text-caption text-medium-emphasis">{{ position.symbol }}</div>
            </div>
            <v-chip color="primary" size="small" variant="flat" class="bg-primary-container">
              {{ position.sector }}
            </v-chip>
          </v-card-title>

          <v-card-text>
            <v-list density="compact" class="pa-0">
              <v-list-item class="px-0">
                <template v-slot:prepend>
                  <v-icon color="grey" size="small">mdi-package-variant</v-icon>
                </template>
                <v-list-item-title class="text-caption">持仓量</v-list-item-title>
                <template v-slot:append>
                  <span class="font-weight-bold">{{ position.quantity }}</span>
                </template>
              </v-list-item>

              <v-list-item class="px-0">
                <template v-slot:prepend>
                  <v-icon color="grey" size="small">mdi-currency-usd</v-icon>
                </template>
                <v-list-item-title class="text-caption">成本价</v-list-item-title>
                <template v-slot:append>
                  <span class="font-weight-bold">¥{{ position.cost_price?.toFixed(2) }}</span>
                </template>
              </v-list-item>

              <v-list-item class="px-0">
                <template v-slot:prepend>
                  <v-icon color="grey" size="small">mdi-chart-line</v-icon>
                </template>
                <v-list-item-title class="text-caption">现价</v-list-item-title>
                <template v-slot:append>
                  <span class="font-weight-bold">¥{{ position.current_price?.toFixed(2) }}</span>
                </template>
              </v-list-item>

              <v-list-item class="px-0">
                <template v-slot:prepend>
                  <v-icon color="grey" size="small">mdi-wallet</v-icon>
                </template>
                <v-list-item-title class="text-caption">市值</v-list-item-title>
                <template v-slot:append>
                  <span class="font-weight-bold">¥{{ position.market_value?.toLocaleString() }}</span>
                </template>
              </v-list-item>

              <v-divider class="my-2"></v-divider>

              <v-list-item class="px-0">
                <template v-slot:prepend>
                  <v-icon :color="position.unrealized_pnl > 0 ? 'success' : 'error'" size="small">
                    {{ position.unrealized_pnl > 0 ? 'mdi-trending-up' : 'mdi-trending-down' }}
                  </v-icon>
                </template>
                <v-list-item-title class="text-caption">收益</v-list-item-title>
                <template v-slot:append>
                  <div class="text-end">
                    <div :class="position.unrealized_pnl > 0 ? 'text-success' : 'text-error'" class="font-weight-bold">
                      ¥{{ position.unrealized_pnl?.toFixed(2) }}
                    </div>
                    <div :class="position.unrealized_pnl > 0 ? 'text-success' : 'text-error'" class="text-caption">
                      {{ position.pnl_rate?.toFixed(2) }}%
                    </div>
                  </div>
                </template>
              </v-list-item>
            </v-list>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>

    <v-alert v-if="dashboardStore.positions.length === 0" type="info" variant="tonal" class="mt-4">
      暂无持仓数据
    </v-alert>
  </v-container>
</template>

<script setup>
import { onMounted } from 'vue'
import { useDashboardStore } from '@/stores/dashboard'

const dashboardStore = useDashboardStore()

onMounted(() => {
  dashboardStore.fetchPositions()
})
</script>

<style lang="scss" scoped>
.portfolio-view {
  max-width: 1600px;
  margin: 0 auto;
}
</style>

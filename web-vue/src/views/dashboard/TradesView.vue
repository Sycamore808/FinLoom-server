<template>
  <v-container fluid class="trades-view pa-6">
    <div class="mb-6">
      <h1 class="text-h3 font-weight-bold">交易记录</h1>
      <p class="text-body-1 text-medium-emphasis">查看所有交易历史记录</p>
    </div>

    <v-card variant="elevated">
      <v-card-title class="d-flex align-center justify-space-between pa-6">
        <div class="d-flex align-center">
          <v-avatar color="primary" variant="tonal" size="40" class="mr-3">
            <v-icon>mdi-swap-horizontal</v-icon>
          </v-avatar>
          <span class="text-h6 font-weight-bold">所有交易</span>
        </div>
        <v-btn color="primary" prepend-icon="mdi-refresh" @click="dashboardStore.fetchRecentTrades()">
          刷新
        </v-btn>
      </v-card-title>
      
      <v-card-text class="pa-0">
        <v-data-table
          :headers="headers"
          :items="dashboardStore.recentTrades"
          :items-per-page="10"
          class="elevation-0"
        >
          <template v-slot:item.action="{ item }">
            <v-chip :color="item.action === 'BUY' ? 'success' : 'error'" size="small">
              {{ item.action === 'BUY' ? '买入' : '卖出' }}
            </v-chip>
          </template>

          <template v-slot:item.price="{ item }">
            ¥{{ item.price.toFixed(2) }}
          </template>

          <template v-slot:item.amount="{ item }">
            ¥{{ item.amount.toLocaleString() }}
          </template>

          <template v-slot:item.pnl="{ item }">
            <span :class="item.pnl >= 0 ? 'text-success' : 'text-error'" class="font-weight-bold">
              ¥{{ item.pnl.toFixed(2) }}
            </span>
          </template>

          <template v-slot:item.status="{ item }">
            <v-chip color="info" size="small" variant="tonal">
              {{ item.status }}
            </v-chip>
          </template>
        </v-data-table>
      </v-card-text>
    </v-card>
  </v-container>
</template>

<script setup>
import { onMounted } from 'vue'
import { useDashboardStore } from '@/stores/dashboard'

const dashboardStore = useDashboardStore()

const headers = [
  { title: '时间', key: 'time', sortable: true },
  { title: '股票代码', key: 'symbol', sortable: true },
  { title: '股票名称', key: 'name', sortable: true },
  { title: '操作', key: 'action', sortable: true },
  { title: '价格', key: 'price', sortable: true },
  { title: '数量', key: 'quantity', sortable: true },
  { title: '金额', key: 'amount', sortable: true },
  { title: '盈亏', key: 'pnl', sortable: true },
  { title: '状态', key: 'status', sortable: true }
]

onMounted(() => {
  dashboardStore.fetchRecentTrades()
})
</script>

<style lang="scss" scoped>
.trades-view {
  max-width: 1600px;
  margin: 0 auto;
}
</style>

<template>
  <v-container fluid class="reports-view pa-6">
    <div class="mb-6">
      <h1 class="text-h3 font-weight-bold mb-2">报告中心</h1>
      <p class="text-body-1 text-medium-emphasis">查看和管理您的投资报告</p>
    </div>

    <v-row class="mb-4">
      <v-col v-for="reportType in reportTypes" :key="reportType.id" cols="12" sm="6" md="3">
        <v-card variant="elevated" hover class="h-100">
          <v-card-text class="pa-8 d-flex flex-column">
            <v-avatar :color="reportType.color" variant="tonal" size="64" class="mb-6">
              <v-icon :icon="reportType.icon" size="36"></v-icon>
            </v-avatar>
            <h3 class="text-h6 font-weight-bold mb-3">{{ reportType.title }}</h3>
            <p class="text-body-2 text-medium-emphasis mb-6 flex-grow-1">{{ reportType.description }}</p>
            <v-btn color="primary" variant="elevated" size="large" block>
              <v-icon start>mdi-file-document</v-icon>
              生成报告
            </v-btn>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>

    <v-card variant="elevated" class="mt-6">
      <v-card-title class="d-flex align-center pa-6">
        <v-avatar color="primary" variant="tonal" size="40" class="mr-3">
          <v-icon>mdi-history</v-icon>
        </v-avatar>
        <span class="text-h6 font-weight-bold">最近生成的报告</span>
      </v-card-title>
      <v-list lines="two">
        <v-list-item v-for="report in recentReports" :key="report.id" class="py-4">
          <template v-slot:prepend>
            <v-avatar color="primary" variant="tonal" size="48" class="mr-4">
              <v-icon icon="mdi-file-pdf-box" size="28"></v-icon>
            </v-avatar>
          </template>
          <v-list-item-title class="font-weight-bold text-body-1 mb-1">{{ report.name }}</v-list-item-title>
          <v-list-item-subtitle>{{ report.date }}</v-list-item-subtitle>
          <template v-slot:append>
            <div class="d-flex gap-2">
              <v-btn icon="mdi-download" variant="tonal" size="small" color="primary"></v-btn>
              <v-btn icon="mdi-share" variant="tonal" size="small" color="primary"></v-btn>
              <v-btn icon="mdi-delete" variant="tonal" size="small" color="error"></v-btn>
            </div>
          </template>
        </v-list-item>
      </v-list>
    </v-card>
  </v-container>
</template>

<script setup>
import { ref } from 'vue'

const reportTypes = [
  { id: 1, title: '月度绩效报告', description: '详细的月度投资组合表现分析', icon: 'mdi-chart-line', color: 'primary' },
  { id: 2, title: '风险评估报告', description: '全面的风险指标和压力测试结果', icon: 'mdi-shield-check', color: 'error' },
  { id: 3, title: '交易分析报告', description: '交易记录和执行质量分析', icon: 'mdi-swap-horizontal', color: 'success' },
  { id: 4, title: '策略回测报告', description: '策略历史表现和统计分析', icon: 'mdi-flask', color: 'info' }
]

const recentReports = ref([
  { id: 1, name: '2025年1月月度报告', date: '2025-02-01' },
  { id: 2, name: '风险评估报告 - Q4', date: '2025-01-15' },
  { id: 3, name: '策略回测分析', date: '2025-01-10' },
  { id: 4, name: '交易执行质量报告', date: '2025-01-05' }
])
</script>

<template>
  <v-container fluid class="notifications-view pa-6">
    <div class="mb-6">
      <h1 class="text-h3 font-weight-bold mb-2">通知中心</h1>
      <p class="text-body-1 text-medium-emphasis">查看系统通知和重要提醒</p>
    </div>

    <v-row>
      <v-col cols="12" md="8">
        <v-card variant="elevated">
          <v-card-title class="d-flex align-center justify-space-between pa-6">
            <div class="d-flex align-center">
              <v-avatar color="primary" variant="tonal" size="40" class="mr-3">
                <v-icon>mdi-bell</v-icon>
              </v-avatar>
              <span class="text-h6 font-weight-bold">所有通知</span>
            </div>
            <v-btn color="primary" size="small" @click="markAllRead">
              全部标记为已读
            </v-btn>
          </v-card-title>
          
          <v-list lines="three">
            <v-list-item
              v-for="notification in notifications"
              :key="notification.id"
              :class="{ 'bg-grey-lighten-4': !notification.read }"
            >
              <template v-slot:prepend>
                <v-avatar :color="notification.color" size="40">
                  <v-icon :icon="notification.icon" color="white"></v-icon>
                </v-avatar>
              </template>

              <v-list-item-title class="font-weight-bold">
                {{ notification.title }}
              </v-list-item-title>
              <v-list-item-subtitle>
                {{ notification.message }}
              </v-list-item-subtitle>
              <v-list-item-subtitle class="text-caption">
                {{ notification.time }}
              </v-list-item-subtitle>

              <template v-slot:append>
                <v-btn
                  icon="mdi-close"
                  variant="text"
                  size="small"
                  @click="removeNotification(notification.id)"
                ></v-btn>
              </template>
            </v-list-item>
          </v-list>

          <v-alert v-if="notifications.length === 0" type="info" variant="tonal" class="ma-4">
            暂无通知
          </v-alert>
        </v-card>
      </v-col>

      <v-col cols="12" md="4">
        <v-card variant="elevated" class="mb-4">
          <v-card-title class="d-flex align-center pa-4">
            <v-avatar color="primary" variant="tonal" size="40" class="mr-3">
              <v-icon>mdi-cog</v-icon>
            </v-avatar>
            <span class="text-h6 font-weight-bold">通知设置</span>
          </v-card-title>
          <v-card-text>
            <v-switch
              v-model="settings.priceAlerts"
              label="价格预警"
              color="primary"
              hide-details
              class="mb-2"
            ></v-switch>
            <v-switch
              v-model="settings.tradeNotifications"
              label="交易通知"
              color="primary"
              hide-details
              class="mb-2"
            ></v-switch>
            <v-switch
              v-model="settings.systemUpdates"
              label="系统更新"
              color="primary"
              hide-details
            ></v-switch>
          </v-card-text>
        </v-card>

        <v-card variant="elevated">
          <v-card-title class="d-flex align-center pa-4">
            <v-avatar color="success" variant="tonal" size="40" class="mr-3">
              <v-icon>mdi-chart-box</v-icon>
            </v-avatar>
            <span class="text-h6 font-weight-bold">通知统计</span>
          </v-card-title>
          <v-card-text>
            <v-list density="compact">
              <v-list-item>
                <v-list-item-title>总通知数</v-list-item-title>
                <template v-slot:append>
                  <v-chip color="primary" size="small">{{ notifications.length }}</v-chip>
                </template>
              </v-list-item>
              <v-list-item>
                <v-list-item-title>未读通知</v-list-item-title>
                <template v-slot:append>
                  <v-chip color="error" size="small">{{ unreadCount }}</v-chip>
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
import { ref, computed } from 'vue'

const notifications = ref([
  {
    id: 1,
    title: '价格预警',
    message: '股票 000001 已达到目标价格 ¥50.00',
    time: '5分钟前',
    read: false,
    color: 'warning',
    icon: 'mdi-alert'
  },
  {
    id: 2,
    title: '交易完成',
    message: '您的买入订单已成功执行',
    time: '1小时前',
    read: false,
    color: 'success',
    icon: 'mdi-check-circle'
  },
  {
    id: 3,
    title: '系统维护',
    message: '系统将于今晚22:00进行维护',
    time: '2小时前',
    read: true,
    color: 'info',
    icon: 'mdi-information'
  }
])

const settings = ref({
  priceAlerts: true,
  tradeNotifications: true,
  systemUpdates: true
})

const unreadCount = computed(() => {
  return notifications.value.filter(n => !n.read).length
})

function markAllRead() {
  notifications.value.forEach(n => n.read = true)
}

function removeNotification(id) {
  const index = notifications.value.findIndex(n => n.id === id)
  if (index > -1) {
    notifications.value.splice(index, 1)
  }
}
</script>

<template>
  <v-container fluid class="settings-view pa-6">
    <!-- 页面标题 -->
    <div class="mb-8">
      <h1 class="text-h3 font-weight-bold mb-2">系统设置</h1>
      <p class="text-body-1 text-medium-emphasis">管理您的账户和系统配置</p>
    </div>

    <!-- 主内容区域 -->
    <v-row>
      <!-- 侧边栏导航 - Material 3 风格 -->
      <v-col cols="12" md="3">
        <v-card variant="flat">
          <v-list nav density="comfortable" bg-color="transparent">
            <v-list-item
              v-for="tab in settingsTabs"
              :key="tab.id"
              :value="tab.id"
              :active="activeTab === tab.id"
              @click="activeTab = tab.id"
              :prepend-icon="tab.icon"
              :title="tab.label"
              rounded="xl"
              color="primary"
            >
            </v-list-item>
          </v-list>
        </v-card>
      </v-col>

      <!-- 内容区域 -->
      <v-col cols="12" md="9">
        <v-window v-model="activeTab" class="settings-window">
          <!-- 个人信息 - Material 3 风格 -->
          <v-window-item value="profile">
            <v-card variant="elevated">
              <v-card-title class="text-h5 font-weight-bold d-flex align-center pa-6">
                <v-avatar color="primary" variant="tonal" size="48" class="mr-4">
                  <v-icon size="32">mdi-account</v-icon>
                </v-avatar>
                <div>
                  <div>个人信息</div>
                  <div class="text-caption text-medium-emphasis font-weight-regular">管理您的账户信息</div>
                </div>
              </v-card-title>
              <v-card-text class="pa-6">
                <v-form>
                  <v-text-field
                    v-model="profile.username"
                    label="用户名"
                    prepend-inner-icon="mdi-account-circle"
                    class="mb-4"
                  ></v-text-field>

                  <v-text-field
                    v-model="profile.email"
                    label="邮箱"
                    type="email"
                    prepend-inner-icon="mdi-email"
                    class="mb-4"
                  ></v-text-field>

                  <v-text-field
                    v-model="profile.phone"
                    label="电话"
                    type="tel"
                    prepend-inner-icon="mdi-phone"
                    class="mb-4"
                  ></v-text-field>
                </v-form>
              </v-card-text>
              <v-card-actions class="px-6 pb-6">
                <v-btn
                  color="primary"
                  size="large"
                  variant="elevated"
                  prepend-icon="mdi-content-save"
                  @click="saveProfile"
                >
                  保存更改
                </v-btn>
                <v-btn
                  size="large"
                  variant="text"
                  @click="resetProfile"
                >
                  重置
                </v-btn>
              </v-card-actions>
            </v-card>
          </v-window-item>

          <!-- 交易设置 -->
          <v-window-item value="trading">
            <v-card elevation="2" rounded="lg">
              <v-card-title class="text-h5 font-weight-bold">
                <v-icon start>mdi-chart-line</v-icon>
                交易设置
              </v-card-title>
              <v-divider></v-divider>
              <v-card-text>
                <v-form>
                  <v-select
                    v-model="trading.orderType"
                    :items="orderTypes"
                    label="默认订单类型"
                    prepend-inner-icon="mdi-file-document-edit"
                    variant="outlined"
                    density="comfortable"
                    class="mb-4"
                  ></v-select>

                  <v-text-field
                    v-model.number="trading.riskLimit"
                    label="风险限额 (%)"
                    type="number"
                    prepend-inner-icon="mdi-shield-alert"
                    variant="outlined"
                    density="comfortable"
                    class="mb-4"
                    min="0"
                    max="100"
                  ></v-text-field>

                  <v-switch
                    v-model="trading.confirmOrders"
                    label="下单前需要确认"
                    color="primary"
                    inset
                    hide-details
                    class="mb-2"
                  ></v-switch>

                  <v-switch
                    v-model="trading.autoStopLoss"
                    label="自动设置止损"
                    color="primary"
                    inset
                    hide-details
                    class="mb-2"
                  ></v-switch>
                </v-form>
              </v-card-text>
              <v-card-actions class="px-6 pb-6">
                <v-btn
                  color="primary"
                  size="large"
                  variant="elevated"
                  prepend-icon="mdi-content-save"
                  @click="saveTrading"
                >
                  保存更改
                </v-btn>
              </v-card-actions>
            </v-card>
          </v-window-item>

          <!-- 通知设置 -->
          <v-window-item value="notifications">
            <v-card elevation="2" rounded="lg">
              <v-card-title class="text-h5 font-weight-bold">
                <v-icon start>mdi-bell</v-icon>
                通知设置
              </v-card-title>
              <v-divider></v-divider>
              <v-card-text>
                <v-list>
                  <v-list-item>
                    <template v-slot:prepend>
                      <v-icon color="primary">mdi-email</v-icon>
                    </template>
                    <v-list-item-title>邮件通知</v-list-item-title>
                    <v-list-item-subtitle>接收交易和价格提醒邮件</v-list-item-subtitle>
                    <template v-slot:append>
                      <v-switch
                        v-model="notifications.email"
                        color="primary"
                        hide-details
                        inset
                      ></v-switch>
                    </template>
                  </v-list-item>

                  <v-divider></v-divider>

                  <v-list-item>
                    <template v-slot:prepend>
                      <v-icon color="primary">mdi-message-text</v-icon>
                    </template>
                    <v-list-item-title>短信通知</v-list-item-title>
                    <v-list-item-subtitle>接收重要交易短信提醒</v-list-item-subtitle>
                    <template v-slot:append>
                      <v-switch
                        v-model="notifications.sms"
                        color="primary"
                        hide-details
                        inset
                      ></v-switch>
                    </template>
                  </v-list-item>

                  <v-divider></v-divider>

                  <v-list-item>
                    <template v-slot:prepend>
                      <v-icon color="primary">mdi-cellphone</v-icon>
                    </template>
                    <v-list-item-title>推送通知</v-list-item-title>
                    <v-list-item-subtitle>接收应用推送消息</v-list-item-subtitle>
                    <template v-slot:append>
                      <v-switch
                        v-model="notifications.push"
                        color="primary"
                        hide-details
                        inset
                      ></v-switch>
                    </template>
                  </v-list-item>

                  <v-divider></v-divider>

                  <v-list-item>
                    <template v-slot:prepend>
                      <v-icon color="warning">mdi-alert</v-icon>
                    </template>
                    <v-list-item-title>价格预警</v-list-item-title>
                    <v-list-item-subtitle>价格达到设定值时通知</v-list-item-subtitle>
                    <template v-slot:append>
                      <v-switch
                        v-model="notifications.priceAlerts"
                        color="primary"
                        hide-details
                        inset
                      ></v-switch>
                    </template>
                  </v-list-item>
                </v-list>
              </v-card-text>
              <v-card-actions class="px-6 pb-6">
                <v-btn
                  color="primary"
                  size="large"
                  variant="elevated"
                  prepend-icon="mdi-content-save"
                  @click="saveNotifications"
                >
                  保存更改
                </v-btn>
              </v-card-actions>
            </v-card>
          </v-window-item>

          <!-- 安全设置 -->
          <v-window-item value="security">
            <v-card elevation="2" rounded="lg">
              <v-card-title class="text-h5 font-weight-bold">
                <v-icon start>mdi-shield-lock</v-icon>
                安全设置
              </v-card-title>
              <v-divider></v-divider>
              <v-card-text>
                <v-form ref="securityForm">
                  <v-text-field
                    v-model="security.currentPassword"
                    label="当前密码"
                    type="password"
                    prepend-inner-icon="mdi-lock"
                    variant="outlined"
                    density="comfortable"
                    class="mb-4"
                    :rules="[rules.required]"
                  ></v-text-field>

                  <v-text-field
                    v-model="security.newPassword"
                    label="新密码"
                    type="password"
                    prepend-inner-icon="mdi-lock-reset"
                    variant="outlined"
                    density="comfortable"
                    class="mb-4"
                    :rules="[rules.required, rules.minLength]"
                  ></v-text-field>

                  <v-text-field
                    v-model="security.confirmPassword"
                    label="确认新密码"
                    type="password"
                    prepend-inner-icon="mdi-lock-check"
                    variant="outlined"
                    density="comfortable"
                    class="mb-4"
                    :rules="[rules.required, rules.passwordMatch]"
                  ></v-text-field>

                  <v-divider class="my-4"></v-divider>

                  <v-switch
                    v-model="security.twoFactorAuth"
                    label="启用两步验证"
                    color="primary"
                    inset
                    hide-details
                    class="mb-2"
                  >
                    <template v-slot:prepend>
                      <v-icon color="success">mdi-shield-check</v-icon>
                    </template>
                  </v-switch>

                  <v-alert
                    v-if="security.twoFactorAuth"
                    type="info"
                    variant="tonal"
                    class="mt-4"
                    density="compact"
                  >
                    两步验证将增强您的账户安全性
                  </v-alert>
                </v-form>
              </v-card-text>
              <v-card-actions class="px-6 pb-6">
                <v-btn
                  color="primary"
                  size="large"
                  variant="elevated"
                  prepend-icon="mdi-key-change"
                  @click="updatePassword"
                >
                  更新密码
                </v-btn>
              </v-card-actions>
            </v-card>
          </v-window-item>

          <!-- 系统偏好 -->
          <v-window-item value="preferences">
            <v-card elevation="2" rounded="lg">
              <v-card-title class="text-h5 font-weight-bold">
                <v-icon start>mdi-cog</v-icon>
                系统偏好
              </v-card-title>
              <v-divider></v-divider>
              <v-card-text>
                <v-form>
                  <v-select
                    v-model="preferences.language"
                    :items="languages"
                    label="语言"
                    prepend-inner-icon="mdi-translate"
                    variant="outlined"
                    density="comfortable"
                    class="mb-4"
                  ></v-select>

                  <v-select
                    v-model="preferences.timezone"
                    :items="timezones"
                    label="时区"
                    prepend-inner-icon="mdi-clock-outline"
                    variant="outlined"
                    density="comfortable"
                    class="mb-4"
                  ></v-select>

                  <v-select
                    v-model="preferences.theme"
                    :items="themes"
                    label="主题"
                    prepend-inner-icon="mdi-palette"
                    variant="outlined"
                    density="comfortable"
                    class="mb-4"
                    @update:model-value="changeTheme"
                  ></v-select>

                  <v-select
                    v-model="preferences.chartStyle"
                    :items="chartStyles"
                    label="图表风格"
                    prepend-inner-icon="mdi-chart-areaspline"
                    variant="outlined"
                    density="comfortable"
                    class="mb-4"
                  ></v-select>
                </v-form>
              </v-card-text>
              <v-card-actions class="px-6 pb-6">
                <v-btn
                  color="primary"
                  size="large"
                  variant="elevated"
                  prepend-icon="mdi-content-save"
                  @click="savePreferences"
                >
                  保存更改
                </v-btn>
              </v-card-actions>
            </v-card>
          </v-window-item>
        </v-window>
      </v-col>
    </v-row>

    <!-- Snackbar 提示 -->
    <v-snackbar
      v-model="snackbar.show"
      :color="snackbar.color"
      :timeout="3000"
      location="top"
    >
      {{ snackbar.message }}
      <template v-slot:actions>
        <v-btn icon="mdi-close" size="small" @click="snackbar.show = false"></v-btn>
      </template>
    </v-snackbar>
  </v-container>
</template>

<script setup>
import { ref, reactive } from 'vue'
import { useTheme } from 'vuetify'

const theme = useTheme()
const activeTab = ref('profile')

// 设置标签页
const settingsTabs = [
  { id: 'profile', label: '个人信息', icon: 'mdi-account' },
  { id: 'trading', label: '交易设置', icon: 'mdi-chart-line' },
  { id: 'notifications', label: '通知设置', icon: 'mdi-bell' },
  { id: 'security', label: '安全设置', icon: 'mdi-shield-lock' },
  { id: 'preferences', label: '系统偏好', icon: 'mdi-cog' }
]

// 个人信息
const profile = ref({
  username: 'FinLoom用户',
  email: 'user@finloom.com',
  phone: '+86 138 0000 0000'
})

// 交易设置
const trading = ref({
  orderType: 'market',
  riskLimit: 2,
  confirmOrders: true,
  autoStopLoss: false
})

const orderTypes = [
  { title: '市价单', value: 'market' },
  { title: '限价单', value: 'limit' },
  { title: '止损单', value: 'stop' }
]

// 通知设置
const notifications = ref({
  email: true,
  sms: false,
  push: true,
  priceAlerts: true
})

// 安全设置
const security = ref({
  currentPassword: '',
  newPassword: '',
  confirmPassword: '',
  twoFactorAuth: false
})

// 表单验证规则
const rules = {
  required: value => !!value || '此字段为必填项',
  minLength: value => (value && value.length >= 8) || '密码至少需要8个字符',
  passwordMatch: value => value === security.value.newPassword || '两次输入的密码不匹配'
}

// 系统偏好
const preferences = ref({
  language: 'zh-CN',
  timezone: 'Asia/Shanghai',
  theme: 'light',
  chartStyle: 'candlestick'
})

const languages = [
  { title: '简体中文', value: 'zh-CN' },
  { title: 'English', value: 'en-US' }
]

const timezones = [
  { title: '上海 (GMT+8)', value: 'Asia/Shanghai' },
  { title: '香港 (GMT+8)', value: 'Asia/Hong_Kong' },
  { title: '纽约 (GMT-5)', value: 'America/New_York' }
]

const themes = [
  { title: '浅色', value: 'light' },
  { title: '深色', value: 'dark' },
  { title: '自动', value: 'auto' }
]

const chartStyles = [
  { title: 'K线图', value: 'candlestick' },
  { title: '折线图', value: 'line' },
  { title: '柱状图', value: 'bar' }
]

// Snackbar 状态
const snackbar = reactive({
  show: false,
  message: '',
  color: 'success'
})

// 显示提示信息
const showMessage = (message, color = 'success') => {
  snackbar.message = message
  snackbar.color = color
  snackbar.show = true
}

// 保存方法
const saveProfile = () => {
  showMessage('个人信息已保存')
}

const saveTrading = () => {
  showMessage('交易设置已保存')
}

const saveNotifications = () => {
  showMessage('通知设置已保存')
}

const updatePassword = () => {
  if (security.value.newPassword !== security.value.confirmPassword) {
    showMessage('两次输入的密码不匹配', 'error')
    return
  }
  showMessage('密码已更新')
  security.value = {
    currentPassword: '',
    newPassword: '',
    confirmPassword: '',
    twoFactorAuth: security.value.twoFactorAuth
  }
}

const savePreferences = () => {
  showMessage('系统偏好已保存')
}

const resetProfile = () => {
  showMessage('已重置为默认值', 'info')
}

// 切换主题
const changeTheme = (value) => {
  if (value === 'dark') {
    theme.global.name.value = 'finloomDarkTheme'
  } else if (value === 'light') {
    theme.global.name.value = 'finloomTheme'
  } else {
    // 自动模式：根据系统偏好
    const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches
    theme.global.name.value = isDark ? 'finloomDarkTheme' : 'finloomTheme'
  }
}
</script>

<style lang="scss" scoped>
.settings-view {
  max-width: 1400px;
  margin: 0 auto;
}

.settings-window {
  // 确保窗口项之间平滑过渡
  :deep(.v-window__container) {
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  }
}

// 自定义卡片样式
:deep(.v-card) {
  transition: all 0.3s ease;

  &:hover {
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12) !important;
  }
}

// 自定义列表项样式
:deep(.v-list-item) {
  margin-bottom: 4px;
  
  &.v-list-item--active {
    font-weight: 600;
  }
}

// 响应式调整
@media (max-width: 960px) {
  .settings-view {
    padding: 1rem !important;
  }
}
</style>


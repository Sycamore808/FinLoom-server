<template>
  <div class="settings-view">
    <div class="page-header">
      <h1>系统设置</h1>
      <p>管理您的账户和系统配置</p>
    </div>

    <div class="settings-container">
      <!-- 侧边栏标签 -->
      <div class="settings-sidebar">
        <div
          v-for="tab in settingsTabs"
          :key="tab.id"
          class="settings-tab"
          :class="{ active: activeTab === tab.id }"
          @click="activeTab = tab.id"
        >
          <i :class="tab.icon"></i>
          <span>{{ tab.label }}</span>
        </div>
      </div>

      <!-- 设置内容 -->
      <div class="settings-content">
        <!-- 个人信息 -->
        <Card v-if="activeTab === 'profile'">
          <template #header>
            <h3>个人信息</h3>
          </template>
          <div class="setting-section">
            <div class="setting-row">
              <label>用户名</label>
              <input type="text" v-model="profile.username" class="input-field" />
            </div>
            <div class="setting-row">
              <label>邮箱</label>
              <input type="email" v-model="profile.email" class="input-field" />
            </div>
            <div class="setting-row">
              <label>电话</label>
              <input type="tel" v-model="profile.phone" class="input-field" />
            </div>
            <button class="btn-save">保存更改</button>
          </div>
        </Card>

        <!-- 交易设置 -->
        <Card v-if="activeTab === 'trading'">
          <template #header>
            <h3>交易设置</h3>
          </template>
          <div class="setting-section">
            <div class="setting-row">
              <label>默认订单类型</label>
              <select v-model="trading.orderType" class="select-field">
                <option value="market">市价单</option>
                <option value="limit">限价单</option>
                <option value="stop">止损单</option>
              </select>
            </div>
            <div class="setting-row">
              <label>风险限额 (%)</label>
              <input type="number" v-model="trading.riskLimit" class="input-field" />
            </div>
            <div class="setting-row checkbox-row">
              <input type="checkbox" v-model="trading.confirmOrders" id="confirm-orders" />
              <label for="confirm-orders">下单前需要确认</label>
            </div>
            <button class="btn-save">保存更改</button>
          </div>
        </Card>

        <!-- 通知设置 -->
        <Card v-if="activeTab === 'notifications'">
          <template #header>
            <h3>通知设置</h3>
          </template>
          <div class="setting-section">
            <div class="setting-row checkbox-row">
              <input type="checkbox" v-model="notifications.email" id="email-notif" />
              <label for="email-notif">邮件通知</label>
            </div>
            <div class="setting-row checkbox-row">
              <input type="checkbox" v-model="notifications.sms" id="sms-notif" />
              <label for="sms-notif">短信通知</label>
            </div>
            <div class="setting-row checkbox-row">
              <input type="checkbox" v-model="notifications.push" id="push-notif" />
              <label for="push-notif">推送通知</label>
            </div>
            <div class="setting-row checkbox-row">
              <input type="checkbox" v-model="notifications.priceAlerts" id="price-alerts" />
              <label for="price-alerts">价格预警</label>
            </div>
            <button class="btn-save">保存更改</button>
          </div>
        </Card>

        <!-- 安全设置 -->
        <Card v-if="activeTab === 'security'">
          <template #header>
            <h3>安全设置</h3>
          </template>
          <div class="setting-section">
            <div class="setting-row">
              <label>当前密码</label>
              <input type="password" class="input-field" placeholder="输入当前密码" />
            </div>
            <div class="setting-row">
              <label>新密码</label>
              <input type="password" class="input-field" placeholder="输入新密码" />
            </div>
            <div class="setting-row">
              <label>确认新密码</label>
              <input type="password" class="input-field" placeholder="再次输入新密码" />
            </div>
            <div class="setting-row checkbox-row">
              <input type="checkbox" id="two-factor" />
              <label for="two-factor">启用两步验证</label>
            </div>
            <button class="btn-save">更新密码</button>
          </div>
        </Card>

        <!-- 系统偏好 -->
        <Card v-if="activeTab === 'preferences'">
          <template #header>
            <h3>系统偏好</h3>
          </template>
          <div class="setting-section">
            <div class="setting-row">
              <label>语言</label>
              <select v-model="preferences.language" class="select-field">
                <option value="zh-CN">简体中文</option>
                <option value="en-US">English</option>
              </select>
            </div>
            <div class="setting-row">
              <label>时区</label>
              <select v-model="preferences.timezone" class="select-field">
                <option value="Asia/Shanghai">上海 (GMT+8)</option>
                <option value="Asia/Hong_Kong">香港 (GMT+8)</option>
                <option value="America/New_York">纽约 (GMT-5)</option>
              </select>
            </div>
            <div class="setting-row">
              <label>主题</label>
              <select v-model="preferences.theme" class="select-field">
                <option value="light">浅色</option>
                <option value="dark">深色</option>
                <option value="auto">自动</option>
              </select>
            </div>
            <button class="btn-save">保存更改</button>
          </div>
        </Card>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import Card from '@/components/ui/Card.vue'

const activeTab = ref('profile')

const settingsTabs = [
  { id: 'profile', label: '个人信息', icon: 'fas fa-user' },
  { id: 'trading', label: '交易设置', icon: 'fas fa-exchange-alt' },
  { id: 'notifications', label: '通知设置', icon: 'fas fa-bell' },
  { id: 'security', label: '安全设置', icon: 'fas fa-shield-alt' },
  { id: 'preferences', label: '系统偏好', icon: 'fas fa-sliders-h' }
]

const profile = ref({
  username: 'FinLoom用户',
  email: 'user@finloom.com',
  phone: '+86 138 0000 0000'
})

const trading = ref({
  orderType: 'market',
  riskLimit: 2,
  confirmOrders: true
})

const notifications = ref({
  email: true,
  sms: false,
  push: true,
  priceAlerts: true
})

const preferences = ref({
  language: 'zh-CN',
  timezone: 'Asia/Shanghai',
  theme: 'light'
})
</script>

<style lang="scss" scoped>
.settings-view {
  .page-header {
    margin-bottom: 2rem;

    h1 {
      font-size: 2rem;
      font-weight: 700;
      color: #0f172a;
      margin-bottom: 0.5rem;
    }

    p {
      color: #64748b;
    }
  }
}

.settings-container {
  display: grid;
  grid-template-columns: 250px 1fr;
  gap: 2rem;

  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
}

.settings-sidebar {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.settings-tab {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem;
  background: white;
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.3s ease;
  border: 2px solid transparent;

  i {
    font-size: 1.25rem;
    color: #64748b;
    transition: all 0.3s ease;
  }

  span {
    font-weight: 500;
    color: #475569;
    transition: all 0.3s ease;
  }

  &:hover {
    background: #f8fafc;
    transform: translateX(4px);
  }

  &.active {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-color: #667eea;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);

    i,
    span {
      color: white;
    }
  }
}

.settings-content {
  h3 {
    font-size: 1.25rem;
    font-weight: 700;
    color: #0f172a;
  }
}

.setting-section {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.setting-row {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;

  label {
    font-weight: 500;
    color: #475569;
    font-size: 0.95rem;
  }

  &.checkbox-row {
    flex-direction: row;
    align-items: center;
    gap: 0.75rem;

    input[type="checkbox"] {
      width: 20px;
      height: 20px;
      cursor: pointer;
      accent-color: #667eea;
    }

    label {
      margin: 0;
      cursor: pointer;
    }
  }
}

.input-field,
.select-field {
  padding: 0.75rem 1rem;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  font-size: 0.95rem;
  transition: all 0.3s ease;
  background: white;

  &:focus {
    outline: none;
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
  }
}

.btn-save {
  padding: 0.875rem 2rem;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 8px;
  font-weight: 600;
  font-size: 0.95rem;
  cursor: pointer;
  transition: all 0.3s ease;
  align-self: flex-start;

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
  }

  &:active {
    transform: translateY(0);
  }
}
</style>


<template>
  <div class="login-view">
    <v-container fluid class="login-container">
      <v-row no-gutters class="fill-height">
        <!-- 登录表单区域 -->
        <v-col cols="12" md="6" class="d-flex align-center justify-center pa-6">
          <v-card variant="elevated" class="login-card" max-width="480" width="100%">
            <v-card-text class="pa-8">
              <!-- Logo和标题 -->
              <div class="text-center mb-8">
                <div class="logo mb-4">
                  <v-icon size="48" color="white">mdi-chart-network</v-icon>
                </div>
                <h1 class="text-h3 font-weight-bold mb-2">欢迎回来</h1>
                <p class="text-body-1 text-medium-emphasis">登录您的FinLoom账户</p>
              </div>

              <!-- 登录表单 -->
              <v-form @submit.prevent="handleLogin">
                <v-text-field
                  v-model="form.username"
                  label="用户名"
                  prepend-inner-icon="mdi-account"
                  variant="outlined"
                  class="mb-4"
                  required
                  :rules="[v => !!v || '请输入用户名']"
                ></v-text-field>

                <v-text-field
                  v-model="form.password"
                  label="密码"
                  prepend-inner-icon="mdi-lock"
                  :append-inner-icon="showPassword ? 'mdi-eye-off' : 'mdi-eye'"
                  :type="showPassword ? 'text' : 'password'"
                  variant="outlined"
                  class="mb-4"
                  required
                  :rules="[v => !!v || '请输入密码']"
                  @click:append-inner="showPassword = !showPassword"
                ></v-text-field>

                <div class="d-flex justify-space-between align-center mb-6">
                  <v-checkbox
                    v-model="form.remember"
                    label="记住我"
                    density="compact"
                    hide-details
                  ></v-checkbox>
                  <v-btn
                    variant="text"
                    color="primary"
                    size="small"
                    class="text-none"
                  >
                    忘记密码？
                  </v-btn>
                </div>

                <v-btn
                  type="submit"
                  color="primary"
                  size="large"
                  block
                  :loading="loading"
                  class="mb-4"
                >
                  登录
                </v-btn>

                <v-alert
                  type="info"
                  variant="tonal"
                  density="compact"
                  class="mb-4"
                  rounded="lg"
                >
                  <template v-slot:prepend>
                    <v-icon>mdi-information</v-icon>
                  </template>
                  演示账号：任意用户名/密码即可登录
                </v-alert>
              </v-form>

              <!-- 注册链接 -->
              <div class="text-center">
                <span class="text-body-2 text-medium-emphasis">还没有账户？</span>
                <v-btn
                  variant="text"
                  color="primary"
                  size="small"
                  class="text-none ml-1"
                  @click="goToRegister"
                >
                  立即注册
                </v-btn>
              </div>
            </v-card-text>
          </v-card>
        </v-col>

        <!-- 信息展示区域 -->
        <v-col cols="12" md="6" class="info-panel d-none d-md-flex">
          <div class="info-content">
            <v-icon size="120" color="white" class="mb-6">mdi-chart-timeline-variant</v-icon>
            <h2 class="text-h2 font-weight-bold text-white mb-6">FinLoom量化投资平台</h2>
            
            <v-list class="bg-transparent" density="comfortable">
              <v-list-item
                v-for="feature in features"
                :key="feature.text"
                class="px-0"
              >
                <template v-slot:prepend>
                  <v-icon color="#10b981" size="24">mdi-check-circle</v-icon>
                </template>
                <v-list-item-title class="text-h6 text-white font-weight-medium">
                  {{ feature.text }}
                </v-list-item-title>
              </v-list-item>
            </v-list>
          </div>
        </v-col>
      </v-row>
    </v-container>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter, useRoute } from 'vue-router'

const router = useRouter()
const route = useRoute()

const form = ref({
  username: '',
  password: '',
  remember: false
})

const showPassword = ref(false)
const loading = ref(false)

const features = ref([
  { text: 'AI驱动的投资决策' },
  { text: '实时市场分析' },
  { text: '智能风险管理' },
  { text: '自动化交易执行' }
])

async function handleLogin() {
  loading.value = true
  
  // 模拟登录延迟
  await new Promise(resolve => setTimeout(resolve, 1000))
  
  // 简化的登录逻辑（演示用）
  localStorage.setItem('finloom_auth', 'true')
  localStorage.setItem('finloom_token', 'demo_token_' + Date.now())
  
  loading.value = false
  
  // 跳转到目标页面或仪表盘
  const redirect = route.query.redirect || '/dashboard'
  router.push(redirect)
}

function goToRegister() {
  // 注册功能暂未实现
  alert('注册功能开发中...')
}
</script>

<style lang="scss" scoped>
.login-view {
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 2rem;
}

.login-container {
  max-width: 1000px;
  width: 100%;
  background: white;
  border-radius: 20px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
  overflow: hidden;
  min-height: 600px;

  @media (max-width: 960px) {
    margin: 1rem;
    border-radius: 16px;
  }
}

.login-card {
  border-radius: 0;
  box-shadow: none;
  background: transparent;
}

.logo {
  width: 80px;
  height: 80px;
  margin: 0 auto;
  background: linear-gradient(135deg, #3b82f6, #8b5cf6);
  border-radius: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.info-panel {
  background: linear-gradient(135deg, #3b82f6, #8b5cf6);
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  padding: 3rem;
  color: white;

  .info-content {
    text-align: center;
    max-width: 400px;
  }
}

// 响应式调整
@media (max-width: 960px) {
  .login-view {
    padding: 1rem;
  }
  
  .login-container {
    background: transparent;
    box-shadow: none;
  }
  
  .login-card {
    background: white;
    border-radius: 16px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
  }
}
</style>


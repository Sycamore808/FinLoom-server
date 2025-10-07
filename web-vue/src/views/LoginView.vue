<template>
  <div class="login-view">
    <div class="login-container">
      <div class="login-card">
        <div class="login-header">
          <div class="logo">
            <i class="fas fa-chart-network"></i>
          </div>
          <h1>欢迎回来</h1>
          <p>登录您的FinLoom账户</p>
        </div>

        <form @submit.prevent="handleLogin" class="login-form">
          <div class="form-group">
            <label for="username">用户名</label>
            <div class="input-wrapper">
              <i class="fas fa-user"></i>
              <input
                id="username"
                v-model="form.username"
                type="text"
                placeholder="请输入用户名"
                required
              />
            </div>
          </div>

          <div class="form-group">
            <label for="password">密码</label>
            <div class="input-wrapper">
              <i class="fas fa-lock"></i>
              <input
                id="password"
                v-model="form.password"
                :type="showPassword ? 'text' : 'password'"
                placeholder="请输入密码"
                required
              />
              <button
                type="button"
                class="toggle-password"
                @click="showPassword = !showPassword"
              >
                <i :class="showPassword ? 'fas fa-eye-slash' : 'fas fa-eye'"></i>
              </button>
            </div>
          </div>

          <div class="form-options">
            <label class="checkbox-label">
              <input v-model="form.remember" type="checkbox" />
              <span>记住我</span>
            </label>
            <a href="#" class="forgot-link">忘记密码？</a>
          </div>

          <Button type="submit" size="large" block :loading="loading">
            登录
          </Button>

          <p class="demo-hint">
            演示账号：任意用户名/密码即可登录
          </p>
        </form>

        <div class="login-footer">
          <p>还没有账户？<a href="#" @click.prevent="goToRegister">立即注册</a></p>
        </div>
      </div>

      <div class="info-panel">
        <h2>FinLoom量化投资平台</h2>
        <ul class="feature-list">
          <li><i class="fas fa-check-circle"></i> AI驱动的投资决策</li>
          <li><i class="fas fa-check-circle"></i> 实时市场分析</li>
          <li><i class="fas fa-check-circle"></i> 智能风险管理</li>
          <li><i class="fas fa-check-circle"></i> 自动化交易执行</li>
        </ul>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import Button from '@/components/ui/Button.vue'

const router = useRouter()
const route = useRoute()

const form = ref({
  username: '',
  password: '',
  remember: false
})

const showPassword = ref(false)
const loading = ref(false)

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
  display: grid;
  grid-template-columns: 1fr 1fr;
  max-width: 1000px;
  width: 100%;
  background: white;
  border-radius: 20px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
  overflow: hidden;

  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
}

.login-card {
  padding: 3rem;
}

.login-header {
  text-align: center;
  margin-bottom: 2rem;

  .logo {
    width: 80px;
    height: 80px;
    margin: 0 auto 1.5rem;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    border-radius: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2.5rem;
    color: white;
  }

  h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.5rem;
  }

  p {
    color: #64748b;
    font-size: 1rem;
  }
}

.login-form {
  .form-group {
    margin-bottom: 1.5rem;

    label {
      display: block;
      margin-bottom: 0.5rem;
      font-weight: 600;
      color: #334155;
      font-size: 0.875rem;
    }

    .input-wrapper {
      position: relative;
      display: flex;
      align-items: center;

      i {
        position: absolute;
        left: 1rem;
        color: #94a3b8;
        font-size: 1rem;
      }

      input {
        width: 100%;
        padding: 0.875rem 1rem 0.875rem 3rem;
        border: 2px solid #e2e8f0;
        border-radius: 10px;
        font-size: 1rem;
        transition: all 0.3s ease;

        &:focus {
          outline: none;
          border-color: #3b82f6;
          box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }
      }

      .toggle-password {
        position: absolute;
        right: 1rem;
        background: none;
        border: none;
        color: #94a3b8;
        cursor: pointer;
        padding: 0.5rem;
        transition: color 0.3s ease;

        &:hover {
          color: #3b82f6;
        }
      }
    }
  }

  .form-options {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;

    .checkbox-label {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      cursor: pointer;
      font-size: 0.875rem;
      color: #64748b;

      input[type="checkbox"] {
        width: 18px;
        height: 18px;
        cursor: pointer;
      }
    }

    .forgot-link {
      color: #3b82f6;
      text-decoration: none;
      font-size: 0.875rem;
      font-weight: 600;

      &:hover {
        text-decoration: underline;
      }
    }
  }

  .demo-hint {
    margin-top: 1rem;
    text-align: center;
    font-size: 0.875rem;
    color: #94a3b8;
    font-style: italic;
  }
}

.login-footer {
  margin-top: 2rem;
  text-align: center;
  font-size: 0.875rem;
  color: #64748b;

  a {
    color: #3b82f6;
    text-decoration: none;
    font-weight: 600;

    &:hover {
      text-decoration: underline;
    }
  }
}

.info-panel {
  background: linear-gradient(135deg, #3b82f6, #8b5cf6);
  padding: 3rem;
  display: flex;
  flex-direction: column;
  justify-content: center;
  color: white;

  @media (max-width: 768px) {
    display: none;
  }

  h2 {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 2rem;
  }

  .feature-list {
    list-style: none;
    padding: 0;

    li {
      display: flex;
      align-items: center;
      gap: 1rem;
      padding: 1rem 0;
      font-size: 1.125rem;

      i {
        color: #10b981;
        font-size: 1.25rem;
      }
    }
  }
}
</style>


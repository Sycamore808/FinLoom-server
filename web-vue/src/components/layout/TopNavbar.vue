<template>
  <header class="top-navbar">
    <div class="navbar-content">
      <div class="navbar-left">
        <h1 class="page-title">{{ currentPageTitle }}</h1>
      </div>

      <div class="navbar-right">
        <div class="status-indicator" :class="appStore.appStatus">
          <div class="status-dot"></div>
          <span>{{ statusText }}</span>
        </div>

        <button class="icon-btn" @click="refreshData" title="刷新数据">
          <i class="fas fa-sync-alt" :class="{ spinning: refreshing }"></i>
        </button>

        <div class="user-menu">
          <button class="user-btn" @click="toggleUserMenu">
            <i class="fas fa-user-circle"></i>
            <span>用户</span>
            <i class="fas fa-chevron-down"></i>
          </button>

          <Transition name="dropdown">
            <div v-if="showUserMenu" class="user-dropdown">
              <a href="#" class="dropdown-item">
                <i class="fas fa-cog"></i>
                <span>设置</span>
              </a>
              <a href="#" class="dropdown-item" @click.prevent="logout">
                <i class="fas fa-sign-out-alt"></i>
                <span>退出登录</span>
              </a>
            </div>
          </Transition>
        </div>
      </div>
    </div>
  </header>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAppStore } from '@/stores/app'
import { useDashboardStore } from '@/stores/dashboard'

const route = useRoute()
const router = useRouter()
const appStore = useAppStore()
const dashboardStore = useDashboardStore()

const showUserMenu = ref(false)
const refreshing = ref(false)

const currentPageTitle = computed(() => {
  return route.meta.title || 'FinLoom'
})

const statusText = computed(() => {
  const status = appStore.appStatus
  return {
    healthy: '系统正常',
    unhealthy: '服务异常',
    loading: '加载中...',
    ready: '就绪'
  }[status] || '未知'
})

function toggleUserMenu() {
  showUserMenu.value = !showUserMenu.value
}

async function refreshData() {
  refreshing.value = true
  try {
    await dashboardStore.refreshAll()
  } finally {
    setTimeout(() => {
      refreshing.value = false
    }, 1000)
  }
}

function logout() {
  localStorage.removeItem('finloom_auth')
  localStorage.removeItem('finloom_token')
  router.push('/login')
}
</script>

<style lang="scss" scoped>
.top-navbar {
  height: 70px;
  background: rgba(255, 255, 255, 0.9);
  backdrop-filter: blur(20px);
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
  position: sticky;
  top: 0;
  z-index: 100;
}

.navbar-content {
  height: 100%;
  padding: 0 2rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.navbar-left {
  .page-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0;
  }
}

.navbar-right {
  display: flex;
  align-items: center;
  gap: 1.5rem;
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: rgba(0, 0, 0, 0.05);
  border-radius: 20px;
  font-size: 0.875rem;
  font-weight: 500;

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    animation: pulse 2s infinite;
  }

  &.ready .status-dot,
  &.healthy .status-dot {
    background: #10b981;
  }

  &.loading .status-dot {
    background: #f59e0b;
  }

  &.unhealthy .status-dot {
    background: #ef4444;
  }
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}

.icon-btn {
  width: 40px;
  height: 40px;
  border: none;
  background: rgba(59, 130, 246, 0.1);
  color: #3b82f6;
  border-radius: 10px;
  cursor: pointer;
  transition: all 0.3s ease;
  font-size: 1.125rem;

  &:hover {
    background: rgba(59, 130, 246, 0.2);
    transform: scale(1.05);
  }

  i.spinning {
    animation: spin 1s linear infinite;
  }
}

@keyframes spin {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

.user-menu {
  position: relative;

  .user-btn {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background: transparent;
    border: none;
    border-radius: 10px;
    color: #475569;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.3s ease;

    &:hover {
      background: rgba(0, 0, 0, 0.05);
    }

    i:first-child {
      font-size: 1.5rem;
    }

    i:last-child {
      font-size: 0.75rem;
    }
  }

  .user-dropdown {
    position: absolute;
    top: calc(100% + 0.5rem);
    right: 0;
    min-width: 200px;
    background: white;
    border-radius: 12px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
    overflow: hidden;
    z-index: 1000;
  }

  .dropdown-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 1rem 1.25rem;
    color: #475569;
    text-decoration: none;
    transition: all 0.2s ease;

    &:hover {
      background: rgba(59, 130, 246, 0.1);
      color: #3b82f6;
    }

    i {
      width: 20px;
      text-align: center;
    }
  }
}

.dropdown-enter-active,
.dropdown-leave-active {
  transition: all 0.3s ease;
}

.dropdown-enter-from,
.dropdown-leave-to {
  opacity: 0;
  transform: translateY(-10px);
}
</style>


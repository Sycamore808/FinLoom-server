import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'splash',
      component: () => import('@/views/SplashView.vue'),
      meta: { title: 'FinLoom - 智能量化投资平台' }
    },
    {
      path: '/home',
      name: 'home',
      component: () => import('@/views/HomeView.vue'),
      meta: { title: 'FinLoom - 首页' }
    },
    {
      path: '/login',
      name: 'login',
      component: () => import('@/views/LoginView.vue'),
      meta: { title: 'FinLoom - 登录' }
    },
    {
      path: '/dashboard',
      name: 'dashboard',
      component: () => import('@/layouts/DashboardLayout.vue'),
      meta: { 
        title: 'FinLoom - 仪表盘',
        requiresAuth: true 
      },
      children: [
        {
          path: '',
          name: 'dashboard-overview',
          component: () => import('@/views/dashboard/OverviewView.vue'),
          meta: { title: '仪表盘概览' }
        },
        {
          path: 'portfolio',
          name: 'dashboard-portfolio',
          component: () => import('@/views/dashboard/PortfolioView.vue'),
          meta: { title: '投资组合' }
        },
        {
          path: 'trades',
          name: 'dashboard-trades',
          component: () => import('@/views/dashboard/TradesView.vue'),
          meta: { title: '交易记录' }
        },
        {
          path: 'backtest',
          name: 'dashboard-backtest',
          component: () => import('@/views/dashboard/BacktestView.vue'),
          meta: { title: '策略回测' }
        },
        {
          path: 'data',
          name: 'dashboard-data',
          component: () => import('@/views/dashboard/DataView.vue'),
          meta: { title: '数据管理' }
        },
        {
          path: 'market',
          name: 'dashboard-market',
          component: () => import('@/views/dashboard/MarketView.vue'),
          meta: { title: '市场分析' }
        },
        {
          path: 'chat',
          name: 'dashboard-chat',
          component: () => import('@/views/dashboard/ChatView.vue'),
          meta: { title: 'AI对话' }
        },
        {
          path: 'strategy',
          name: 'dashboard-strategy',
          component: () => import('@/views/dashboard/StrategyView.vue'),
          meta: { title: '策略模式' }
        },
        {
          path: 'reports',
          name: 'dashboard-reports',
          component: () => import('@/views/dashboard/ReportsView.vue'),
          meta: { title: '报告中心' }
        },
        {
          path: 'notifications',
          name: 'dashboard-notifications',
          component: () => import('@/views/dashboard/NotificationsView.vue'),
          meta: { title: '通知中心' }
        },
        {
          path: 'settings',
          name: 'dashboard-settings',
          component: () => import('@/views/dashboard/SettingsView.vue'),
          meta: { title: '系统设置' }
        }
      ]
    },
    {
      path: '/test',
      name: 'test',
      component: () => import('@/views/TestView.vue'),
      meta: { title: '测试页面' }
    },
    {
      path: '/:pathMatch(.*)*',
      name: 'not-found',
      component: () => import('@/views/NotFoundView.vue'),
      meta: { title: '页面未找到' }
    }
  ]
})

// 路由守卫
router.beforeEach((to, from, next) => {
  // 更新页面标题
  document.title = to.meta.title || 'FinLoom'
  
  // 认证检查（简化版）
  if (to.meta.requiresAuth) {
    const isAuthenticated = localStorage.getItem('finloom_auth') === 'true'
    if (!isAuthenticated) {
      next({ name: 'login', query: { redirect: to.fullPath } })
      return
    }
  }
  
  next()
})

export default router


<template>
  <v-container fluid class="strategy-view pa-6">
    <!-- 进度条 - Material 3 增强 -->
    <v-card variant="flat" class="mb-6" color="primary-container">
      <v-card-text class="pa-6">
        <v-stepper v-model="currentStep" alt-labels bg-color="transparent" flat>
          <v-stepper-header>
            <v-stepper-item 
              value="1" 
              title="需求分析"
              :complete="parseInt(currentStep) > 1"
              color="primary"
            ></v-stepper-item>
            <v-divider></v-divider>
            <v-stepper-item 
              value="2" 
              title="策略生成"
              :complete="parseInt(currentStep) > 2"
              color="primary"
            ></v-stepper-item>
            <v-divider></v-divider>
            <v-stepper-item 
              value="3" 
              title="回测优化"
              :complete="parseInt(currentStep) > 3"
              color="primary"
            ></v-stepper-item>
            <v-divider></v-divider>
            <v-stepper-item 
              value="4" 
              title="代码生成"
              color="primary"
            ></v-stepper-item>
          </v-stepper-header>
        </v-stepper>
      </v-card-text>
    </v-card>

    <!-- 步骤内容 -->
    <v-window v-model="currentStep">
      <!-- 步骤1: 需求分析 -->
      <v-window-item value="1">
        <v-card variant="elevated">
          <v-card-title class="text-h5 font-weight-bold pa-6">
            投资需求分析
          </v-card-title>
          <v-card-subtitle class="px-6 pb-4">请告诉我们您的投资目标和偏好</v-card-subtitle>
          <v-card-text>
            <v-row>
              <v-col cols="12" md="6">
                <v-card variant="flat" class="bg-primary-container pa-4 mb-4">
                  <div class="d-flex align-center mb-4">
                    <v-avatar color="primary" variant="tonal" size="48" class="mr-3">
                      <v-icon size="28">mdi-bullseye</v-icon>
                    </v-avatar>
                    <h3 class="text-h6 font-weight-bold">投资目标</h3>
        </div>
                  <v-text-field
                    v-model.number="form.targetReturn"
                    label="收益目标 (%/年)"
                    type="number"
                    prepend-inner-icon="mdi-percent"
                    class="mb-4"
                    bg-color="surface"
                  ></v-text-field>
                  <v-select
                    v-model="form.investmentPeriod"
                    :items="[
                      { title: '短期 (1-3个月)', value: 'short' },
                      { title: '中期 (3-12个月)', value: 'medium' },
                      { title: '长期 (1年以上)', value: 'long' }
                    ]"
                    label="投资期限"
                    prepend-inner-icon="mdi-calendar"
                    class="mb-4"
                    bg-color="surface"
                  ></v-select>
                  <v-text-field
                    v-model.number="form.initialCapital"
                    label="初始资金 (万元)"
                    type="number"
                    prepend-inner-icon="mdi-currency-usd"
                    bg-color="surface"
                  ></v-text-field>
                </v-card>
              </v-col>

              <v-col cols="12" md="6">
                <v-card variant="flat" class="bg-secondary-container pa-4 mb-4">
                  <div class="d-flex align-center mb-4">
                    <v-avatar color="secondary" variant="tonal" size="48" class="mr-3">
                      <v-icon size="28">mdi-shield-check</v-icon>
                    </v-avatar>
                    <h3 class="text-h6 font-weight-bold">风险偏好</h3>
            </div>
                  <v-chip-group
                    v-model="form.riskPreference"
                    mandatory
                    class="mb-4"
                  >
                    <v-chip 
                      value="conservative" 
                      :variant="form.riskPreference === 'conservative' ? 'elevated' : 'flat'" 
                      :color="form.riskPreference === 'conservative' ? 'success' : undefined"
                      :class="form.riskPreference === 'conservative' ? 'text-white' : 'bg-success-container'"
                      size="large" 
                      class="font-weight-bold"
                    >
                      <v-icon start>mdi-shield</v-icon>
                      保守型
                    </v-chip>
                    <v-chip 
                      value="moderate" 
                      :variant="form.riskPreference === 'moderate' ? 'elevated' : 'flat'" 
                      :color="form.riskPreference === 'moderate' ? 'primary' : undefined"
                      :class="form.riskPreference === 'moderate' ? 'text-white' : 'bg-primary-container'"
                      size="large" 
                      class="font-weight-bold"
                    >
                      <v-icon start>mdi-scale-balance</v-icon>
                      稳健型
                    </v-chip>
                    <v-chip 
                      value="aggressive" 
                      :variant="form.riskPreference === 'aggressive' ? 'elevated' : 'flat'" 
                      :color="form.riskPreference === 'aggressive' ? 'error' : undefined"
                      :class="form.riskPreference === 'aggressive' ? 'text-white' : 'bg-error-container'"
                      size="large" 
                      class="font-weight-bold"
                    >
                      <v-icon start>mdi-rocket-launch</v-icon>
                      进取型
                    </v-chip>
                  </v-chip-group>
                  <v-text-field
                    v-model.number="form.maxDrawdown"
                    label="最大回撤容忍度 (%)"
                    type="number"
                    prepend-inner-icon="mdi-arrow-down"
                    bg-color="surface"
                  ></v-text-field>
                </v-card>
              </v-col>

              <v-col cols="12" md="6">
                <v-card variant="flat" class="bg-tertiary-container pa-4 mb-4">
                  <div class="d-flex align-center mb-4">
                    <v-avatar color="tertiary" variant="tonal" size="48" class="mr-3">
                      <v-icon size="28">mdi-tag-multiple</v-icon>
                    </v-avatar>
                    <h3 class="text-h6 font-weight-bold">偏好行业</h3>
                  </div>
                  <v-chip-group v-model="form.preferredTags" multiple column>
                    <v-chip 
                      v-for="tag in allTags" 
                      :key="tag" 
                      :value="tag" 
                      variant="elevated"
                      color="tertiary"
                      size="large"
                    >
                      {{ tag }}
                    </v-chip>
                  </v-chip-group>
                </v-card>
              </v-col>

              <v-col cols="12" md="6">
                <v-card variant="flat" class="bg-success-container pa-4 mb-4">
                  <div class="d-flex align-center mb-4">
                    <v-avatar color="success" variant="tonal" size="48" class="mr-3">
                      <v-icon size="28">mdi-cog</v-icon>
                    </v-avatar>
                    <h3 class="text-h6 font-weight-bold">策略偏好</h3>
                  </div>
                  <v-select
                    v-model="form.strategyType"
                    :items="[
                      { title: '价值投资', value: 'value' },
                      { title: '成长投资', value: 'growth' },
                      { title: '动量策略', value: 'momentum' },
                      { title: '均值回归', value: 'mean_reversion' }
                    ]"
                    label="策略类型"
                    prepend-inner-icon="mdi-strategy"
                    class="mb-4"
                    bg-color="surface"
                  ></v-select>
                  <v-select
                    v-model="form.tradingFrequency"
                    :items="[
                      { title: '日内交易', value: 'daily' },
                      { title: '周级调仓', value: 'weekly' },
                      { title: '月度调仓', value: 'monthly' }
                    ]"
                    label="交易频率"
                    prepend-inner-icon="mdi-clock-outline"
                    bg-color="surface"
                  ></v-select>
                </v-card>
              </v-col>

              <v-col cols="12">
                <v-card variant="flat" class="bg-warning-container pa-4">
                  <div class="d-flex align-center mb-4">
                    <v-avatar color="warning" variant="tonal" size="48" class="mr-3">
                      <v-icon size="28">mdi-comment-text</v-icon>
                    </v-avatar>
                    <div>
                      <h3 class="text-h6 font-weight-bold">补充需求说明</h3>
                      <p class="text-caption text-medium-emphasis mb-0">提供更详细的需求可以帮助AI生成更符合您期望的策略</p>
                    </div>
                  </div>
                  <v-textarea
                    v-model="form.additionalRequirements"
                    label="其他特殊需求（可选）"
                    rows="4"
                    placeholder="例如：希望避开某些行业、关注特定市场事件、特殊的止损要求等..."
                    bg-color="surface"
                  ></v-textarea>
                </v-card>
              </v-col>
            </v-row>
          </v-card-text>
          <v-card-actions class="px-6 pb-6">
            <v-spacer></v-spacer>
            <v-btn color="primary" size="large" @click="nextStep" prepend-icon="mdi-arrow-right">
              下一步：生成策略
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-window-item>

      <!-- 步骤2: 策略生成 -->
      <v-window-item value="2">
        <v-card variant="elevated">
          <v-card-title class="text-h5 font-weight-bold pa-6">AI策略生成</v-card-title>
          <v-card-subtitle class="px-6 pb-4">基于您的需求，我们正在为您生成个性化投资策略</v-card-subtitle>
          <v-card-text class="pa-10">
            <div v-if="!showResult">
              <v-card variant="flat" class="bg-primary-container pa-8 text-center">
                <v-avatar color="primary" size="120" class="mb-6">
                  <v-icon size="64">mdi-brain</v-icon>
                </v-avatar>
                <h3 class="text-h4 mb-4 font-weight-bold">AI正在分析您的需求...</h3>
                <v-progress-linear
                  v-model="generationProgress"
                  height="24"
                  color="primary"
                  rounded
                  class="mb-6"
                >
                  <strong class="text-white">{{ generationProgress }}%</strong>
                </v-progress-linear>
                <v-chip color="primary" size="large" variant="flat">
                  <v-icon start>mdi-cog</v-icon>
                  {{ statusText }}
                </v-chip>
              </v-card>
            </div>

            <div v-else>
              <v-alert 
                type="success" 
                variant="tonal" 
                prominent
                border="start"
                class="mb-6"
              >
                <template v-slot:prepend>
                  <v-icon size="48">mdi-check-circle</v-icon>
                </template>
                <div class="text-h6 font-weight-bold mb-2">策略生成成功！</div>
                <div>您的个性化投资策略已经准备就绪</div>
              </v-alert>
              <v-card variant="flat" class="bg-secondary-container">
                <v-card-title class="text-h5 pa-6 d-flex align-center">
                  <v-avatar color="secondary" variant="tonal" size="48" class="mr-3">
                    <v-icon>mdi-strategy</v-icon>
                  </v-avatar>
                  {{ result.title }}
                </v-card-title>
                <v-divider></v-divider>
                <v-card-text class="pa-6">
                  <p class="text-h6">{{ result.description }}</p>
                </v-card-text>
              </v-card>
            </div>
          </v-card-text>
          <v-card-actions class="px-6 pb-6">
            <v-btn variant="text" @click="prevStep" prepend-icon="mdi-arrow-left">
              上一步
            </v-btn>
            <v-spacer></v-spacer>
            <v-btn v-if="showResult" color="primary" size="large" @click="nextStep" prepend-icon="mdi-arrow-right">
              下一步：回测优化
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-window-item>

      <!-- 步骤3: 回测优化 -->
      <v-window-item value="3">
        <v-card variant="elevated">
          <v-card-title class="text-h5 font-weight-bold pa-6">回测优化</v-card-title>
          <v-card-subtitle class="px-6 pb-4">使用历史数据测试策略表现</v-card-subtitle>
          <v-card-text class="pa-10">
            <v-card variant="flat" class="bg-tertiary-container pa-8 text-center">
              <v-avatar color="tertiary" size="120" class="mb-6">
                <v-icon size="64">mdi-chart-timeline-variant</v-icon>
              </v-avatar>
              <h3 class="text-h5 mb-4 font-weight-bold">回测功能</h3>
              <p class="text-body-1">该功能正在开发中，敬请期待...</p>
              <v-chip color="tertiary" size="large" variant="tonal" class="mt-4">
                <v-icon start>mdi-clock-outline</v-icon>
                Coming Soon
              </v-chip>
            </v-card>
          </v-card-text>
          <v-card-actions class="px-6 pb-6">
            <v-btn variant="text" @click="prevStep" prepend-icon="mdi-arrow-left">
              上一步
            </v-btn>
            <v-spacer></v-spacer>
            <v-btn color="primary" size="large" @click="nextStep" prepend-icon="mdi-arrow-right">
              下一步：代码生成
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-window-item>

      <!-- 步骤4: 代码生成 -->
      <v-window-item value="4">
        <v-card variant="elevated">
          <v-card-title class="text-h5 font-weight-bold pa-6">代码生成</v-card-title>
          <v-card-subtitle class="px-6 pb-4">生成可执行的策略代码</v-card-subtitle>
          <v-card-text class="pa-10">
            <v-card variant="flat" class="bg-success-lighten-4 pa-8 text-center">
              <v-avatar color="success" size="120" class="mb-6">
                <v-icon size="64">mdi-code-braces</v-icon>
              </v-avatar>
              <h3 class="text-h5 mb-4 font-weight-bold">代码生成</h3>
              <p class="text-body-1">该功能正在开发中，敬请期待...</p>
              <v-chip color="success" size="large" variant="tonal" class="mt-4">
                <v-icon start>mdi-clock-outline</v-icon>
                Coming Soon
              </v-chip>
            </v-card>
          </v-card-text>
          <v-card-actions class="px-6 pb-6">
            <v-btn variant="text" @click="prevStep" prepend-icon="mdi-arrow-left">
              上一步
            </v-btn>
            <v-spacer></v-spacer>
            <v-btn color="success" size="large" prepend-icon="mdi-content-save">
              保存策略
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-window-item>
    </v-window>
  </v-container>
</template>

<script setup>
import { ref } from 'vue'

const currentStep = ref('1')
const allTags = ['科技', '金融', '医药', '消费', '能源', '制造']

const form = ref({
  targetReturn: 15,
  investmentPeriod: 'medium',
  initialCapital: 100,
  riskPreference: 'moderate',
  maxDrawdown: 20,
  preferredTags: [],
  strategyType: 'value',
  tradingFrequency: 'weekly',
  additionalRequirements: ''
})

const showResult = ref(false)
const generationProgress = ref(0)
const statusText = ref('正在分析市场数据...')
const result = ref({
  title: '价值投资策略 V1.0',
  description: '基于您的需求生成的稳健型价值投资策略'
})

let progressInterval = null

function nextStep() {
  const step = parseInt(currentStep.value)
  if (step < 4) {
    currentStep.value = String(step + 1)
    
    if (step === 1) {
    startGeneration()
    }
  }
}

function prevStep() {
  const step = parseInt(currentStep.value)
  if (step > 1) {
    currentStep.value = String(step - 1)
  }
}

function startGeneration() {
  showResult.value = false
  generationProgress.value = 0
  
  progressInterval = setInterval(() => {
    generationProgress.value += 10
    
    if (generationProgress.value === 30) {
      statusText.value = '正在筛选合适的股票...'
    } else if (generationProgress.value === 60) {
      statusText.value = '正在优化参数...'
    } else if (generationProgress.value === 90) {
      statusText.value = '即将完成...'
    } else if (generationProgress.value >= 100) {
      clearInterval(progressInterval)
      showResult.value = true
    }
  }, 500)
}
</script>

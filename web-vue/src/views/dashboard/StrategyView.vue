<template>
  <v-container fluid class="strategy-view pa-6">
    <!-- 进度条 -->
    <v-card class="mb-6" rounded="xl">
      <v-card-text class="pa-6">
        <v-stepper v-model="currentStep" alt-labels flat>
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
        <v-card rounded="xl">
          <v-card-title class="text-h5 font-weight-medium pa-6">
            <v-icon start>mdi-chart-line</v-icon>
            投资需求分析
          </v-card-title>
          <v-card-subtitle class="px-6 pb-4">请告诉我们您的投资目标和偏好</v-card-subtitle>
          <v-card-text>
            <v-row>
              <v-col cols="12" md="6">
                <v-sheet rounded="lg" class="pa-4 mb-4" color="primary-container" style="background-color: rgba(30, 136, 229, 0.06) !important;">
                  <div class="d-flex align-center mb-4">
                    <v-icon size="24" class="mr-3 text-primary">mdi-target</v-icon>
                    <h3 class="text-h6 font-weight-medium">投资目标</h3>
                  </div>
                  <v-text-field
                    v-model.number="form.targetReturn"
                    label="收益目标 (%/年)"
                    type="number"
                    prepend-inner-icon="mdi-percent"
                    class="mb-4"
                    variant="filled"
                    density="comfortable"
                    rounded="lg"
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
                    variant="filled"
                    density="comfortable"
                    rounded="lg"
                  ></v-select>
                  <v-text-field
                    v-model.number="form.initialCapital"
                    label="初始资金 (万元)"
                    type="number"
                    prepend-inner-icon="mdi-currency-usd"
                    variant="filled"
                    density="comfortable"
                    rounded="lg"
                  ></v-text-field>
                </v-sheet>
              </v-col>

              <v-col cols="12" md="6">
                <v-sheet rounded="lg" class="pa-4 mb-4" color="secondary-container" style="background-color: rgba(0, 172, 193, 0.06) !important;">
                  <div class="d-flex align-center mb-4">
                    <v-icon size="24" class="mr-3 text-secondary">mdi-shield-check</v-icon>
                    <h3 class="text-h6 font-weight-medium">风险偏好</h3>
                  </div>
                  <v-chip-group
                    v-model="form.riskPreference"
                    mandatory
                    class="mb-4"
                  >
                    <v-chip 
                      value="conservative" 
                      :variant="form.riskPreference === 'conservative' ? 'elevated' : 'outlined'" 
                      :color="form.riskPreference === 'conservative' ? 'success' : undefined"
                      size="large" 
                      class="font-weight-medium"
                      rounded="lg"
                    >
                      <v-icon start>mdi-shield</v-icon>
                      保守型
                    </v-chip>
                    <v-chip 
                      value="moderate" 
                      :variant="form.riskPreference === 'moderate' ? 'elevated' : 'outlined'" 
                      :color="form.riskPreference === 'moderate' ? 'primary' : undefined"
                      size="large" 
                      class="font-weight-medium"
                      rounded="lg"
                    >
                      <v-icon start>mdi-scale-balance</v-icon>
                      稳健型
                    </v-chip>
                    <v-chip 
                      value="aggressive" 
                      :variant="form.riskPreference === 'aggressive' ? 'elevated' : 'outlined'" 
                      :color="form.riskPreference === 'aggressive' ? 'error' : undefined"
                      size="large" 
                      class="font-weight-medium"
                      rounded="lg"
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
                    variant="filled"
                    density="comfortable"
                    rounded="lg"
                  ></v-text-field>
                </v-sheet>
              </v-col>

              <v-col cols="12" md="6">
                <v-sheet rounded="lg" class="pa-4 mb-4" color="tertiary-container" style="background-color: rgba(123, 97, 255, 0.06) !important;">
                  <div class="d-flex align-center mb-4">
                    <v-icon size="24" class="mr-3 text-tertiary">mdi-tag-multiple</v-icon>
                    <h3 class="text-h6 font-weight-medium">偏好行业</h3>
                  </div>
                  <v-chip-group v-model="form.preferredTags" multiple column>
                    <v-chip 
                      v-for="tag in allTags" 
                      :key="tag" 
                      :value="tag" 
                      variant="outlined"
                      color="tertiary"
                      size="large"
                      rounded="lg"
                    >
                      {{ tag }}
                    </v-chip>
                  </v-chip-group>
                </v-sheet>
              </v-col>

              <v-col cols="12" md="6">
                <v-sheet rounded="lg" class="pa-4 mb-4" color="success-container" style="background-color: rgba(0, 168, 107, 0.08) !important;">
                  <div class="d-flex align-center mb-4">
                    <v-icon size="24" class="mr-3 text-success">mdi-cog</v-icon>
                    <h3 class="text-h6 font-weight-medium">策略偏好</h3>
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
                    variant="filled"
                    density="comfortable"
                    rounded="lg"
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
                    variant="filled"
                    density="comfortable"
                    rounded="lg"
                  ></v-select>
                </v-sheet>
              </v-col>

              <v-col cols="12">
                <v-sheet rounded="lg" class="pa-4" color="warning-container" style="background-color: rgba(255, 152, 0, 0.06) !important;">
                  <div class="d-flex align-center mb-4">
                    <v-icon size="24" class="mr-3 text-warning">mdi-comment-text</v-icon>
                    <div>
                      <h3 class="text-h6 font-weight-medium">补充需求说明</h3>
                      <p class="text-caption text-medium-emphasis mb-0">提供更详细的需求可以帮助AI生成更符合您期望的策略</p>
                    </div>
                  </div>
                  <v-textarea
                    v-model="form.additionalRequirements"
                    label="其他特殊需求（可选）"
                    rows="4"
                    placeholder="例如：希望避开某些行业、关注特定市场事件、特殊的止损要求等..."
                    variant="filled"
                    density="comfortable"
                    rounded="lg"
                  ></v-textarea>
                </v-sheet>
              </v-col>
            </v-row>
          </v-card-text>
          <v-card-actions class="px-6 pb-6">
            <v-spacer></v-spacer>
            <v-btn color="primary" size="large" @click="nextStep" prepend-icon="mdi-arrow-right" rounded="pill" variant="flat">
              下一步：生成策略
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-window-item>

      <!-- 步骤2: 策略生成 -->
      <v-window-item value="2">
        <v-card rounded="xl">
          <v-card-title class="text-h5 font-weight-medium pa-6">
            <v-icon start>mdi-brain</v-icon>
            AI策略生成
          </v-card-title>
          <v-card-subtitle class="px-6 pb-4">基于您的需求，我们正在为您生成个性化投资策略</v-card-subtitle>
          <v-card-text class="pa-10">
            <div v-if="!showResult">
              <v-sheet rounded="lg" class="pa-8 text-center" color="primary-container" style="background-color: rgba(30, 136, 229, 0.08) !important;">
                <v-icon size="64" class="mb-6 text-primary">mdi-brain</v-icon>
                <h3 class="text-h4 mb-4 font-weight-medium">AI正在分析您的需求...</h3>
                <v-progress-linear
                  v-model="generationProgress"
                  height="8"
                  color="primary"
                  rounded
                  class="mb-6"
                >
                  <strong class="text-primary">{{ generationProgress }}%</strong>
                </v-progress-linear>
                <v-chip color="primary" size="large" variant="outlined" rounded="lg">
                  <v-icon start>mdi-cog</v-icon>
                  {{ statusText }}
                </v-chip>
              </v-sheet>
            </div>

            <div v-else>
              <v-alert 
                type="success" 
                variant="tonal" 
                border="start"
                class="mb-6"
                rounded="lg"
              >
                <template v-slot:prepend>
                  <v-icon size="32">mdi-check-circle</v-icon>
                </template>
                <div class="text-h6 font-weight-medium mb-2">策略生成成功！</div>
                <div>您的个性化投资策略已经准备就绪</div>
              </v-alert>
              <v-sheet rounded="lg" color="secondary-container" style="background-color: rgba(0, 172, 193, 0.08) !important;">
                <v-card-title class="text-h5 pa-6 d-flex align-center">
                  <v-icon size="24" class="mr-3 text-secondary">mdi-strategy</v-icon>
                  {{ result.title }}
                </v-card-title>
                <v-divider></v-divider>
                <v-card-text class="pa-6">
                  <p class="text-h6">{{ result.description }}</p>
                </v-card-text>
              </v-sheet>
            </div>
          </v-card-text>
          <v-card-actions class="px-6 pb-6">
            <v-btn variant="text" @click="prevStep" prepend-icon="mdi-arrow-left" rounded="pill">
              上一步
            </v-btn>
            <v-spacer></v-spacer>
            <v-btn v-if="showResult" color="primary" size="large" @click="nextStep" prepend-icon="mdi-arrow-right" rounded="pill" variant="flat">
              下一步：回测优化
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-window-item>

      <!-- 步骤3: 回测优化 -->
      <v-window-item value="3">
        <v-card rounded="xl">
          <v-card-title class="text-h5 font-weight-medium pa-6">
            <v-icon start>mdi-chart-timeline-variant</v-icon>
            回测优化
          </v-card-title>
          <v-card-subtitle class="px-6 pb-4">使用历史数据测试策略表现</v-card-subtitle>
          <v-card-text class="pa-10">
            <v-sheet rounded="lg" class="pa-8 text-center" color="tertiary-container" style="background-color: rgba(123, 97, 255, 0.08) !important;">
              <v-icon size="64" class="mb-6 text-tertiary">mdi-chart-timeline-variant</v-icon>
              <h3 class="text-h5 mb-4 font-weight-medium">回测功能</h3>
              <p class="text-body-1">该功能正在开发中，敬请期待...</p>
              <v-chip color="tertiary" size="large" variant="outlined" class="mt-4" rounded="lg">
                <v-icon start>mdi-clock-outline</v-icon>
                Coming Soon
              </v-chip>
            </v-sheet>
          </v-card-text>
          <v-card-actions class="px-6 pb-6">
            <v-btn variant="text" @click="prevStep" prepend-icon="mdi-arrow-left" rounded="pill">
              上一步
            </v-btn>
            <v-spacer></v-spacer>
            <v-btn color="primary" size="large" @click="nextStep" prepend-icon="mdi-arrow-right" rounded="pill" variant="flat">
              下一步：代码生成
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-window-item>

      <!-- 步骤4: 代码生成 -->
      <v-window-item value="4">
        <v-card rounded="xl">
          <v-card-title class="text-h5 font-weight-medium pa-6">
            <v-icon start>mdi-code-braces</v-icon>
            代码生成
          </v-card-title>
          <v-card-subtitle class="px-6 pb-4">生成可执行的策略代码</v-card-subtitle>
          <v-card-text class="pa-10">
            <v-sheet rounded="lg" class="pa-8 text-center" color="success-container" style="background-color: rgba(0, 168, 107, 0.08) !important;">
              <v-icon size="64" class="mb-6 text-success">mdi-code-braces</v-icon>
              <h3 class="text-h5 mb-4 font-weight-medium">代码生成</h3>
              <p class="text-body-1">该功能正在开发中，敬请期待...</p>
              <v-chip color="success" size="large" variant="outlined" class="mt-4" rounded="lg">
                <v-icon start>mdi-clock-outline</v-icon>
                Coming Soon
              </v-chip>
            </v-sheet>
          </v-card-text>
          <v-card-actions class="px-6 pb-6">
            <v-btn variant="text" @click="prevStep" prepend-icon="mdi-arrow-left" rounded="pill">
              上一步
            </v-btn>
            <v-spacer></v-spacer>
            <v-btn color="success" size="large" prepend-icon="mdi-content-save" rounded="pill" variant="flat">
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

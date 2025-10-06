/**
 * 策略模式模块 - AUE3架构实现
 */

// ==================== Effects 副作用层 ====================
const StrategyEffects = {
    /**
     * 生成策略
     */
    async generateStrategy(payload, state, app) {
        const { requirements } = payload;
        
        // 模拟API调用，实际应该调用真实API
        // const response = await FinLoomAPI.strategy.generate(requirements);
        
        // 模拟生成进度
        const steps = [
            { progress: 20, text: '正在分析市场数据...', delay: 800 },
            { progress: 40, text: '筛选因子指标...', delay: 800 },
            { progress: 60, text: '构建策略模型...', delay: 800 },
            { progress: 80, text: '优化参数配置...', delay: 800 },
            { progress: 100, text: '生成完成！', delay: 500 }
        ];

        for (const step of steps) {
            await new Promise(resolve => setTimeout(resolve, step.delay));
            app.setState({
                generationProgress: step.progress,
                generationStatus: step.text
            });
            app.render('generationProgress');
        }

        // 模拟生成的策略
        const strategy = {
            id: 'strategy_' + Date.now(),
            name: '动量价值组合策略',
            description: '基于动量因子和价值因子的双因子选股策略，适合中长期投资',
            factors: ['动量因子', '市盈率', 'ROE', '市净率'],
            expectedReturn: requirements.targetReturn || 15,
            riskLevel: requirements.riskProfile || 'moderate',
            rebalanceFrequency: requirements.tradingFrequency || 'monthly',
            code: `# FinLoom策略代码
import pandas as pd
import numpy as np

class MomentumValueStrategy:
    def __init__(self):
        self.name = "动量价值组合策略"
        self.version = "1.0"
        
    def select_stocks(self, universe, date):
        """选股逻辑"""
        # 计算动量因子
        momentum = universe['close'].pct_change(20)
        
        # 计算价值因子
        pe_ratio = universe['price'] / universe['earnings']
        pb_ratio = universe['price'] / universe['book_value']
        
        # 综合评分
        scores = momentum * 0.6 + (1 / pe_ratio) * 0.2 + (1 / pb_ratio) * 0.2
        
        # 选择前20只股票
        selected = scores.nlargest(20)
        
        return selected.index.tolist()
    
    def calculate_weights(self, stocks):
        """计算权重"""
        # 等权重配置
        return {stock: 1.0/len(stocks) for stock in stocks}
    
    def rebalance(self, portfolio, date):
        """调仓逻辑"""
        new_stocks = self.select_stocks(universe, date)
        new_weights = self.calculate_weights(new_stocks)
        return new_weights`,
            backtest: {
                totalReturn: 45.6,
                annualReturn: 18.3,
                sharpeRatio: 1.85,
                maxDrawdown: -12.5,
                winRate: 62.5,
                trades: 156
            }
        };

        return strategy;
    },

    /**
     * 保存策略
     */
    async saveStrategy(payload, state, app) {
        const { strategy } = payload;
        
        // 保存到本地存储
        const savedStrategies = AUE3Utils.storage.get('saved_strategies', []);
        savedStrategies.push({
            ...strategy,
            savedAt: new Date().toISOString()
        });
        AUE3Utils.storage.set('saved_strategies', savedStrategies);

        // 实际应该保存到服务器
        // await FinLoomAPI.strategy.save(strategy);

        return { success: true };
    },

    /**
     * 加载保存的策略草稿
     */
    loadDraft(payload, state, app) {
        const draft = AUE3Utils.storage.get('strategy_draft', null);
        return draft;
    },

    /**
     * 保存策略草稿
     */
    saveDraft(payload, state, app) {
        const { formData } = payload;
        AUE3Utils.storage.set('strategy_draft', {
            ...formData,
            savedAt: new Date().toISOString()
        });
        return { success: true };
    }
};

// ==================== Actions 动作层 ====================
const StrategyActions = {
    /**
     * 初始化策略模式
     */
    async initialize(payload, state, app) {
        // 加载草稿
        const draft = await app.runEffect('loadDraft');
        
        return {
            state: {
                currentStep: 1,
                formData: draft || {
                    targetReturn: 15,
                    investmentPeriod: 'medium',
                    initialCapital: 100,
                    riskProfile: 'moderate',
                    maxDrawdown: 20,
                    strategyType: 'momentum',
                    tradingFrequency: 'monthly',
                    selectedIndustries: ['科技']
                },
                generatedStrategy: null,
                generationProgress: 0,
                generationStatus: '',
                loading: false
            },
            render: 'formData'
        };
    },

    /**
     * 更新表单数据
     */
    updateFormData(payload, state, app) {
        const { field, value } = payload;
        
        return {
            state: {
                formData: {
                    ...state.formData,
                    [field]: value
                }
            }
        };
    },

    /**
     * 切换行业标签
     */
    toggleIndustry(payload, state, app) {
        const { industry } = payload;
        const industries = state.formData.selectedIndustries || [];
        
        const newIndustries = industries.includes(industry)
            ? industries.filter(i => i !== industry)
            : [...industries, industry];

        return {
            state: {
                formData: {
                    ...state.formData,
                    selectedIndustries: newIndustries
                }
            }
        };
    },

    /**
     * 下一步
     */
    async nextStep(payload, state, app) {
        const { currentStep } = state;

        // 验证当前步骤
        if (currentStep === 1) {
            const validation = this.validateStep1(state.formData);
            if (!validation.valid) {
                Components.toast(validation.message, 'warning');
                return;
            }

            // 保存草稿
            await app.runEffect('saveDraft', { formData: state.formData });
        }

        // 如果进入第2步，开始生成策略
        if (currentStep === 1) {
            app.setState({ 
                currentStep: 2,
                generationProgress: 0,
                generationStatus: '准备生成策略...'
            });
            app.render('stepChange');

            // 生成策略
            try {
                const strategy = await app.runEffect('generateStrategy', {
                    requirements: state.formData
                });

                return {
                    state: {
                        generatedStrategy: strategy,
                        generationProgress: 100
                    },
                    render: 'strategyResult'
                };
            } catch (error) {
                Components.toast('策略生成失败', 'error');
                return {
                    state: {
                        currentStep: 1
                    },
                    render: 'stepChange'
                };
            }
        } else {
            // 其他步骤直接跳转
            return {
                state: {
                    currentStep: Math.min(4, currentStep + 1)
                },
                render: 'stepChange'
            };
        }
    },

    /**
     * 上一步
     */
    prevStep(payload, state, app) {
        return {
            state: {
                currentStep: Math.max(1, state.currentStep - 1)
            },
            render: 'stepChange'
        };
    },

    /**
     * 验证步骤1
     */
    validateStep1(formData) {
        if (!formData.targetReturn || formData.targetReturn <= 0) {
            return { valid: false, message: '请输入有效的收益目标' };
        }
        if (!formData.initialCapital || formData.initialCapital <= 0) {
            return { valid: false, message: '请输入有效的初始资金' };
        }
        if (!formData.maxDrawdown || formData.maxDrawdown <= 0) {
            return { valid: false, message: '请输入有效的最大回撤容忍度' };
        }
        return { valid: true };
    },

    /**
     * 保存策略
     */
    async saveStrategy(payload, state, app) {
        try {
            await app.runEffect('saveStrategy', { 
                strategy: state.generatedStrategy 
            });
            Components.toast('策略已保存', 'success');
        } catch (error) {
            Components.toast('保存失败', 'error');
        }
    },

    /**
     * 复制代码
     */
    copyCode(payload, state, app) {
        if (!state.generatedStrategy || !state.generatedStrategy.code) {
            Components.toast('暂无代码可复制', 'warning');
            return;
        }

        navigator.clipboard.writeText(state.generatedStrategy.code)
            .then(() => {
                Components.toast('代码已复制到剪贴板', 'success');
            })
            .catch(() => {
                Components.toast('复制失败', 'error');
            });
    },

    /**
     * 下载代码
     */
    downloadCode(payload, state, app) {
        if (!state.generatedStrategy || !state.generatedStrategy.code) {
            Components.toast('暂无代码可下载', 'warning');
            return;
        }

        const blob = new Blob([state.generatedStrategy.code], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'momentum_value_strategy.py';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        Components.toast('代码已下载', 'success');
    }
};

// ==================== UI 界面层 ====================
const StrategyUI = {
    /**
     * 渲染表单数据
     */
    formData(state, app) {
        if (!state.formData) return;

        // 更新表单字段
        const fields = ['targetReturn', 'investmentPeriod', 'initialCapital', 
                       'riskProfile', 'maxDrawdown', 'strategyType', 'tradingFrequency'];
        
        fields.forEach(field => {
            const element = document.getElementById(field);
            if (element && state.formData[field] !== undefined) {
                element.value = state.formData[field];
            }
        });

        // 更新行业标签
        if (state.formData.selectedIndustries) {
            AUE3Utils.$$('.tag-item').forEach(tag => {
                const industry = tag.textContent.trim();
                if (state.formData.selectedIndustries.includes(industry)) {
                    tag.classList.add('active');
                } else {
                    tag.classList.remove('active');
                }
            });
        }
    },

    /**
     * 渲染步骤变化
     */
    stepChange(state, app) {
        // 更新步骤内容显示
        AUE3Utils.$$('.step-content').forEach(content => {
            content.classList.remove('active');
        });
        const currentContent = AUE3Utils.$('#step' + state.currentStep);
        if (currentContent) {
            currentContent.classList.add('active');
        }

        // 更新进度指示器
        AUE3Utils.$$('.step').forEach(step => {
            const stepNum = parseInt(step.dataset.step);
            if (stepNum < state.currentStep) {
                step.classList.add('completed');
                step.classList.remove('active');
            } else if (stepNum === state.currentStep) {
                step.classList.add('active');
                step.classList.remove('completed');
            } else {
                step.classList.remove('active', 'completed');
            }
        });
    },

    /**
     * 渲染生成进度
     */
    generationProgress(state, app) {
        const progressFill = AUE3Utils.$('#progressFill');
        const statusText = AUE3Utils.$('#statusText');

        if (progressFill) {
            progressFill.style.width = state.generationProgress + '%';
        }
        if (statusText) {
            statusText.textContent = state.generationStatus;
        }
    },

    /**
     * 渲染策略结果
     */
    strategyResult(state, app) {
        if (!state.generatedStrategy) return;

        const strategy = state.generatedStrategy;
        const resultContainer = AUE3Utils.$('#strategyResult');
        const statusContainer = AUE3Utils.$('#generationStatus');

        if (statusContainer) {
            statusContainer.style.display = 'none';
        }

        if (resultContainer) {
            resultContainer.style.display = 'block';
            resultContainer.innerHTML = '';

            // 策略信息卡片
            const infoCard = AUE3Utils.createElement('div', {
                className: 'strategy-card',
                style: {
                    background: 'white',
                    borderRadius: '20px',
                    padding: '2.5rem',
                    boxShadow: '0 8px 30px rgba(0,0,0,0.1)',
                    marginBottom: '2rem'
                }
            }, [
                AUE3Utils.createElement('h4', {
                    style: {
                        fontSize: '1.8rem',
                        fontWeight: '800',
                        color: '#0f172a',
                        marginBottom: '1rem'
                    }
                }, strategy.name),
                AUE3Utils.createElement('p', {
                    style: {
                        fontSize: '1.1rem',
                        color: '#64748b',
                        lineHeight: '1.8',
                        marginBottom: '2rem'
                    }
                }, strategy.description),
                AUE3Utils.createElement('div', {
                    style: {
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                        gap: '1.5rem'
                    }
                }, [
                    this.createMetricCard('预期年化收益', `${strategy.expectedReturn}%`, 'success'),
                    this.createMetricCard('风险等级', this.getRiskText(strategy.riskLevel), 'primary'),
                    this.createMetricCard('调仓频率', this.getFrequencyText(strategy.rebalanceFrequency), 'warning'),
                    this.createMetricCard('因子数量', `${strategy.factors.length}个`, 'primary')
                ])
            ]);

            resultContainer.appendChild(infoCard);

            // 回测结果卡片
            if (strategy.backtest) {
                const backtestCard = this.createBacktestCard(strategy.backtest);
                resultContainer.appendChild(backtestCard);
            }

            // 代码预览卡片
            if (strategy.code) {
                const codeCard = this.createCodeCard(strategy.code, app);
                resultContainer.appendChild(codeCard);
            }

            // 操作按钮
            const actions = AUE3Utils.createElement('div', {
                style: {
                    display: 'flex',
                    justifyContent: 'center',
                    gap: '1rem',
                    marginTop: '2rem'
                }
            }, [
                this.createButton('保存策略', 'fa-save', () => app.dispatch('saveStrategy')),
                this.createButton('下一步：回测优化', 'fa-arrow-right', () => app.dispatch('nextStep'))
            ]);

            resultContainer.appendChild(actions);
        }
    },

    /**
     * 创建指标卡片
     */
    createMetricCard(label, value, variant) {
        const colors = {
            success: '#10b981',
            primary: '#3b82f6',
            warning: '#f59e0b',
            danger: '#ef4444'
        };

        return AUE3Utils.createElement('div', {
            style: {
                padding: '1.5rem',
                background: '#f8fafc',
                borderRadius: '12px',
                borderLeft: `4px solid ${colors[variant]}`
            }
        }, [
            AUE3Utils.createElement('div', {
                style: {
                    fontSize: '0.85rem',
                    color: '#64748b',
                    marginBottom: '0.5rem',
                    fontWeight: '600'
                }
            }, label),
            AUE3Utils.createElement('div', {
                style: {
                    fontSize: '1.5rem',
                    fontWeight: '700',
                    color: colors[variant]
                }
            }, value)
        ]);
    },

    /**
     * 创建回测结果卡片
     */
    createBacktestCard(backtest) {
        return AUE3Utils.createElement('div', {
            className: 'strategy-card',
            style: {
                background: 'white',
                borderRadius: '20px',
                padding: '2.5rem',
                boxShadow: '0 8px 30px rgba(0,0,0,0.1)',
                marginBottom: '2rem'
            }
        }, [
            AUE3Utils.createElement('h4', {
                style: {
                    fontSize: '1.5rem',
                    fontWeight: '800',
                    color: '#0f172a',
                    marginBottom: '2rem'
                }
            }, '回测结果'),
            AUE3Utils.createElement('div', {
                style: {
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
                    gap: '1rem'
                }
            }, [
                this.createMetricCard('累计收益', `${backtest.totalReturn}%`, 'success'),
                this.createMetricCard('年化收益', `${backtest.annualReturn}%`, 'success'),
                this.createMetricCard('夏普比率', backtest.sharpeRatio.toFixed(2), 'primary'),
                this.createMetricCard('最大回撤', `${backtest.maxDrawdown}%`, 'danger'),
                this.createMetricCard('胜率', `${backtest.winRate}%`, 'primary'),
                this.createMetricCard('交易次数', `${backtest.trades}次`, 'warning')
            ])
        ]);
    },

    /**
     * 创建代码卡片
     */
    createCodeCard(code, app) {
        return AUE3Utils.createElement('div', {
            className: 'strategy-card',
            style: {
                background: 'white',
                borderRadius: '20px',
                padding: '2.5rem',
                boxShadow: '0 8px 30px rgba(0,0,0,0.1)',
                marginBottom: '2rem'
            }
        }, [
            AUE3Utils.createElement('div', {
                style: {
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: '1.5rem'
                }
            }, [
                AUE3Utils.createElement('h4', {
                    style: {
                        fontSize: '1.5rem',
                        fontWeight: '800',
                        color: '#0f172a'
                    }
                }, '策略代码'),
                AUE3Utils.createElement('div', {
                    style: {
                        display: 'flex',
                        gap: '0.5rem'
                    }
                }, [
                    this.createIconButton('fa-copy', () => app.dispatch('copyCode')),
                    this.createIconButton('fa-download', () => app.dispatch('downloadCode'))
                ])
            ]),
            AUE3Utils.createElement('pre', {
                style: {
                    background: '#0f172a',
                    color: '#e2e8f0',
                    padding: '1.5rem',
                    borderRadius: '12px',
                    overflow: 'auto',
                    fontSize: '0.9rem',
                    lineHeight: '1.6',
                    maxHeight: '400px'
                }
            }, AUE3Utils.createElement('code', {}, code))
        ]);
    },

    /**
     * 创建按钮
     */
    createButton(text, icon, onClick) {
        return AUE3Utils.createElement('button', {
            style: {
                padding: '1rem 2rem',
                background: 'linear-gradient(135deg, #ec4899, #8b5cf6)',
                color: 'white',
                border: 'none',
                borderRadius: '12px',
                fontWeight: '600',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                transition: 'all 0.2s'
            },
            onClick: onClick
        }, [
            AUE3Utils.createElement('i', {
                className: `fas ${icon}`
            }),
            text
        ]);
    },

    /**
     * 创建图标按钮
     */
    createIconButton(icon, onClick) {
        return AUE3Utils.createElement('button', {
            style: {
                padding: '0.5rem 1rem',
                background: '#f1f5f9',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                transition: 'all 0.2s'
            },
            onClick: onClick
        }, AUE3Utils.createElement('i', {
            className: `fas ${icon}`,
            style: { color: '#64748b' }
        }));
    },

    /**
     * 获取风险文本
     */
    getRiskText(risk) {
        const map = {
            conservative: '保守型',
            moderate: '稳健型',
            aggressive: '进取型'
        };
        return map[risk] || risk;
    },

    /**
     * 获取频率文本
     */
    getFrequencyText(freq) {
        const map = {
            daily: '日级',
            weekly: '周级',
            monthly: '月度'
        };
        return map[freq] || freq;
    }
};

// ==================== 初始化应用 ====================
const strategyApp = new AUE3Core();

// 注册各层
strategyApp
    .registerEffects(StrategyEffects)
    .registerActions(StrategyActions)
    .registerUI(StrategyUI);

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    // 初始化状态
    strategyApp.init({
        currentStep: 1,
        formData: {},
        generatedStrategy: null,
        generationProgress: 0,
        generationStatus: '',
        loading: false
    });

    // 加载数据
    strategyApp.dispatch('initialize');

    // 绑定表单事件
    const fields = ['targetReturn', 'investmentPeriod', 'initialCapital', 
                   'maxDrawdown', 'strategyType', 'tradingFrequency'];
    
    fields.forEach(field => {
        const element = document.getElementById(field);
        if (element) {
            element.addEventListener('change', (e) => {
                strategyApp.dispatch('updateFormData', {
                    field: field,
                    value: e.target.value
                });
            });
        }
    });

    // 绑定风险选择器
    AUE3Utils.$$('.risk-option').forEach(option => {
        option.addEventListener('click', function() {
            AUE3Utils.$$('.risk-option').forEach(o => o.classList.remove('active'));
            this.classList.add('active');
            strategyApp.dispatch('updateFormData', {
                field: 'riskProfile',
                value: this.dataset.risk
            });
        });
    });

    // 绑定行业标签
    AUE3Utils.$$('.tag-item').forEach(tag => {
        tag.addEventListener('click', function() {
            const industry = this.textContent.trim();
            strategyApp.dispatch('toggleIndustry', { industry });
            this.classList.toggle('active');
        });
    });
});

// 全局函数
window.strategyApp = strategyApp;
window.nextStep = () => {
    strategyApp.dispatch('nextStep');
};
window.prevStep = () => {
    strategyApp.dispatch('prevStep');
};











/**
 * 仪表板模块 - AUE3架构实现
 */

// ==================== Mock Data 模拟数据 ====================
const MockData = {
    // 市场指数
    marketIndices: [
        { name: '上证指数', code: '000001', value: 3245.67, change: 1.2, volume: '2850亿' },
        { name: '深证成指', code: '399001', value: 10856.23, change: 0.8, volume: '3120亿' },
        { name: '创业板指', code: '399006', value: 2156.89, change: -0.3, volume: '1580亿' },
        { name: '沪深300', code: '000300', value: 4123.45, change: 0.9, volume: '1920亿' }
    ],

    // 投资组合统计
    portfolioStats: {
        totalAssets: 1250000,
        todayProfit: 31250,
        sharpeRatio: 1.85,
        maxDrawdown: -2.3,
        todayChange: 2.5,
        profitChange: 1.8,
        sharpeChange: 0.15,
        drawdownChange: -0.5
    },

    // 持仓分布
    positions: [
        { symbol: '600036', name: '招商银行', weight: 15.2, profit: 8.5, value: 190000 },
        { symbol: '000001', name: '平安银行', weight: 12.8, profit: 6.2, value: 160000 },
        { symbol: '601318', name: '中国平安', weight: 11.5, profit: 4.8, value: 143750 },
        { symbol: '000002', name: '万科A', weight: 10.3, profit: -2.1, value: 128750 },
        { symbol: '600519', name: '贵州茅台', weight: 9.8, profit: 12.3, value: 122500 }
    ],

    // 最近交易
    recentTrades: [
        {
            time: '2024-01-15 14:30',
            symbol: '000001',
            name: '平安银行',
            type: 'buy',
            quantity: 1000,
            price: 12.45,
            profit: 1250,
            status: '已成交'
        },
        {
            time: '2024-01-15 10:15',
            symbol: '600036',
            name: '招商银行',
            type: 'sell',
            quantity: 500,
            price: 45.67,
            profit: -230,
            status: '已成交'
        },
        {
            time: '2024-01-14 15:45',
            symbol: '601318',
            name: '中国平安',
            type: 'buy',
            quantity: 200,
            price: 56.78,
            profit: 0,
            status: '已成交'
        },
        {
            time: '2024-01-14 11:20',
            symbol: '000002',
            name: '万科A',
            type: 'buy',
            quantity: 1500,
            price: 18.32,
            profit: 0,
            status: '已成交'
        },
        {
            time: '2024-01-13 14:10',
            symbol: '600519',
            name: '贵州茅台',
            type: 'sell',
            quantity: 10,
            price: 1825.50,
            profit: 3500,
            status: '已成交'
        }
    ],

    // 资产净值曲线数据（3个月）
    equityCurve: (() => {
        const dates = [];
        const values = [];
        let baseValue = 1000000;
        
        for (let i = 90; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            dates.push(AUE3Utils.formatDate(date, 'MM-DD'));
            
            // 随机波动，总体向上
            baseValue *= (1 + (Math.random() * 0.03 - 0.01));
            values.push(Math.round(baseValue));
        }
        
        return { dates, values };
    })(),

    // 市场新闻
    marketNews: [
        { time: '10:30', title: 'A股三大指数集体高开，科技板块领涨' },
        { time: '09:15', title: '央行今日进行100亿元逆回购操作' },
        { time: '08:45', title: '多家机构上调全年经济增长预期至5.5%' },
        { time: '昨天', title: '证监会：继续推进注册制改革' },
        { time: '昨天', title: '外资持续流入A股市场，本周净买入超200亿' }
    ]
};

// ==================== Effects 副作用层 ====================
const DashboardEffects = {
    /**
     * 加载仪表板数据
     */
    async loadDashboardData(payload, state, app) {
        // 模拟API延迟
        await new Promise(resolve => setTimeout(resolve, 500));
        
        return {
            marketIndices: MockData.marketIndices,
            portfolioStats: MockData.portfolioStats,
            positions: MockData.positions,
            recentTrades: MockData.recentTrades,
            equityCurve: MockData.equityCurve,
            marketNews: MockData.marketNews,
            lastUpdate: new Date().toISOString()
        };
    },

    /**
     * 刷新市场数据
     */
    async refreshMarketData(payload, state, app) {
        await new Promise(resolve => setTimeout(resolve, 300));
        
        // 随机更新指数
        return {
            marketIndices: MockData.marketIndices.map(index => ({
                ...index,
                value: index.value * (1 + (Math.random() * 0.02 - 0.01)),
                change: (Math.random() * 3 - 1).toFixed(2)
            }))
        };
    },

    /**
     * 加载更多交易记录
     */
    async loadMoreTrades(payload, state, app) {
        await new Promise(resolve => setTimeout(resolve, 500));
        
        // 返回更多模拟数据
        return MockData.recentTrades;
    }
};

// ==================== Actions 动作层 ====================
const DashboardActions = {
    /**
     * 初始化仪表板
     */
    async initializeDashboard(payload, state, app) {
        try {
            const data = await app.runEffect('loadDashboardData');
            
            return {
                state: {
                    ...data,
                    loading: false,
                    error: null
                },
                render: 'dashboard'
            };
        } catch (error) {
            console.error('Initialize dashboard failed:', error);
            return {
                state: {
                    loading: false,
                    error: error.message
                },
                render: 'error'
            };
        }
    },

    /**
     * 刷新数据
     */
    async refreshData(payload, state, app) {
        try {
            const data = await app.runEffect('refreshMarketData');
            
            Components.toast('数据已刷新', 'success');
            
            return {
                state: {
                    ...data,
                    lastUpdate: new Date().toISOString()
                },
                render: 'marketBar'
            };
        } catch (error) {
            Components.toast('刷新失败', 'error');
            return { state: { error: error.message } };
        }
    },

    /**
     * 导航到其他页面
     */
    navigateToPage(payload, state, app) {
        const { page } = payload;
        const routes = {
            'chat-mode': '/web/pages/chat-mode.html',
            'strategy-mode': '/web/pages/strategy-mode.html',
            'home': '/index.html'
        };
        
        if (routes[page]) {
            window.location.href = routes[page];
        }
    },

    /**
     * 显示功能开发中提示
     */
    showComingSoon(payload, state, app) {
        const { feature } = payload;
        Components.modal(
            AUE3Utils.createElement('div', {
                style: {
                    textAlign: 'center',
                    padding: '2rem'
                }
            }, [
                AUE3Utils.createElement('div', {
                    style: {
                        fontSize: '4rem',
                        marginBottom: '1.5rem'
                    }
                }, '🚀'),
                AUE3Utils.createElement('h3', {
                    style: {
                        fontSize: '1.5rem',
                        fontWeight: '700',
                        marginBottom: '1rem',
                        color: '#0f172a'
                    }
                }, feature),
                AUE3Utils.createElement('p', {
                    style: {
                        fontSize: '1rem',
                        color: '#64748b'
                    }
                }, '该功能正在开发中，敬请期待！')
            ]),
            {
                title: '',
                width: '500px',
                closable: true
            }
        );
    }
};

// ==================== UI 界面层 ====================
const DashboardUI = {
    /**
     * 渲染整个仪表板
     */
    dashboard(state, app) {
        // 如果正在加载，显示加载状态
        if (state.loading) {
            const container = AUE3Utils.$('#dashboardContent');
            if (container) {
                container.innerHTML = '';
                container.appendChild(Components.loading('加载仪表板数据...'));
            }
            return;
        }

        // 渲染各个部分
        this.marketBar(state, app);
        this.statsCards(state, app);
        this.positionsChart(state, app);
        this.tradesTable(state, app);
    },

    /**
     * 渲染市场状态栏
     */
    marketBar(state, app) {
        const marketBar = AUE3Utils.$('.market-indicators');
        if (!marketBar || !state.marketIndices) return;

        marketBar.innerHTML = '';
        
        state.marketIndices.forEach(index => {
            const changeClass = index.change >= 0 ? 'up' : 'down';
            const changeIcon = index.change >= 0 ? '↑' : '↓';
            
            const item = AUE3Utils.createElement('div', {
                className: 'market-item'
            }, [
                AUE3Utils.createElement('div', {
                    className: 'market-label'
                }, index.name),
                AUE3Utils.createElement('div', {
                    className: 'market-value'
                }, index.value.toFixed(2)),
                AUE3Utils.createElement('div', {
                    className: `market-change ${changeClass}`
                }, `${changeIcon} ${Math.abs(index.change).toFixed(2)}%`)
            ]);
            
            marketBar.appendChild(item);
        });
    },

    /**
     * 渲染统计卡片
     */
    statsCards(state, app) {
        if (!state.portfolioStats) return;

        const stats = state.portfolioStats;
        const cardsData = [
            {
                label: '总资产',
                value: `¥${stats.totalAssets.toLocaleString()}`,
                trend: stats.todayChange,
                icon: 'fa-wallet',
                variant: 'primary'
            },
            {
                label: '今日收益',
                value: `¥${stats.todayProfit.toLocaleString()}`,
                trend: stats.profitChange,
                icon: 'fa-chart-line',
                variant: 'success'
            },
            {
                label: '夏普比率',
                value: stats.sharpeRatio.toFixed(2),
                trend: stats.sharpeChange,
                icon: 'fa-shield-alt',
                variant: 'primary'
            },
            {
                label: '最大回撤',
                value: `${stats.maxDrawdown}%`,
                trend: stats.drawdownChange,
                icon: 'fa-exclamation-triangle',
                variant: 'warning'
            }
        ];

        // 更新每个卡片
        cardsData.forEach((data, index) => {
            const card = AUE3Utils.$(`.stat-card:nth-child(${index + 1})`);
            if (!card) return;

            const valueEl = card.querySelector('.stat-value');
            const trendEl = card.querySelector('.stat-trend span');
            
            if (valueEl) valueEl.textContent = data.value;
            if (trendEl) {
                const trendText = `${data.trend >= 0 ? '+' : ''}${data.trend}% ${data.label === '今日收益' ? '今日' : '本周'}`;
                trendEl.textContent = trendText;
            }
        });
    },

    /**
     * 渲染持仓图表
     */
    positionsChart(state, app) {
        // 这里可以使用Chart.js等图表库绘制
        // 为了简化，只更新占位文本
        const chartContainers = AUE3Utils.$$('.chart-container');
        chartContainers.forEach((container, index) => {
            if (index === 0 && state.positions) {
                container.innerHTML = `<div style="padding: 2rem; text-align: center; color: #64748b;">
                    <i class="fas fa-chart-area" style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.5;"></i>
                    <p>资产净值曲线数据已加载</p>
                    <p style="font-size: 0.9rem; margin-top: 0.5rem;">最新净值: ¥${state.equityCurve.values[state.equityCurve.values.length - 1].toLocaleString()}</p>
                </div>`;
            } else if (index === 1 && state.positions) {
                const html = `
                    <div style="padding: 2rem;">
                        <div style="margin-bottom: 1rem; font-weight: 600; color: #0f172a;">持仓分布 TOP5</div>
                        ${state.positions.map(pos => `
                            <div style="display: flex; justify-content: space-between; padding: 0.75rem 0; border-bottom: 1px solid #f1f5f9;">
                                <div>
                                    <div style="font-weight: 600; color: #0f172a;">${pos.name}</div>
                                    <div style="font-size: 0.85rem; color: #94a3b8;">${pos.symbol}</div>
                                </div>
                                <div style="text-align: right;">
                                    <div style="font-weight: 700; color: ${pos.profit >= 0 ? '#10b981' : '#ef4444'};">
                                        ${pos.profit >= 0 ? '+' : ''}${pos.profit}%
                                    </div>
                                    <div style="font-size: 0.85rem; color: #64748b;">
                                        占比 ${pos.weight}%
                                    </div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                `;
                container.innerHTML = html;
            }
        });
    },

    /**
     * 渲染交易记录表格
     */
    tradesTable(state, app) {
        const tbody = AUE3Utils.$('.table-card tbody');
        if (!tbody || !state.recentTrades) return;

        tbody.innerHTML = '';
        
        state.recentTrades.forEach(trade => {
            const tr = AUE3Utils.createElement('tr', {}, [
                AUE3Utils.createElement('td', {}, trade.time),
                AUE3Utils.createElement('td', {}, `${trade.symbol} ${trade.name}`),
                AUE3Utils.createElement('td', {}, 
                    `<span class="badge ${trade.type === 'buy' ? 'success' : 'danger'}">
                        ${trade.type === 'buy' ? '买入' : '卖出'}
                    </span>`
                ),
                AUE3Utils.createElement('td', {}, trade.quantity.toLocaleString()),
                AUE3Utils.createElement('td', {}, `¥${trade.price.toFixed(2)}`),
                AUE3Utils.createElement('td', {
                    style: {
                        color: trade.profit > 0 ? '#10b981' : trade.profit < 0 ? '#ef4444' : '#64748b',
                        fontWeight: '600'
                    }
                }, trade.profit ? `${trade.profit > 0 ? '+' : ''}¥${trade.profit.toLocaleString()}` : '-'),
                AUE3Utils.createElement('td', {}, trade.status)
            ]);
            tbody.appendChild(tr);
        });
    },

    /**
     * 渲染错误状态
     */
    error(state, app) {
        const container = AUE3Utils.$('#dashboardContent');
        if (!container) return;

        container.innerHTML = '';
        container.appendChild(Components.empty('加载失败，请刷新重试', 'exclamation-circle'));
    }
};

// ==================== 初始化应用 ====================
const dashboardApp = new AUE3Core();

// 注册各层
dashboardApp
    .registerEffects(DashboardEffects)
    .registerActions(DashboardActions)
    .registerUI(DashboardUI);

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    // 初始化状态
    dashboardApp.init({
        loading: true,
        error: null,
        marketIndices: [],
        portfolioStats: null,
        positions: [],
        recentTrades: [],
        equityCurve: { dates: [], values: [] },
        marketNews: []
    });

    // 加载仪表板数据
    dashboardApp.dispatch('initializeDashboard');

    // 设置自动刷新（每30秒）
    setInterval(() => {
        dashboardApp.dispatch('refreshData');
    }, 30000);
});

// 全局函数（用于HTML onclick等）
window.dashboardApp = dashboardApp;
window.showComingSoon = (featureName) => {
    dashboardApp.dispatch('showComingSoon', { feature: featureName });
};
window.refreshDashboard = () => {
    dashboardApp.dispatch('refreshData');
};
window.navigateTo = (page) => {
    dashboardApp.dispatch('navigateToPage', { page });
};













// FinLoom Web应用主逻辑

class FinLoomApp {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000';
        this.currentSection = 'dashboard';
        this.charts = {};
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadDashboard();
        this.setupCharts();
        this.startRealTimeUpdates();
    }

    setupEventListeners() {
        // 导航菜单点击事件
        document.querySelectorAll('[data-section]').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const section = e.target.closest('[data-section]').dataset.section;
                this.showSection(section);
            });
        });

        // 智能分析表单提交
        document.getElementById('analysis-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.analyzeInvestment();
        });

        // 回测表单提交
        document.getElementById('backtest-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.runBacktest();
        });

        // 数据收集表单提交
        document.getElementById('data-collection-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.collectData();
        });

        // 设置默认结束日期为今天
        document.getElementById('end-date').value = new Date().toISOString().split('T')[0];
    }

    showSection(sectionName) {
        // 隐藏所有内容区域
        document.querySelectorAll('.content-section').forEach(section => {
            section.classList.remove('active');
        });

        // 显示选中的内容区域
        document.getElementById(sectionName).classList.add('active');

        // 更新导航状态
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        document.querySelector(`[data-section="${sectionName}"]`).classList.add('active');

        this.currentSection = sectionName;

        // 根据页面加载相应数据
        switch (sectionName) {
            case 'dashboard':
                this.loadDashboard();
                break;
            case 'portfolio':
                this.loadPortfolio();
                break;
            case 'data':
                this.loadDataOverview();
                break;
        }
    }

    async loadDashboard() {
        try {
            // 加载仪表板数据
            await this.updateDashboardMetrics();
            await this.loadRecentTrades();
            this.updateEquityCurve();
            this.updatePortfolioPie();
        } catch (error) {
            console.error('加载仪表板失败:', error);
        }
    }

    async updateDashboardMetrics() {
        // 模拟数据更新
        const metrics = {
            totalAssets: 1000000 + Math.random() * 50000,
            dailyReturn: 10000 + Math.random() * 5000,
            sharpeRatio: 1.5 + Math.random() * 0.5,
            maxDrawdown: -2 - Math.random() * 2
        };

        document.getElementById('total-assets').textContent = `¥${metrics.totalAssets.toLocaleString()}`;
        document.getElementById('daily-return').textContent = `¥${metrics.dailyReturn.toLocaleString()}`;
        document.getElementById('sharpe-ratio').textContent = metrics.sharpeRatio.toFixed(2);
        document.getElementById('max-drawdown').textContent = `${metrics.maxDrawdown.toFixed(1)}%`;
    }

    async loadRecentTrades() {
        const trades = [
            {
                time: '2024-01-15 14:30',
                symbol: '000001',
                action: '买入',
                quantity: 1000,
                price: 12.45,
                pnl: 1250,
                status: '已成交'
            },
            {
                time: '2024-01-15 10:15',
                symbol: '600036',
                action: '卖出',
                quantity: 500,
                price: 45.67,
                pnl: -230,
                status: '已成交'
            },
            {
                time: '2024-01-14 15:45',
                symbol: '601318',
                action: '买入',
                quantity: 200,
                price: 56.78,
                pnl: 0,
                status: '已成交'
            }
        ];

        const tbody = document.getElementById('recent-trades');
        tbody.innerHTML = trades.map(trade => `
            <tr>
                <td>${trade.time}</td>
                <td>${trade.symbol}</td>
                <td><span class="badge ${trade.action === '买入' ? 'bg-success' : 'bg-danger'}">${trade.action}</span></td>
                <td>${trade.quantity}</td>
                <td>¥${trade.price}</td>
                <td class="${trade.pnl >= 0 ? 'text-success' : 'text-danger'}">${trade.pnl >= 0 ? '+' : ''}¥${trade.pnl}</td>
                <td><span class="badge bg-success">${trade.status}</span></td>
            </tr>
        `).join('');
    }

    updateEquityCurve() {
        const ctx = document.getElementById('equityCurveChart').getContext('2d');
        
        if (this.charts.equityCurve) {
            this.charts.equityCurve.destroy();
        }

        // 生成模拟净值数据
        const dates = [];
        const values = [];
        let currentValue = 1000000;
        const startDate = new Date();
        startDate.setDate(startDate.getDate() - 30);

        for (let i = 0; i < 30; i++) {
            const date = new Date(startDate);
            date.setDate(date.getDate() + i);
            dates.push(date.toISOString().split('T')[0]);
            
            // 模拟价格波动
            const change = (Math.random() - 0.5) * 0.05;
            currentValue *= (1 + change);
            values.push(currentValue);
        }

        this.charts.equityCurve = new Chart(ctx, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [{
                    label: '资产净值',
                    data: values,
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        ticks: {
                            callback: function(value) {
                                return '¥' + value.toLocaleString();
                            }
                        }
                    }
                }
            }
        });
    }

    updatePortfolioPie() {
        const ctx = document.getElementById('portfolioPieChart').getContext('2d');
        
        if (this.charts.portfolioPie) {
            this.charts.portfolioPie.destroy();
        }

        this.charts.portfolioPie = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['银行股', '科技股', '消费股', '医药股', '其他'],
                datasets: [{
                    data: [35, 25, 20, 15, 5],
                    backgroundColor: [
                        '#3498db',
                        '#e74c3c',
                        '#f39c12',
                        '#27ae60',
                        '#9b59b6'
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    async analyzeInvestment() {
        const text = document.getElementById('investment-text').value;
        const amount = document.getElementById('investment-amount').value;
        const riskTolerance = document.getElementById('risk-tolerance').value;

        if (!text.trim()) {
            alert('请输入投资需求描述');
            return;
        }

        this.showLoading();

        try {
            const response = await fetch(`${this.apiBaseUrl}/api/v1/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text: text,
                    amount: amount,
                    risk_tolerance: riskTolerance
                })
            });

            const result = await response.json();
            this.displayAnalysisResult(result);
        } catch (error) {
            console.error('分析失败:', error);
            alert('分析失败，请稍后重试');
        } finally {
            this.hideLoading();
        }
    }

    displayAnalysisResult(result) {
        const resultDiv = document.getElementById('analysis-result');
        const contentDiv = document.getElementById('analysis-content');

        if (result.error) {
            contentDiv.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    分析失败: ${result.error}
                </div>
            `;
        } else {
            const parsed = result.parsed_requirement;
            const strategy = result.strategy_params;
            const risk = result.risk_params;

            contentDiv.innerHTML = `
                <div class="row">
                    <div class="col-md-6">
                        <h6>需求解析结果</h6>
                        <ul class="list-unstyled">
                            <li><strong>投资期限:</strong> ${parsed.investment_horizon || '未指定'}</li>
                            <li><strong>风险偏好:</strong> ${parsed.risk_tolerance || '未指定'}</li>
                            <li><strong>投资目标:</strong> ${parsed.investment_goals.map(g => g.goal_type).join(', ')}</li>
                            <li><strong>投资金额:</strong> ${parsed.investment_amount ? '¥' + parsed.investment_amount.toLocaleString() : '未指定'}</li>
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <h6>策略参数</h6>
                        <ul class="list-unstyled">
                            <li><strong>调仓频率:</strong> ${strategy.rebalance_frequency}</li>
                            <li><strong>仓位管理:</strong> ${strategy.position_sizing_method}</li>
                            <li><strong>策略组合:</strong></li>
                            <li class="ms-3">
                                <small>
                                    趋势跟踪: ${(strategy.strategy_mix.trend_following * 100).toFixed(0)}%<br>
                                    均值回归: ${(strategy.strategy_mix.mean_reversion * 100).toFixed(0)}%<br>
                                    动量策略: ${(strategy.strategy_mix.momentum * 100).toFixed(0)}%<br>
                                    价值投资: ${(strategy.strategy_mix.value * 100).toFixed(0)}%
                                </small>
                            </li>
                        </ul>
                    </div>
                </div>
                <div class="row mt-3">
                    <div class="col-12">
                        <h6>风险参数</h6>
                        <div class="row">
                            <div class="col-md-3">
                                <div class="text-center">
                                    <h5 class="text-danger">${(risk.max_drawdown * 100).toFixed(1)}%</h5>
                                    <small class="text-muted">最大回撤</small>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="text-center">
                                    <h5 class="text-warning">${(risk.position_limit * 100).toFixed(1)}%</h5>
                                    <small class="text-muted">仓位限制</small>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="text-center">
                                    <h5 class="text-info">${risk.leverage}x</h5>
                                    <small class="text-muted">杠杆倍数</small>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="text-center">
                                    <h5 class="text-secondary">${(risk.stop_loss * 100).toFixed(1)}%</h5>
                                    <small class="text-muted">止损比例</small>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        resultDiv.style.display = 'block';
    }

    async loadPortfolio() {
        // 模拟持仓数据
        const positions = [
            {
                symbol: '000001',
                name: '平安银行',
                quantity: 1000,
                costPrice: 12.00,
                currentPrice: 12.45,
                marketValue: 12450,
                pnl: 450,
                pnlRate: 3.75
            },
            {
                symbol: '600036',
                name: '招商银行',
                quantity: 500,
                costPrice: 45.00,
                currentPrice: 45.67,
                marketValue: 22835,
                pnl: 335,
                pnlRate: 1.49
            },
            {
                symbol: '601318',
                name: '中国平安',
                quantity: 200,
                costPrice: 55.00,
                currentPrice: 56.78,
                marketValue: 11356,
                pnl: 356,
                pnlRate: 3.24
            }
        ];

        const tbody = document.getElementById('portfolio-positions');
        tbody.innerHTML = positions.map(pos => `
            <tr>
                <td>
                    <strong>${pos.symbol}</strong><br>
                    <small class="text-muted">${pos.name}</small>
                </td>
                <td>${pos.quantity}</td>
                <td>¥${pos.costPrice}</td>
                <td>¥${pos.currentPrice}</td>
                <td>¥${pos.marketValue.toLocaleString()}</td>
                <td class="${pos.pnl >= 0 ? 'text-success' : 'text-danger'}">
                    ${pos.pnl >= 0 ? '+' : ''}¥${pos.pnl}
                </td>
                <td class="${pos.pnlRate >= 0 ? 'text-success' : 'text-danger'}">
                    ${pos.pnlRate >= 0 ? '+' : ''}${pos.pnlRate.toFixed(2)}%
                </td>
                <td>
                    <button class="btn btn-sm btn-outline-primary me-1">调整</button>
                    <button class="btn btn-sm btn-outline-danger">清仓</button>
                </td>
            </tr>
        `).join('');

        // 更新组合统计
        const totalMarketValue = positions.reduce((sum, pos) => sum + pos.marketValue, 0);
        const totalCost = positions.reduce((sum, pos) => sum + (pos.quantity * pos.costPrice), 0);
        const totalPnl = totalMarketValue - totalCost;

        document.getElementById('total-market-value').textContent = `¥${totalMarketValue.toLocaleString()}`;
        document.getElementById('total-cost').textContent = `¥${totalCost.toLocaleString()}`;
        document.getElementById('floating-pnl').textContent = `+¥${totalPnl.toLocaleString()}`;
        document.getElementById('position-count').textContent = positions.length;
    }

    async runBacktest() {
        const strategy = document.getElementById('strategy-select').value;
        const symbol = document.getElementById('symbol-select').value;
        const startDate = document.getElementById('start-date').value;
        const endDate = document.getElementById('end-date').value;
        const initialCapital = document.getElementById('initial-capital').value;

        this.showLoading();

        try {
            // 模拟回测结果
            await new Promise(resolve => setTimeout(resolve, 2000));

            const results = {
                totalReturn: 25.6,
                annualReturn: 12.8,
                sharpeRatio: 1.85,
                maxDrawdown: -8.2
            };

            this.displayBacktestResults(results);
        } catch (error) {
            console.error('回测失败:', error);
            alert('回测失败，请稍后重试');
        } finally {
            this.hideLoading();
        }
    }

    displayBacktestResults(results) {
        document.getElementById('total-return').textContent = `+${results.totalReturn}%`;
        document.getElementById('annual-return').textContent = `+${results.annualReturn}%`;
        document.getElementById('sharpe-ratio-backtest').textContent = results.sharpeRatio.toFixed(2);
        document.getElementById('max-drawdown-backtest').textContent = `${results.maxDrawdown}%`;

        // 显示回测结果
        document.getElementById('backtest-results').style.display = 'block';
        document.getElementById('backtest-placeholder').style.display = 'none';

        // 更新回测图表
        this.updateBacktestChart();
    }

    updateBacktestChart() {
        const ctx = document.getElementById('backtestChart').getContext('2d');
        
        if (this.charts.backtest) {
            this.charts.backtest.destroy();
        }

        // 生成模拟回测数据
        const dates = [];
        const strategyValues = [];
        const benchmarkValues = [];
        let strategyValue = 1000000;
        let benchmarkValue = 1000000;
        const startDate = new Date('2023-01-01');

        for (let i = 0; i < 252; i++) {
            const date = new Date(startDate);
            date.setDate(date.getDate() + i);
            dates.push(date.toISOString().split('T')[0]);
            
            // 模拟策略收益
            const strategyChange = (Math.random() - 0.45) * 0.02;
            strategyValue *= (1 + strategyChange);
            strategyValues.push(strategyValue);
            
            // 模拟基准收益
            const benchmarkChange = (Math.random() - 0.5) * 0.015;
            benchmarkValue *= (1 + benchmarkChange);
            benchmarkValues.push(benchmarkValue);
        }

        this.charts.backtest = new Chart(ctx, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [{
                    label: '策略净值',
                    data: strategyValues,
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    borderWidth: 2,
                    fill: false
                }, {
                    label: '基准净值',
                    data: benchmarkValues,
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    borderWidth: 2,
                    fill: false,
                    borderDash: [5, 5]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        ticks: {
                            callback: function(value) {
                                return '¥' + value.toLocaleString();
                            }
                        }
                    }
                }
            }
        });
    }

    async collectData() {
        const symbol = document.getElementById('data-symbol').value;
        const period = document.getElementById('data-period').value;

        if (!symbol.trim()) {
            alert('请输入股票代码');
            return;
        }

        this.showLoading();

        try {
            // 模拟数据收集
            await new Promise(resolve => setTimeout(resolve, 1500));
            
            alert(`成功收集 ${symbol} 的 ${period} 数据`);
            this.loadDataOverview();
        } catch (error) {
            console.error('数据收集失败:', error);
            alert('数据收集失败，请稍后重试');
        } finally {
            this.hideLoading();
        }
    }

    async loadDataOverview() {
        // 模拟数据概览
        const dataOverview = [
            {
                symbol: '000001',
                name: '平安银行',
                count: 252,
                price: 12.45,
                updateTime: '2024-01-15 15:00'
            },
            {
                symbol: '600036',
                name: '招商银行',
                count: 252,
                price: 45.67,
                updateTime: '2024-01-15 15:00'
            },
            {
                symbol: '601318',
                name: '中国平安',
                count: 252,
                price: 56.78,
                updateTime: '2024-01-15 15:00'
            }
        ];

        const tbody = document.getElementById('data-overview');
        tbody.innerHTML = dataOverview.map(item => `
            <tr>
                <td>${item.symbol}</td>
                <td>${item.name}</td>
                <td>${item.count}</td>
                <td>¥${item.price}</td>
                <td>${item.updateTime}</td>
                <td>
                    <button class="btn btn-sm btn-outline-primary me-1">查看</button>
                    <button class="btn btn-sm btn-outline-success">更新</button>
                </td>
            </tr>
        `).join('');
    }

    setupCharts() {
        // 初始化图表
        this.updateEquityCurve();
        this.updatePortfolioPie();
    }

    startRealTimeUpdates() {
        // 每30秒更新一次数据
        setInterval(() => {
            if (this.currentSection === 'dashboard') {
                this.updateDashboardMetrics();
            }
        }, 30000);
    }

    showLoading() {
        document.getElementById('loading-overlay').style.display = 'flex';
    }

    hideLoading() {
        document.getElementById('loading-overlay').style.display = 'none';
    }
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    window.finLoomApp = new FinLoomApp();
});

// FinLoom Web应用主逻辑

class FinLoomApp {
    constructor() {
        // 自动检测当前端口
        this.apiBaseUrl = window.location.origin;
        this.currentSection = 'dashboard';
        this.version = 'v2.5'; // 添加版本号强制刷新缓存
        this.charts = {};
        this.init();
    }

    init() {
        console.log('FinLoom App initialized with version:', this.version);
        console.log('User Agent:', navigator.userAgent);
        console.log('API Base URL:', this.apiBaseUrl);
        console.log('Current URL:', window.location.href);
        
        // 检测浏览器类型
        const isChrome = /Chrome/.test(navigator.userAgent) && /Google Inc/.test(navigator.vendor);
        const isSafari = /Safari/.test(navigator.userAgent) && /Apple Computer/.test(navigator.vendor);
        console.log('Browser detected - Chrome:', isChrome, 'Safari:', isSafari);
        
        // 统一初始化流程，修复Chrome兼容性问题
        this.initializeApp();
    }
    
    // 移除单独的Chrome修复函数，将修复整合到统一的初始化流程中
    
    initializeApp() {
        // 修复Chrome兼容性问题 - 使用更可靠的DOM加载检测
        const checkAndInit = () => {
            // 确保必要的DOM元素存在
            if (document.querySelector('.app-container')) {
                this.setupEventListeners();
                this.loadDashboard();
                this.setupCharts();
                this.startRealTimeUpdates();
                this.addAdvancedInteractions();
            } else {
                // 如果DOM元素还未准备好，稍后再试
                setTimeout(checkAndInit, 100);
            }
        };
        
        // 使用多种方式确保DOM加载完成
        if (document.readyState === 'complete' || document.readyState === 'interactive') {
            // 页面已经加载或正在交互
            setTimeout(checkAndInit, 50); // 稍微延迟确保所有元素都已渲染
        } else {
            // 页面仍在加载中
            document.addEventListener('DOMContentLoaded', checkAndInit);
        }
    }

    // 通用API调用函数
    async apiCall(endpoint, options = {}) {
        try {
            const response = await fetch(`${this.apiBaseUrl}${endpoint}`, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.status === 'error') {
                throw new Error(result.error);
            }
            
            return result;
        } catch (error) {
            console.error(`API call failed for ${endpoint}:`, error);
            throw error;
        }
    }

    setupEventListeners() {
        // 修复Chrome兼容性问题 - 延迟绑定事件监听器确保DOM完全准备好
        setTimeout(() => {
            // 导航菜单点击事件 - 修复事件绑定问题
            document.querySelectorAll('[data-section]').forEach(link => {
                // 移除可能已存在的事件监听器
                const handleClick = (e) => {
                    e.preventDefault();
                    const section = e.target.closest('[data-section]')?.dataset.section;
                    if (section) {
                        this.showSection(section);
                    }
                };
                link.removeEventListener('click', handleClick);
                link.addEventListener('click', handleClick);
            });

            // 智能分析表单提交 - 统一修复Chrome兼容性问题
            const analysisForm = document.getElementById('analysisForm');
            if (analysisForm) {
                console.log('Found analysisForm, adding event listener');
                
                // 创建统一的表单处理函数
                const handleAnalysisSubmit = (e) => {
                    console.log('Analysis form submitted or button clicked');
                    e.preventDefault();
                    e.stopPropagation();
                    this.analyzeInvestment();
                };
                
                // 移除旧的事件监听器（如果有的话）
                analysisForm.removeEventListener('submit', handleAnalysisSubmit);
                // 绑定新的事件监听器
                analysisForm.addEventListener('submit', handleAnalysisSubmit);
                
                // 同时为按钮添加点击事件（Chrome兼容性修复）
                const submitBtn = analysisForm.querySelector('button[type="submit"]');
                if (submitBtn) {
                    submitBtn.removeEventListener('click', handleAnalysisSubmit);
                    submitBtn.addEventListener('click', handleAnalysisSubmit);
                }
            } else {
                console.log('analysisForm not found');
            }

            // 回测表单提交 - 统一修复Chrome兼容性问题
            const backtestForm = document.getElementById('backtestForm');
            if (backtestForm) {
                console.log('Found backtestForm, adding event listener');
                
                // 创建统一的表单处理函数
                const handleBacktestSubmit = (e) => {
                    console.log('Backtest form submitted or button clicked');
                    e.preventDefault();
                    e.stopPropagation();
                    this.runBacktest();
                };
                
                // 移除旧的事件监听器（如果有的话）
                backtestForm.removeEventListener('submit', handleBacktestSubmit);
                // 绑定新的事件监听器
                backtestForm.addEventListener('submit', handleBacktestSubmit);
                
                // 同时为按钮添加点击事件（Chrome兼容性修复）
                const submitBtn = backtestForm.querySelector('button[type="submit"]');
                if (submitBtn) {
                    submitBtn.removeEventListener('click', handleBacktestSubmit);
                    submitBtn.addEventListener('click', handleBacktestSubmit);
                }
            } else {
                console.log('backtestForm not found');
            }

            // 数据收集表单提交 - 统一修复Chrome兼容性问题
            const dataCollectionForm = document.getElementById('dataCollectionForm');
            if (dataCollectionForm) {
                console.log('Found dataCollectionForm, adding event listener');
                
                // 创建统一的表单处理函数
                const handleDataCollectionSubmit = (e) => {
                    console.log('Data collection form submitted or button clicked');
                    e.preventDefault();
                    e.stopPropagation();
                    this.collectData();
                };
                
                // 移除旧的事件监听器（如果有的话）
                dataCollectionForm.removeEventListener('submit', handleDataCollectionSubmit);
                // 绑定新的事件监听器
                dataCollectionForm.addEventListener('submit', handleDataCollectionSubmit);
                
                // 同时为按钮添加点击事件（Chrome兼容性修复）
                const submitBtn = dataCollectionForm.querySelector('button[type="submit"]');
                if (submitBtn) {
                    submitBtn.removeEventListener('click', handleDataCollectionSubmit);
                    submitBtn.addEventListener('click', handleDataCollectionSubmit);
                }
            } else {
                console.log('dataCollectionForm not found');
            }

            // 设置默认结束日期为今天
            const endDateElement = document.getElementById('end-date');
            if (endDateElement) {
                endDateElement.value = new Date().toISOString().split('T')[0];
            }

            // 刷新按钮事件 - 修复Chrome兼容性问题
            document.querySelectorAll('.btn-outline-primary .fa-sync-alt').forEach(btn => {
                const button = btn.closest('button');
                if (button) {
                    const handleRefreshClick = (e) => {
                        e.preventDefault();
                        const icon = btn;
                        // 添加旋转动画
                        icon.classList.add('fa-spin');
                        setTimeout(() => {
                            icon.classList.remove('fa-spin');
                        }, 1000);
                        
                        // 根据当前页面刷新数据
                        switch(this.currentSection) {
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
                    };
                    button.removeEventListener('click', handleRefreshClick);
                    button.addEventListener('click', handleRefreshClick);
                }
            });

            // 时间选择器事件 - 修复Chrome兼容性问题
            document.querySelectorAll('.time-selector .btn').forEach(btn => {
                const handleTimeClick = (e) => {
                    e.preventDefault();
                    // 移除所有按钮的active状态
                    document.querySelectorAll('.time-selector .btn').forEach(b => {
                        b.classList.remove('active');
                    });
                    // 添加当前按钮的active状态
                    btn.classList.add('active');
                    // 这里可以添加更新图表的逻辑
                    this.updateEquityCurve();
                };
                btn.removeEventListener('click', handleTimeClick);
                btn.addEventListener('click', handleTimeClick);
            });
            
            console.log('Event listeners setup completed');
        }, 100); // 延迟100ms确保DOM完全准备好
    }

    addAdvancedInteractions() {
        // 为所有卡片添加悬停效果
        document.querySelectorAll('.card').forEach(card => {
            card.addEventListener('mouseenter', () => {
                card.style.transform = 'translateY(-5px)';
            });
            card.addEventListener('mouseleave', () => {
                card.style.transform = 'translateY(0)';
            });
        });

        // 为表格行添加悬停效果
        document.querySelectorAll('tbody tr').forEach(row => {
            row.addEventListener('mouseenter', () => {
                row.style.backgroundColor = 'rgba(52, 152, 219, 0.05)';
            });
            row.addEventListener('mouseleave', () => {
                row.style.backgroundColor = '';
            });
        });

        // 为金融数据项添加悬停效果
        document.querySelectorAll('.market-item, .stock-item').forEach(item => {
            item.addEventListener('mouseenter', () => {
                item.style.backgroundColor = 'rgba(52, 152, 219, 0.03)';
                item.style.borderRadius = '8px';
                item.style.paddingLeft = '10px';
            });
            item.addEventListener('mouseleave', () => {
                item.style.backgroundColor = '';
                item.style.borderRadius = '';
                item.style.paddingLeft = '';
            });
        });

        // 添加滚动动画
        this.setupScrollAnimations();
        
        // 添加侧边栏改进
        this.setupSidebarEnhancements();
    }

    setupScrollAnimations() {
        // 创建Intersection Observer来触发滚动动画
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate');
                }
            });
        }, observerOptions);

        // 观察所有需要动画的元素
        document.querySelectorAll('.scroll-animate').forEach(el => {
            observer.observe(el);
        });
    }

    setupSidebarEnhancements() {
        const sidebar = document.querySelector('.sidebar');
        const sidebarToggle = document.getElementById('sidebarToggle');
        
        if (sidebar && sidebarToggle) {
            // 添加侧边栏切换功能
            sidebarToggle.addEventListener('click', () => {
                sidebar.classList.toggle('collapsed');
                
                // 保存状态到localStorage
                const isCollapsed = sidebar.classList.contains('collapsed');
                localStorage.setItem('sidebarCollapsed', isCollapsed);
            });

            // 恢复侧边栏状态
            const isCollapsed = localStorage.getItem('sidebarCollapsed') === 'true';
            if (isCollapsed) {
                sidebar.classList.add('collapsed');
            }

            // 添加菜单项点击效果
            document.querySelectorAll('.menu-item').forEach(item => {
                item.addEventListener('click', (e) => {
                    // 移除所有active状态
                    document.querySelectorAll('.menu-item').forEach(menuItem => {
                        menuItem.classList.remove('active');
                    });
                    
                    // 添加当前active状态
                    item.classList.add('active');
                    
                    // 添加点击波纹效果
                    this.createRippleEffect(e, item);
                });
            });
        }
    }

    createRippleEffect(event, element) {
        const ripple = document.createElement('span');
        const rect = element.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        const x = event.clientX - rect.left - size / 2;
        const y = event.clientY - rect.top - size / 2;
        
        ripple.style.width = ripple.style.height = size + 'px';
        ripple.style.left = x + 'px';
        ripple.style.top = y + 'px';
        ripple.classList.add('ripple');
        
        element.appendChild(ripple);
        
        setTimeout(() => {
            ripple.remove();
        }, 600);
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
            // 显示加载状态
            this.showLoading();
            
            // 加载仪表板数据
            await this.updateDashboardMetrics();
            await this.loadRecentTrades();
            this.updateEquityCurve();
            this.updatePortfolioPie();
            
            // 隐藏加载状态
            this.hideLoading();
        } catch (error) {
            console.error('加载仪表板失败:', error);
            this.hideLoading();
        }
    }

    async updateDashboardMetrics() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/v1/dashboard/metrics`);
            const result = await response.json();
            
            if (result.status === 'error') {
                throw new Error(result.error);
            }
            
            const metrics = result.data;
            document.getElementById('total-assets').textContent = `¥${metrics.total_assets.toLocaleString('zh-CN', {maximumFractionDigits: 0})}`;
            document.getElementById('daily-return').textContent = `¥${metrics.daily_return.toLocaleString('zh-CN', {maximumFractionDigits: 0})}`;
            document.getElementById('sharpe-ratio').textContent = metrics.sharpe_ratio.toFixed(2);
            document.getElementById('max-drawdown').textContent = `${metrics.max_drawdown.toFixed(1)}%`;
            
            // 更新其他指标
            if (document.getElementById('portfolio-value')) {
                document.getElementById('portfolio-value').textContent = `¥${metrics.portfolio_value.toLocaleString('zh-CN', {maximumFractionDigits: 0})}`;
            }
            if (document.getElementById('unrealized-pnl')) {
                document.getElementById('unrealized-pnl').textContent = `¥${metrics.unrealized_pnl.toLocaleString('zh-CN', {maximumFractionDigits: 0})}`;
            }
            if (document.getElementById('win-rate')) {
                document.getElementById('win-rate').textContent = `${(metrics.win_rate * 100).toFixed(1)}%`;
            }
            if (document.getElementById('total-trades')) {
                document.getElementById('total-trades').textContent = metrics.total_trades.toString();
            }
            
        } catch (error) {
            console.error('Failed to update dashboard metrics:', error);
            this.showAlert('获取仪表板数据失败', 'error');
        }
    }

    async loadRecentTrades() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/v1/trades/recent`);
            const result = await response.json();
            
            if (result.status === 'error') {
                throw new Error(result.error);
            }
            
            const trades = result.data.trades;
            
            const tbody = document.getElementById('recent-trades');
            if (tbody) {
                tbody.innerHTML = trades.map(trade => `
                    <tr>
                        <td>${trade.time}</td>
                        <td><strong>${trade.symbol}</strong><br><small>${trade.name}</small></td>
                        <td><span class="badge ${trade.action === 'BUY' ? 'bg-success' : 'bg-danger'}">${trade.action === 'BUY' ? '买入' : '卖出'}</span></td>
                        <td>${trade.quantity.toLocaleString()}</td>
                        <td>¥${trade.price.toFixed(2)}</td>
                        <td class="${trade.pnl >= 0 ? 'text-success' : 'text-danger'}">${trade.pnl >= 0 ? '+' : ''}¥${Math.abs(trade.pnl).toLocaleString()}</td>
                        <td><span class="badge bg-primary">${trade.status}</span></td>
                    </tr>
                `).join('');
            }
            
        } catch (error) {
            console.error('Failed to load recent trades:', error);
            this.showAlert('获取交易记录失败', 'error');
        }
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
        startDate.setDate(startDate.getDate() - 90); // 默认显示3个月数据

        for (let i = 0; i < 90; i++) {
            const date = new Date(startDate);
            date.setDate(date.getDate() + i);
            dates.push(date.toLocaleDateString('zh-CN', {month: 'short', day: 'numeric'}));
            
            // 模拟价格波动
            const change = (Math.random() - 0.5) * 0.03;
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
                    borderWidth: 3,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    fill: true,
                    tension: 0.4,
                    spanGaps: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(44, 62, 80, 0.9)',
                        titleColor: '#fff',
                        bodyColor: '#ecf0f1',
                        borderColor: 'rgba(52, 152, 219, 0.5)',
                        borderWidth: 1,
                        cornerRadius: 8,
                        displayColors: false,
                        callbacks: {
                            label: function(context) {
                                return `净值: ¥${context.parsed.y.toLocaleString('zh-CN', {maximumFractionDigits: 0})}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: '#7f8c8d',
                            maxRotation: 0,
                            autoSkip: true,
                            maxTicksLimit: 8
                        }
                    },
                    y: {
                        beginAtZero: false,
                        grid: {
                            color: 'rgba(127, 140, 141, 0.1)'
                        },
                        ticks: {
                            color: '#7f8c8d',
                            callback: function(value) {
                                return '¥' + (value / 10000).toFixed(0) + '万';
                            }
                        }
                    }
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
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
                    borderWidth: 0,
                    hoverOffset: 10
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            usePointStyle: true,
                            pointStyle: 'circle',
                            font: {
                                size: 12
                            }
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(44, 62, 80, 0.9)',
                        titleColor: '#fff',
                        bodyColor: '#ecf0f1',
                        borderColor: 'rgba(52, 152, 219, 0.5)',
                        borderWidth: 1,
                        cornerRadius: 8,
                        callbacks: {
                            label: function(context) {
                                return `${context.label}: ${context.parsed}%`;
                            }
                        }
                    }
                },
                cutout: '70%'
            }
        });
    }

    async analyzeInvestment() {
        // 修复Chrome兼容性问题 - 确保元素存在
        const textElement = document.getElementById('investment-text');
        const amountElement = document.getElementById('investment-amount');
        const riskToleranceElement = document.getElementById('risk-tolerance');
        
        // 添加延迟确保DOM元素完全加载
        if (!textElement || !amountElement || !riskToleranceElement) {
            console.log('Form elements not ready, waiting...');
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        const text = textElement ? textElement.value : '';
        const amount = amountElement ? amountElement.value : '1000000';
        const riskTolerance = riskToleranceElement ? riskToleranceElement.value : 'moderate';

        if (!text.trim()) {
            this.showAlert('请输入投资需求描述', 'warning');
            return;
        }

        this.showLoading();

        try {
            console.log('Starting comprehensive market analysis...');
            
            // 主要股票列表
            const symbols = ['000001', '000002', '600000', '600036', '601318'];
            
            // 1. 情感分析
            console.log('Step 1: Advanced sentiment analysis...');
            const sentimentApiUrl = `${this.apiBaseUrl}/api/v1/analysis/sentiment/analyze`;
            const sentimentResponse = await fetch(sentimentApiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    texts: [text],
                    text_type: 'investment_requirement',
                    symbols: symbols,
                    use_local_model: true,
                    batch_size: 1
                })
            });
            
            let sentimentResult = null;
            if (sentimentResponse.ok) {
                sentimentResult = await sentimentResponse.json();
                console.log('Advanced sentiment analysis completed:', sentimentResult);
            } else {
                console.warn('Advanced sentiment analysis failed, using fallback');
            }
            
            // 2. 异常检测分析
            console.log('Step 2: Anomaly detection analysis...');
            const anomalyPromises = symbols.slice(0, 3).map(async (symbol) => {
                try {
                    const anomalyResponse = await fetch(`${this.apiBaseUrl}/api/v1/analysis/anomaly/detect`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            symbol: symbol,
                            detection_method: 'all',
                            start_date: new Date(Date.now() - 90*24*60*60*1000).toISOString().split('T')[0]
                        })
                    });
                    
                    if (anomalyResponse.ok) {
                        return await anomalyResponse.json();
                    }
                } catch (e) {
                    console.warn(`Anomaly detection failed for ${symbol}:`, e);
                }
                return null;
            });
            
            const anomalyResults = await Promise.all(anomalyPromises);
            const validAnomalyResults = anomalyResults.filter(r => r !== null);
            console.log('Anomaly detection completed:', validAnomalyResults.length, 'results');
            
            // 3. 相关性分析
            console.log('Step 3: Correlation analysis...');
            let correlationResult = null;
            try {
                const correlationResponse = await fetch(`${this.apiBaseUrl}/api/v1/analysis/correlation/analyze`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        symbols: symbols,
                        method: 'pearson',
                        window_size: 60
                    })
                });
                
                if (correlationResponse.ok) {
                    correlationResult = await correlationResponse.json();
                    console.log('Correlation analysis completed:', correlationResult);
                }
            } catch (e) {
                console.warn('Correlation analysis failed:', e);
            }
            
            // 4. 市场状态检测
            console.log('Step 4: Market regime detection...');
            let regimeResult = null;
            try {
                const regimeResponse = await fetch(`${this.apiBaseUrl}/api/v1/analysis/regime/detect`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        symbols: symbols.slice(0, 2),
                        regime_count: 3,
                        detection_methods: ['hmm', 'clustering']
                    })
                });
                
                if (regimeResponse.ok) {
                    regimeResult = await regimeResponse.json();
                    console.log('Market regime detection completed:', regimeResult);
                }
            } catch (e) {
                console.warn('Market regime detection failed:', e);
            }
            
            // 5. 如果所有高级分析都失败，使用基础分析
            if (!sentimentResult && !correlationResult && !regimeResult && validAnomalyResults.length === 0) {
                console.log('All advanced analysis failed, trying basic sentiment analysis...');
                const basicSentimentResponse = await fetch(`${this.apiBaseUrl}/api/v1/market/sentiment/analyze`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        symbols: symbols,
                        include_news: true,
                        include_social: true,
                        days_back: 7
                    })
                });
                
                if (basicSentimentResponse.ok) {
                    const basicSentimentResult = await basicSentimentResponse.json();
                    this.displaySentimentAnalysisResult(basicSentimentResult);
                    this.showAlert('基础情感分析完成', 'success');
                    return;
                }
            }
            
            // 显示综合分析结果
            this.displayComprehensiveAnalysisResult({
                investment_text: text,
                amount: amount,
                risk_tolerance: riskTolerance,
                sentiment_analysis: sentimentResult,
                anomaly_analysis: validAnomalyResults,
                correlation_analysis: correlationResult,
                regime_analysis: regimeResult,
                symbols: symbols
            });
            
            this.showAlert('综合市场分析完成！', 'success');
            
        } catch (error) {
            console.error('综合分析失败:', error);
            
            // 最后的备用方案：使用主服务器分析API
            try {
                const fallbackApiUrl = `${this.apiBaseUrl}/api/v1/analyze`;
                console.log('Trying fallback analysis API:', fallbackApiUrl);
                
                const fallbackResponse = await fetch(fallbackApiUrl, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        text: text,
                        amount: parseFloat(amount) || 1000000,
                        risk_tolerance: riskTolerance || 'moderate'
                    })
                });
                
                if (!fallbackResponse.ok) {
                    throw new Error(`HTTP error! status: ${fallbackResponse.status}`);
                }
                
                const fallbackResult = await fallbackResponse.json();
                console.log('Fallback analysis API response:', fallbackResult);
                
                // 显示结果
                this.displayAnalysisResult(fallbackResult);
                this.showAlert('分析完成（使用备用方案）', 'success');
            } catch (fallbackError) {
                console.error('备用分析也失败:', fallbackError);
                this.showAlert(`分析失败，请稍后重试: ${error.message}`, 'danger');
            }
        } finally {
            this.hideLoading();
        }
    }

    displayAnalysisResult(result) {
        console.log('Displaying analysis result:', result);
        
        // 修复Chrome兼容性问题 - 确保元素存在并正确显示
        setTimeout(() => {
            const resultDiv = document.getElementById('analysis-result');
            const contentDiv = document.getElementById('analysis-content');
            
            console.log('Found resultDiv:', resultDiv);
            console.log('Found contentDiv:', contentDiv);
            
            // 显示结果区域
            if (resultDiv) {
                resultDiv.style.display = 'block';
                resultDiv.style.opacity = '0';
                console.log('Showing result div');
                
                // 添加淡入动画效果
                setTimeout(() => {
                    resultDiv.style.transition = 'opacity 0.3s ease-in-out';
                    resultDiv.style.opacity = '1';
                }, 50);
            } else {
                console.error('analysis-result div not found!');
                // 尝试查找其他可能的元素
                const alternativeDiv = document.querySelector('.analysis-result') || document.querySelector('#analysisResult');
                if (alternativeDiv) {
                    alternativeDiv.style.display = 'block';
                    alternativeDiv.style.opacity = '0';
                    setTimeout(() => {
                        alternativeDiv.style.transition = 'opacity 0.3s ease-in-out';
                        alternativeDiv.style.opacity = '1';
                    }, 50);
                }
            }

            if (result.error) {
                if (contentDiv) {
                    contentDiv.innerHTML = `
                        <div class="alert alert-danger">
                            <i class="fas fa-exclamation-triangle me-2"></i>
                            分析失败: ${result.error}
                        </div>
                    `;
                }
            } else {
                const parsed = result.parsed_requirement;
                const strategy = result.strategy_params;
                const risk = result.risk_params;

                if (contentDiv) {
                    contentDiv.innerHTML = `
                        <div class="row">
                            <div class="col-md-6">
                                <div class="card mb-4">
                                    <div class="card-header">
                                        <h6 class="mb-0">需求解析结果</h6>
                                    </div>
                                    <div class="card-body">
                                        <ul class="list-unstyled">
                                            <li class="mb-2"><strong>投资期限:</strong> ${parsed.investment_horizon || '未指定'}</li>
                                            <li class="mb-2"><strong>风险偏好:</strong> ${parsed.risk_tolerance || '未指定'}</li>
                                            <li class="mb-2"><strong>投资目标:</strong> ${parsed.investment_goals ? parsed.investment_goals.map(g => g.goal_type).join(', ') : '未指定'}</li>
                                            <li class="mb-2"><strong>投资金额:</strong> ${parsed.investment_amount ? '¥' + parsed.investment_amount.toLocaleString() : '未指定'}</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="card mb-4">
                                    <div class="card-header">
                                        <h6 class="mb-0">策略参数</h6>
                                    </div>
                                    <div class="card-body">
                                        <ul class="list-unstyled">
                                            <li class="mb-2"><strong>调仓频率:</strong> ${strategy.rebalance_frequency || '未指定'}</li>
                                            <li class="mb-2"><strong>仓位管理:</strong> ${strategy.position_sizing_method || '未指定'}</li>
                                            <li class="mb-2"><strong>策略组合:</strong></li>
                                            <li class="ms-3">
                                                <small>
                                                    趋势跟踪: ${strategy.strategy_mix ? (strategy.strategy_mix.trend_following * 100).toFixed(0) : '0'}%<br>
                                                    均值回归: ${strategy.strategy_mix ? (strategy.strategy_mix.mean_reversion * 100).toFixed(0) : '0'}%<br>
                                                    动量策略: ${strategy.strategy_mix ? (strategy.strategy_mix.momentum * 100).toFixed(0) : '0'}%<br>
                                                    价值投资: ${strategy.strategy_mix ? (strategy.strategy_mix.value * 100).toFixed(0) : '0'}%
                                                </small>
                                            </li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="row">
                            <div class="col-12">
                                <div class="card">
                                    <div class="card-header">
                                        <h6 class="mb-0">风险参数</h6>
                                    </div>
                                    <div class="card-body">
                                        <div class="row text-center">
                                            <div class="col-md-3 mb-3 mb-md-0">
                                                <div class="financial-stat">
                                                    <span class="value text-danger">${risk.max_drawdown ? (risk.max_drawdown * 100).toFixed(1) : '0.0'}%</span>
                                                    <span class="label">最大回撤</span>
                                                </div>
                                            </div>
                                            <div class="col-md-3 mb-3 mb-md-0">
                                                <div class="financial-stat">
                                                    <span class="value text-warning">${risk.position_limit ? (risk.position_limit * 100).toFixed(1) : '0.0'}%</span>
                                                    <span class="label">仓位限制</span>
                                                </div>
                                            </div>
                                            <div class="col-md-3 mb-3 mb-md-0">
                                                <div class="financial-stat">
                                                    <span class="value text-info">${risk.leverage || '1'}x</span>
                                                    <span class="label">杠杆倍数</span>
                                                </div>
                                            </div>
                                            <div class="col-md-3">
                                                <div class="financial-stat">
                                                    <span class="value text-secondary">${risk.stop_loss ? (risk.stop_loss * 100).toFixed(1) : '0.0'}%</span>
                                                    <span class="label">止损比例</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                }
            }
            
            // 确保结果显示
            if (resultDiv) {
                resultDiv.style.display = 'block';
                
                // 滚动到结果区域 - 修复Chrome兼容性问题
                setTimeout(() => {
                    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }, 100);
            }
        }, 100); // 延迟确保DOM更新完成
    }

    displayComprehensiveAnalysisResult(analysisData) {
        console.log('Displaying comprehensive analysis result:', analysisData);
        
        const resultDiv = document.getElementById('analysis-result');
        const contentDiv = document.getElementById('analysis-content');
        
        if (!resultDiv || !contentDiv) {
            console.error('Required DOM elements not found');
            return;
        }
        
        // 显示结果区域
        resultDiv.style.display = 'block';
        
        const {
            investment_text,
            amount,
            risk_tolerance,
            sentiment_analysis,
            anomaly_analysis,
            correlation_analysis,
            regime_analysis,
            symbols
        } = analysisData;
        
        let html = `
            <div class="alert alert-success">
                <i class="fas fa-check-circle me-2"></i>
                综合市场分析完成！
            </div>
            
            <div class="card mb-4">
                <div class="card-header bg-primary text-white">
                    <h5 class="mb-0">
                        <i class="fas fa-chart-line me-2"></i>
                        投资需求分析
                    </h5>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-6">
                            <p><strong>投资需求:</strong> ${investment_text}</p>
                            <p><strong>投资金额:</strong> ¥${parseFloat(amount || 0).toLocaleString()}</p>
                        </div>
                        <div class="col-md-6">
                            <p><strong>风险偏好:</strong> 
                                <span class="badge ${
                                    risk_tolerance === 'conservative' ? 'bg-success' :
                                    risk_tolerance === 'moderate' ? 'bg-warning' :
                                    risk_tolerance === 'aggressive' ? 'bg-danger' : 'bg-secondary'
                                }">
                                    ${risk_tolerance === 'conservative' ? '保守型' :
                                      risk_tolerance === 'moderate' ? '稳健型' :
                                      risk_tolerance === 'aggressive' ? '激进型' :
                                      risk_tolerance === 'very_aggressive' ? '非常激进型' : '未知'}
                                </span>
                            </p>
                            <p><strong>分析股票:</strong> ${symbols.join(', ')}</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // 情感分析结果
        if (sentiment_analysis) {
            html += `
                <div class="card mb-4">
                    <div class="card-header">
                        <h6 class="mb-0">
                            <i class="fas fa-heart me-2"></i>
                            智能情感分析 (FIN-R1模型)
                        </h6>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <p><strong>分析器:</strong> ${sentiment_analysis.analyzer_model || 'FIN-R1'}</p>
                                <p><strong>文本数量:</strong> ${sentiment_analysis.text_count || 1}</p>
                            </div>
                            <div class="col-md-6">
                                <p><strong>整体情感:</strong> 
                                    <span class="badge ${
                                        sentiment_analysis.results?.overall_sentiment > 0 ? 'bg-success' :
                                        sentiment_analysis.results?.overall_sentiment < 0 ? 'bg-danger' : 'bg-warning'
                                    }">
                                        ${sentiment_analysis.results?.overall_sentiment > 0 ? '正面' :
                                          sentiment_analysis.results?.overall_sentiment < 0 ? '负面' : '中性'}
                                    </span>
                                </p>
                                <p><strong>平均置信度:</strong> ${((sentiment_analysis.results?.average_confidence || 0) * 100).toFixed(1)}%</p>
                            </div>
                        </div>
                        ${sentiment_analysis.results?.sentiment_distribution ? `
                        <div class="mt-3">
                            <h6>情感分布:</h6>
                            <div class="row text-center">
                                <div class="col-md-4">
                                    <div class="financial-stat">
                                        <span class="value text-success">${((sentiment_analysis.results.sentiment_distribution.positive || 0) * 100).toFixed(1)}%</span>
                                        <span class="label">正面情感</span>
                                    </div>
                                </div>
                                <div class="col-md-4">
                                    <div class="financial-stat">
                                        <span class="value text-warning">${((sentiment_analysis.results.sentiment_distribution.neutral || 0) * 100).toFixed(1)}%</span>
                                        <span class="label">中性情感</span>
                                    </div>
                                </div>
                                <div class="col-md-4">
                                    <div class="financial-stat">
                                        <span class="value text-danger">${((sentiment_analysis.results.sentiment_distribution.negative || 0) * 100).toFixed(1)}%</span>
                                        <span class="label">负面情感</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                        ` : ''}
                    </div>
                </div>
            `;
        }
        
        // 异常检测结果
        if (anomaly_analysis && anomaly_analysis.length > 0) {
            html += `
                <div class="card mb-4">
                    <div class="card-header">
                        <h6 class="mb-0">
                            <i class="fas fa-exclamation-triangle me-2"></i>
                            异常检测分析
                        </h6>
                    </div>
                    <div class="card-body">
                        <div class="row">
            `;
            
            anomaly_analysis.forEach((anomaly, index) => {
                if (anomaly && anomaly.results) {
                    const totalAnomalies = Object.values(anomaly.results).reduce((sum, result) => sum + (result.count || 0), 0);
                    html += `
                        <div class="col-md-6 mb-3">
                            <div class="card border-left-warning">
                                <div class="card-body">
                                    <h6 class="card-title text-primary">${anomaly.symbol}</h6>
                                    <p class="mb-1"><strong>检测方法:</strong> ${anomaly.detection_method}</p>
                                    <p class="mb-1"><strong>异常数量:</strong> ${totalAnomalies}</p>
                                    <p class="mb-0"><strong>棄测期间:</strong> ${anomaly.detection_period?.start_date} 至 ${anomaly.detection_period?.end_date}</p>
                                </div>
                            </div>
                        </div>
                    `;
                }
            });
            
            html += `
                        </div>
                    </div>
                </div>
            `;
        }
        
        // 相关性分析结果
        if (correlation_analysis) {
            html += `
                <div class="card mb-4">
                    <div class="card-header">
                        <h6 class="mb-0">
                            <i class="fas fa-project-diagram me-2"></i>
                            股票相关性分析
                        </h6>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <p><strong>分析方法:</strong> ${correlation_analysis.correlation_method}</p>
                                <p><strong>分析股票:</strong> ${correlation_analysis.symbols.join(', ')}</p>
                            </div>
                            <div class="col-md-6">
                                <p><strong>分析期间:</strong> ${correlation_analysis.analysis_period?.start_date} 至 ${correlation_analysis.analysis_period?.end_date}</p>
                                <p><strong>高相关对:</strong> ${correlation_analysis.highly_correlated_pairs?.length || 0} 对</p>
                            </div>
                        </div>
                        ${correlation_analysis.insights && correlation_analysis.insights.length > 0 ? `
                        <div class="mt-3">
                            <h6>分析洞察:</h6>
                            <ul class="list-unstyled">
                                ${correlation_analysis.insights.map(insight => `<li class="mb-1"><i class="fas fa-lightbulb text-warning me-2"></i>${insight}</li>`).join('')}
                            </ul>
                        </div>
                        ` : ''}
                    </div>
                </div>
            `;
        }
        
        // 市场状态检测结果
        if (regime_analysis) {
            html += `
                <div class="card mb-4">
                    <div class="card-header">
                        <h6 class="mb-0">
                            <i class="fas fa-wave-square me-2"></i>
                            市场状态检测
                        </h6>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <p><strong>当前状态:</strong> 
                                    <span class="badge bg-info">${regime_analysis.current_regime?.regime || '未知'}</span>
                                </p>
                                <p><strong>置信度:</strong> ${((regime_analysis.current_regime?.confidence || 0) * 100).toFixed(1)}%</p>
                            </div>
                            <div class="col-md-6">
                                <p><strong>持续天数:</strong> ${regime_analysis.current_regime?.duration_days || 0} 天</p>
                                <p><strong>检测方法:</strong> ${regime_analysis.detection_methods?.join(', ') || '未知'}</p>
                            </div>
                        </div>
                        ${regime_analysis.stress_indicators ? `
                        <div class="mt-3">
                            <h6>市场压力指标:</h6>
                            <div class="small text-muted">
                                <p>检测到的市场状态显示当前市场情况。</p>
                            </div>
                        </div>
                        ` : ''}
                    </div>
                </div>
            `;
        }
        
        // 投资建议汇总
        html += `
            <div class="card mb-4">
                <div class="card-header bg-success text-white">
                    <h5 class="mb-0">
                        <i class="fas fa-lightbulb me-2"></i>
                        综合投资建议
                    </h5>
                </div>
                <div class="card-body">
                    <div class="alert alert-info">
                        <h6>基于综合分析的建议:</h6>
                        <ul class="mb-0">
        `;
        
        // 基于各种分析结果生成建议
        if (sentiment_analysis?.results?.overall_sentiment > 0) {
            html += '<li>情感分析显示正面趋势，市场情绪较好，可考虑适度增持。</li>';
        } else if (sentiment_analysis?.results?.overall_sentiment < 0) {
            html += '<li>情感分析显示负面趋势，建议谨慎操作，等待更好时机。</li>';
        }
        
        if (anomaly_analysis && anomaly_analysis.length > 0) {
            const totalAnomalies = anomaly_analysis.reduce((sum, anomaly) => {
                return sum + Object.values(anomaly.results || {}).reduce((s, r) => s + (r.count || 0), 0);
            }, 0);
            
            if (totalAnomalies > 5) {
                html += '<li>检测到较多异常，市场波动较大，建议控制仓位和风险。</li>';
            } else if (totalAnomalies > 0) {
                html += '<li>检测到少量异常，属于正常波动范围，可正常操作。</li>';
            }
        }
        
        if (correlation_analysis?.highly_correlated_pairs?.length > 3) {
            html += '<li>股票间相关性较高，建议分散投资，减少集中度风险。</li>';
        }
        
        if (risk_tolerance === 'conservative') {
            html += '<li>根据您的保守型风险偏好，建议优先考虑稳健型股票和债券。</li>';
        } else if (risk_tolerance === 'aggressive') {
            html += '<li>根据您的激进型风险偏好，可考虑成长型股票和高收益投资机会。</li>';
        }
        
        html += `
                            <li>建议定期关注市场动态，及时调整投资策略。</li>
                        </ul>
                    </div>
                </div>
            </div>
        `;
        
        contentDiv.innerHTML = html;
        
        // 滚动到结果区域
        setTimeout(() => {
            resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 100);
    }

    displaySentimentAnalysisResult(result) {
        console.log('Displaying sentiment analysis result:', result);
        
        const resultDiv = document.getElementById('analysis-result');
        const contentDiv = document.getElementById('analysis-content');
        
        if (!resultDiv || !contentDiv) {
            console.error('Required DOM elements not found');
            return;
        }
        
        // 显示结果区域
        resultDiv.style.display = 'block';
        
        if (result.status !== 'completed') {
            contentDiv.innerHTML = `
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    情感分析未完成: ${result.message || '未知错误'}
                </div>
            `;
            return;
        }
        
        // 提取关键数据
        const individualResults = result.individual_results || {};
        const marketSentiment = result.market_sentiment || {};
        const symbols = result.symbols || [];
        
        // 生成HTML内容
        let html = `
            <div class="alert alert-success">
                <i class="fas fa-check-circle me-2"></i>
                情感分析完成！
            </div>
            
            <div class="card mb-4">
                <div class="card-header bg-primary text-white">
                    <h5 class="mb-0">
                        <i class="fas fa-chart-line me-2"></i>
                        市场整体情感
                    </h5>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-4">
                            <p><strong>情感分数:</strong> ${(marketSentiment.sentiment_score || 0).toFixed(3)}</p>
                        </div>
                        <div class="col-md-4">
                            <p><strong>情感标签:</strong> 
                                <span class="badge ${(marketSentiment.sentiment_score || 0) > 0 ? 'bg-success' : 'bg-warning'}">
                                    ${(marketSentiment.sentiment_score || 0) > 0 ? '正面' : '中性'}
                                </span>
                            </p>
                        </div>
                        <div class="col-md-4">
                            <p><strong>置信度:</strong> ${((marketSentiment.confidence || 0) * 100).toFixed(1)}%</p>
                        </div>
                    </div>
                    <div class="mt-3">
                        <p><strong>分析股票:</strong> ${symbols.join(', ')}</p>
                        <p><strong>分析时间:</strong> ${result.execution_time ? result.execution_time.toFixed(2) : 'N/A'}秒</p>
                    </div>
                </div>
            </div>
        `;
        
        // 添加个股情感分析
        if (Object.keys(individualResults).length > 0) {
            html += `
                <div class="card mb-4">
                    <div class="card-header">
                        <h6 class="mb-0">个股情感分析</h6>
                    </div>
                    <div class="card-body">
                        <div class="row">
            `;
            
            Object.entries(individualResults).forEach(([symbol, data]) => {
                const sentimentScore = data.sentiment_score || 0;
                const confidence = data.confidence || 0;
                const sentimentLabel = sentimentScore > 0 ? '正面' : sentimentScore < 0 ? '负面' : '中性';
                
                html += `
                    <div class="col-md-6 mb-3">
                        <div class="card border-left-primary">
                            <div class="card-body">
                                <h6 class="card-title text-primary">${symbol}</h6>
                                <p class="mb-1"><strong>情感分数:</strong> ${sentimentScore.toFixed(3)}</p>
                                <p class="mb-1"><strong>情感标签:</strong> 
                                    <span class="badge ${sentimentScore > 0 ? 'bg-success' : sentimentScore < 0 ? 'bg-danger' : 'bg-warning'}">
                                        ${sentimentLabel}
                                    </span>
                                </p>
                                <p class="mb-0"><strong>置信度:</strong> ${(confidence * 100).toFixed(1)}%</p>
                            </div>
                        </div>
                    </div>
                `;
            });
            
            html += `
                        </div>
                    </div>
                </div>
            `;
        }
        
        contentDiv.innerHTML = html;
        
        // 滚动到结果区域
        setTimeout(() => {
            resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 100);
    }

    displaySimpleAnalysisResult(result) {
        console.log('Using simple display for analysis result:', result);
        
        const resultDiv = document.getElementById('analysis-result');
        const contentDiv = document.getElementById('analysis-content');
        
        if (!resultDiv || !contentDiv) {
            console.error('Required DOM elements not found for simple display');
            return;
        }
        
        // 显示结果区域
        resultDiv.style.display = 'block';
        
        if (result.status !== 'success') {
            contentDiv.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    多智能体分析失败: ${result.message || '未知错误'}
                </div>
            `;
            return;
        }
        
        // 简化的显示内容
        const consensusRecommendation = result.consensus_recommendation;
        const consensusConfidence = result.consensus_confidence;
        const consensusReasoning = result.consensus_reasoning;
        const individualAnalyses = result.individual_analyses || [];
        
        let html = `
            <div class="alert alert-success">
                <i class="fas fa-check-circle me-2"></i>
                多智能体分析完成！
            </div>
            
            <div class="card mb-4">
                <div class="card-header bg-primary text-white">
                    <h5 class="mb-0">
                        <i class="fas fa-robot me-2"></i>
                        最终推荐
                    </h5>
                </div>
                <div class="card-body">
                    <p><strong>操作:</strong> 
                        <span class="badge ${consensusRecommendation === 'buy' ? 'bg-success' : consensusRecommendation === 'sell' ? 'bg-danger' : 'bg-warning'}">
                            ${consensusRecommendation === 'buy' ? '买入' : consensusRecommendation === 'sell' ? '卖出' : '持有'}
                        </span>
                    </p>
                    <p><strong>目标股票:</strong> ${result.symbols ? result.symbols.join(', ') : '无'}</p>
                    <p><strong>置信度:</strong> ${(consensusConfidence * 100).toFixed(1)}%</p>
                    <p><strong>推荐理由:</strong> ${consensusReasoning}</p>
                </div>
            </div>
        `;
        
        // 添加各智能体分析（简化版）
        if (individualAnalyses.length > 0) {
            html += `
                <div class="card mb-4">
                    <div class="card-header">
                        <h6 class="mb-0">各智能体分析</h6>
                    </div>
                    <div class="card-body">
            `;
            
            individualAnalyses.forEach(analysis => {
                html += `
                    <div class="mb-3 p-3 border rounded">
                        <h6 class="text-primary">${analysis.agent_name}</h6>
                        <p class="small">${analysis.reasoning}</p>
                        <div class="d-flex justify-content-between">
                            <small class="text-muted">
                                推荐: ${analysis.recommendation === 'buy' ? '买入' : analysis.recommendation === 'sell' ? '卖出' : '持有'}
                            </small>
                            <small class="text-muted">
                                置信度: ${(analysis.confidence * 100).toFixed(1)}%
                            </small>
                        </div>
                        ${analysis.key_factors && analysis.key_factors.length > 0 ? `
                        <div class="mt-2">
                            <small class="text-success">
                                <strong>关键因素:</strong> ${analysis.key_factors.join(', ')}
                            </small>
                        </div>
                        ` : ''}
                        ${analysis.risk_factors && analysis.risk_factors.length > 0 ? `
                        <div class="mt-1">
                            <small class="text-warning">
                                <strong>风险因素:</strong> ${analysis.risk_factors.join(', ')}
                            </small>
                        </div>
                        ` : ''}
                    </div>
                `;
            });
            
            html += `
                    </div>
                </div>
            `;
        }
        
        // 设置内容
        contentDiv.innerHTML = html;
        
        // 滚动到结果区域
        setTimeout(() => {
            resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 100);
    }

    async loadPortfolio() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/v1/portfolio/positions`);
            const result = await response.json();
            
            if (result.status === 'error') {
                throw new Error(result.error);
            }
            
            const positions = result.data.positions;

            const tbody = document.getElementById('portfolio-positions');
            if (tbody) {
                tbody.innerHTML = positions.map(pos => `
                    <tr>
                        <td>
                            <strong>${pos.symbol}</strong><br>
                            <small class="text-muted">${pos.name}</small>
                        </td>
                        <td>${pos.quantity.toLocaleString()}</td>
                        <td>¥${pos.cost_price.toFixed(2)}</td>
                        <td>¥${pos.current_price.toFixed(2)}</td>
                        <td>¥${pos.market_value.toLocaleString()}</td>
                        <td class="${pos.unrealized_pnl >= 0 ? 'text-success' : 'text-danger'}">
                            ${pos.unrealized_pnl >= 0 ? '+' : ''}¥${Math.abs(pos.unrealized_pnl).toLocaleString()}
                        </td>
                        <td class="${pos.pnl_rate >= 0 ? 'text-success' : 'text-danger'}">
                            ${pos.pnl_rate >= 0 ? '+' : ''}${pos.pnl_rate.toFixed(2)}%
                        </td>
                        <td>
                            <span class="badge bg-secondary">${(pos.weight * 100).toFixed(1)}%</span>
                        </td>
                        <td>
                            <button class="btn btn-sm btn-outline-primary me-1">调整</button>
                            <button class="btn btn-sm btn-outline-danger">清仓</button>
                        </td>
                    </tr>
                `).join('');
            }

            // 更新组合统计
            const totalMarketValue = positions.reduce((sum, pos) => sum + pos.market_value, 0);
            const totalCost = positions.reduce((sum, pos) => sum + (pos.quantity * pos.cost_price), 0);
            const totalPnl = totalMarketValue - totalCost;

            if (document.getElementById('total-market-value')) {
                document.getElementById('total-market-value').textContent = `¥${totalMarketValue.toLocaleString()}`;
            }
            if (document.getElementById('total-cost')) {
                document.getElementById('total-cost').textContent = `¥${totalCost.toLocaleString()}`;
            }
            if (document.getElementById('floating-pnl')) {
                document.getElementById('floating-pnl').textContent = `¥${totalPnl.toLocaleString()}`;
            }
            if (document.getElementById('position-count')) {
                document.getElementById('position-count').textContent = positions.length;
            }
            
        } catch (error) {
            console.error('Failed to load portfolio:', error);
            this.showAlert('获取投资组合数据失败', 'error');
        }
    }

    async runBacktest() {
        const strategy = document.getElementById('strategy-select').value;
        const symbol = document.getElementById('symbol-select').value;
        const startDate = document.getElementById('start-date').value;
        const endDate = document.getElementById('end-date').value;
        const initialCapital = document.getElementById('initial-capital').value;

        this.showLoading();

        try {
            // 调用真实的API
            const response = await fetch(`${this.apiBaseUrl}/api/v1/backtest/run`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    strategy: strategy,
                    symbol: symbol,
                    start_date: startDate,
                    end_date: endDate,
                    initial_capital: parseFloat(initialCapital)
                })
            });
            
            const result = await response.json();
            
            if (result.status === 'error') {
                throw new Error(result.error);
            }
            
            this.displayBacktestResults(result.data);
            this.showAlert('回测完成', 'success');
        } catch (error) {
            console.error('回测失败:', error);
            this.showAlert('回测失败，请稍后重试', 'danger');
        } finally {
            this.hideLoading();
        }
    }

    displayBacktestResults(results) {
        document.getElementById('total-return').textContent = `+${results.total_return}%`;
        document.getElementById('annual-return').textContent = `+${results.annualized_return}%`;
        document.getElementById('sharpe-ratio-backtest').textContent = results.sharpe_ratio.toFixed(2);
        document.getElementById('max-drawdown-backtest').textContent = `${results.max_drawdown}%`;
        
        // 更新其他回测指标
        if (document.getElementById('win-rate-backtest')) {
            document.getElementById('win-rate-backtest').textContent = `${(results.win_rate * 100).toFixed(1)}%`;
        }
        if (document.getElementById('total-trades-backtest')) {
            document.getElementById('total-trades-backtest').textContent = results.total_trades.toString();
        }
        if (document.getElementById('profit-factor')) {
            document.getElementById('profit-factor').textContent = results.profit_factor.toFixed(2);
        }

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
            dates.push(date.toLocaleDateString('zh-CN', {month: 'short', day: 'numeric'}));
            
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
                    borderWidth: 3,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    fill: true,
                    tension: 0.4
                }, {
                    label: '基准净值',
                    data: benchmarkValues,
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            usePointStyle: true,
                            padding: 20
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(44, 62, 80, 0.9)',
                        titleColor: '#fff',
                        bodyColor: '#ecf0f1',
                        borderColor: 'rgba(52, 152, 219, 0.5)',
                        borderWidth: 1,
                        cornerRadius: 8,
                        callbacks: {
                            label: function(context) {
                                return `${context.dataset.label}: ¥${context.parsed.y.toLocaleString('zh-CN', {maximumFractionDigits: 0})}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: '#7f8c8d',
                            maxRotation: 0,
                            autoSkip: true,
                            maxTicksLimit: 8
                        }
                    },
                    y: {
                        beginAtZero: false,
                        grid: {
                            color: 'rgba(127, 140, 141, 0.1)'
                        },
                        ticks: {
                            color: '#7f8c8d',
                            callback: function(value) {
                                return '¥' + (value / 10000).toFixed(0) + '万';
                            }
                        }
                    }
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                }
            }
        });
    }

    async collectData() {
        const symbol = document.getElementById('data-symbol').value;
        const period = document.getElementById('data-period').value;

        if (!symbol.trim()) {
            this.showAlert('请输入股票代码', 'warning');
            return;
        }

        this.showLoading();

        try {
            // 调用真实的API
            const response = await fetch(`${this.apiBaseUrl}/api/v1/data/collect`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    symbol: symbol,
                    period: period,
                    data_type: 'daily'
                })
            });
            
            const result = await response.json();
            
            if (result.status === 'error') {
                throw new Error(result.error);
            }
            
            this.showAlert(`成功收集 ${symbol} 的 ${period} 数据`, 'success');
            this.loadDataOverview();
        } catch (error) {
            console.error('数据收集失败:', error);
            this.showAlert('数据收集失败，请稍后重试', 'danger');
        } finally {
            this.hideLoading();
        }
    }

    async loadDataOverview() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/v1/data/overview`);
            const result = await response.json();
            
            if (result.status === 'error') {
                throw new Error(result.error);
            }
            
            const dataOverview = result.data.symbols;

            const tbody = document.getElementById('data-overview');
            if (tbody) {
                tbody.innerHTML = dataOverview.map(item => `
                    <tr>
                        <td><strong>${item.symbol}</strong></td>
                        <td>${item.name}</td>
                        <td>${item.records_count}</td>
                        <td>¥${item.latest_price.toFixed(2)}</td>
                        <td class="${item.price_change >= 0 ? 'text-success' : 'text-danger'}">
                            ${item.price_change >= 0 ? '+' : ''}${item.price_change.toFixed(2)} (${item.price_change_pct.toFixed(2)}%)
                        </td>
                        <td>${item.update_time}</td>
                        <td>
                            <button class="btn btn-sm btn-outline-primary me-1">查看</button>
                            <button class="btn btn-sm btn-outline-success">更新</button>
                        </td>
                    </tr>
                `).join('');
            }
            
        } catch (error) {
            console.error('Failed to load data overview:', error);
            this.showAlert('获取数据概览失败', 'error');
        }
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

    showAlert(message, type = 'info') {
        // 移除现有的alert
        const existingAlert = document.querySelector('.floating-alert');
        if (existingAlert) {
            existingAlert.remove();
        }

        // 创建新的alert
        const alert = document.createElement('div');
        alert.className = `alert alert-${type} floating-alert`;
        alert.innerHTML = `
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'warning' ? 'exclamation-triangle' : type === 'danger' ? 'exclamation-circle' : 'info-circle'} me-2"></i>
            ${message}
        `;
        
        // 添加样式
        alert.style.position = 'fixed';
        alert.style.top = '90px';
        alert.style.right = '20px';
        alert.style.zIndex = '9999';
        alert.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
        alert.style.maxWidth = '400px';
        alert.style.animation = 'fadeIn 0.3s ease-out';
        
        // 添加到页面
        document.body.appendChild(alert);
        
        // 3秒后自动移除
        setTimeout(() => {
            if (alert.parentNode) {
                alert.style.animation = 'fadeOut 0.3s ease-out';
                setTimeout(() => {
                    if (alert.parentNode) {
                        alert.remove();
                    }
                }, 300);
            }
        }, 3000);
    }
}

// 初始化应用 - 修复Chrome兼容性问题
function initializeApp() {
    // 确保DOM完全加载
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            window.finLoomApp = new FinLoomApp();
        });
    } else {
        // DOM已经加载完成
        window.finLoomApp = new FinLoomApp();
    }
}

// 添加淡出动画CSS - 修复Chrome兼容性问题
function addAnimationStyles() {
    const style = document.createElement('style');
    style.textContent = `
        @keyframes fadeOut {
            from { opacity: 1; transform: translateY(0); }
            to { opacity: 0; transform: translateY(-20px); }
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .floating-alert {
            animation: fadeIn 0.3s ease-out;
        }
    `;
    document.head.appendChild(style);
}

// 等待页面完全加载后初始化
if (document.readyState === 'complete' || document.readyState === 'interactive') {
    // 页面已经加载或正在交互，直接初始化
    addAnimationStyles();
    initializeApp();
} else {
    // 页面仍在加载中，等待加载完成
    window.addEventListener('load', () => {
        addAnimationStyles();
        initializeApp();
    });
}
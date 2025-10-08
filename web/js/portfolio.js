/**
 * 投资组合管理 JavaScript
 */

// 全局变量
let portfolioData = null;
let assetChart = null;
let sectorChart = null;

// 页面加载时初始化
document.addEventListener('DOMContentLoaded', () => {
    loadPortfolioData();
    initCharts();
});

/**
 * 加载投资组合数据
 */
async function loadPortfolioData() {
    try {
        const response = await fetch('/api/v1/portfolio/overview');
        portfolioData = await response.json();
        
        if (portfolioData.error) {
            console.error('加载数据失败:', portfolioData.error);
            return;
        }
        
        updateOverviewCards();
        updateCharts();
        await loadPositions();
    } catch (error) {
        console.error('加载投资组合数据失败:', error);
    }
}

/**
 * 更新概览卡片
 */
function updateOverviewCards() {
    if (!portfolioData) return;
    
    document.getElementById('total-value').textContent = 
        `¥${portfolioData.total_value.toLocaleString('zh-CN', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
    
    document.getElementById('total-return').textContent = 
        `¥${portfolioData.total_return.toLocaleString('zh-CN', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
    
    document.getElementById('positions-count').textContent = portfolioData.positions_count || 0;
    
    const availableCash = portfolioData.total_value * (portfolioData.asset_allocation.cash / 100);
    document.getElementById('available-cash').textContent = 
        `¥${availableCash.toLocaleString('zh-CN', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
    
    // 更新变化
    const dailyChangeElement = document.getElementById('daily-change');
    const isPositive = portfolioData.daily_change_rate >= 0;
    dailyChangeElement.className = `metric-change ${isPositive ? 'positive' : 'negative'}`;
    dailyChangeElement.innerHTML = `
        <i class="fas fa-arrow-${isPositive ? 'up' : 'down'}"></i>
        ${isPositive ? '+' : ''}${portfolioData.daily_change_rate.toFixed(2)}%
    `;
    
    // 更新收益率
    const returnRateElement = document.getElementById('return-rate');
    const returnPositive = portfolioData.return_rate >= 0;
    returnRateElement.className = `metric-change ${returnPositive ? 'positive' : 'negative'}`;
    returnRateElement.innerHTML = `
        <i class="fas fa-arrow-${returnPositive ? 'up' : 'down'}"></i>
        ${returnPositive ? '+' : ''}${portfolioData.return_rate.toFixed(2)}%
    `;
}

/**
 * 初始化图表
 */
function initCharts() {
    const assetCtx = document.getElementById('asset-allocation-chart');
    const sectorCtx = document.getElementById('sector-allocation-chart');
    
    assetChart = new Chart(assetCtx, {
        type: 'doughnut',
        data: {
            labels: [],
            datasets: [{
                data: [],
                backgroundColor: [
                    '#667eea',
                    '#f093fb',
                    '#4facfe',
                    '#00f2fe',
                    '#43e97b'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.label}: ${context.parsed}%`;
                        }
                    }
                }
            }
        }
    });
    
    sectorChart = new Chart(sectorCtx, {
        type: 'doughnut',
        data: {
            labels: [],
            datasets: [{
                data: [],
                backgroundColor: [
                    '#10b981',
                    '#3b82f6',
                    '#f59e0b',
                    '#ef4444',
                    '#8b5cf6'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.label}: ${context.parsed}%`;
                        }
                    }
                }
            }
        }
    });
}

/**
 * 更新图表数据
 */
function updateCharts() {
    if (!portfolioData) return;
    
    // 更新资产配置图表
    if (assetChart && portfolioData.asset_allocation) {
        assetChart.data.labels = Object.keys(portfolioData.asset_allocation).map(key => {
            const nameMap = {
                'stocks': '股票',
                'cash': '现金',
                'bonds': '债券',
                'funds': '基金',
                'other': '其他'
            };
            return nameMap[key] || key;
        });
        assetChart.data.datasets[0].data = Object.values(portfolioData.asset_allocation);
        assetChart.update();
    }
    
    // 更新行业分布图表
    if (sectorChart && portfolioData.sector_allocation) {
        sectorChart.data.labels = Object.keys(portfolioData.sector_allocation);
        sectorChart.data.datasets[0].data = Object.values(portfolioData.sector_allocation);
        sectorChart.update();
    }
}

/**
 * 加载持仓明细
 */
async function loadPositions() {
    try {
        const response = await fetch('/api/v1/portfolio/positions');
        const positions = await response.json();
        
        const tbody = document.getElementById('positions-tbody');
        
        if (!positions || positions.length === 0) {
            tbody.innerHTML = '<tr><td colspan="10" class="loading-cell">暂无持仓数据</td></tr>';
            return;
        }
        
        tbody.innerHTML = positions.map(pos => `
            <tr>
                <td class="position-symbol">${pos.symbol}</td>
                <td>${pos.name}</td>
                <td>${pos.quantity}</td>
                <td>¥${pos.cost_price.toFixed(2)}</td>
                <td>¥${pos.current_price.toFixed(2)}</td>
                <td>¥${pos.market_value.toLocaleString('zh-CN', {minimumFractionDigits: 2})}</td>
                <td class="${pos.profit_loss >= 0 ? 'positive' : 'negative'}">
                    ¥${pos.profit_loss.toLocaleString('zh-CN', {minimumFractionDigits: 2})}
                </td>
                <td class="${pos.profit_loss_rate >= 0 ? 'positive' : 'negative'}">
                    ${pos.profit_loss_rate >= 0 ? '+' : ''}${pos.profit_loss_rate.toFixed(2)}%
                </td>
                <td>${pos.weight.toFixed(2)}%</td>
                <td>
                    <button class="btn-action" onclick="viewPositionDetail('${pos.symbol}')">详情</button>
                </td>
            </tr>
        `).join('');
    } catch (error) {
        console.error('加载持仓明细失败:', error);
        document.getElementById('positions-tbody').innerHTML = 
            '<tr><td colspan="10" class="loading-cell">加载失败</td></tr>';
    }
}

/**
 * 刷新数据
 */
async function refreshData() {
    const btn = event.target.closest('button');
    const icon = btn.querySelector('i');
    icon.classList.add('fa-spin');
    
    await loadPortfolioData();
    
    setTimeout(() => {
        icon.classList.remove('fa-spin');
    }, 500);
}

/**
 * 显示优化弹窗
 */
function showOptimizeModal() {
    document.getElementById('optimize-modal').classList.add('active');
}

/**
 * 关闭弹窗
 */
function closeModal(modalId) {
    document.getElementById(modalId).classList.remove('active');
}

/**
 * 开始优化
 */
async function startOptimize() {
    const targetReturn = document.getElementById('target-return').value;
    const riskTolerance = document.getElementById('risk-tolerance').value;
    const optimizeTarget = document.getElementById('optimize-target').value;
    
    try {
        const response = await fetch('/api/v1/portfolio/optimize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                target_return: parseFloat(targetReturn),
                risk_tolerance: riskTolerance,
                optimize_target: optimizeTarget
            })
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            showOptimizationResult(result);
            closeModal('optimize-modal');
        } else {
            alert('优化失败: ' + result.message);
        }
    } catch (error) {
        console.error('优化失败:', error);
        alert('优化失败，请重试');
    }
}

/**
 * 显示优化结果
 */
function showOptimizationResult(result) {
    const modal = document.getElementById('optimize-result-modal');
    const body = document.getElementById('optimize-result-body');
    
    body.innerHTML = `
        <div class="optimize-section">
            <h4>最优权重配置</h4>
            <div class="weight-grid">
                ${Object.entries(result.optimal_weights).map(([symbol, weight]) => `
                    <div class="weight-item">
                        <div class="weight-symbol">${symbol}</div>
                        <div class="weight-value">${(weight * 100).toFixed(1)}%</div>
                    </div>
                `).join('')}
            </div>
        </div>
        
        <div class="optimize-section">
            <h4>预期指标</h4>
            <div class="metrics-grid">
                <div class="metric-item">
                    <div class="metric-item-label">预期收益率</div>
                    <div class="metric-item-value" style="color: #10b981;">
                        ${result.expected_return.toFixed(2)}%
                    </div>
                </div>
                <div class="metric-item">
                    <div class="metric-item-label">预期波动率</div>
                    <div class="metric-item-value" style="color: #f59e0b;">
                        ${result.expected_volatility.toFixed(2)}%
                    </div>
                </div>
                <div class="metric-item">
                    <div class="metric-item-label">夏普比率</div>
                    <div class="metric-item-value" style="color: #3b82f6;">
                        ${result.sharpe_ratio.toFixed(2)}
                    </div>
                </div>
            </div>
        </div>
        
        <div class="optimize-section">
            <h4>调仓建议</h4>
            ${result.rebalance_suggestions.map(sug => `
                <div class="suggestion-item">
                    <span class="suggestion-action ${sug.action}">${sug.action === 'buy' ? '买入' : '减持'}</span>
                    <span class="suggestion-symbol">${sug.name} (${sug.symbol})</span>
                    <span style="margin-left: 8px; color: #6b7280;">${sug.quantity} 股</span>
                    <div class="suggestion-reason">${sug.reason}</div>
                </div>
            `).join('')}
        </div>
    `;
    
    modal.classList.add('active');
}

/**
 * 应用优化建议
 */
function applyOptimization() {
    if (confirm('确认应用优化建议？这将调整您的投资组合配置。')) {
        alert('优化建议已应用！');
        closeModal('optimize-result-modal');
        loadPortfolioData(); // 重新加载数据
    }
}

/**
 * 查看持仓详情
 */
function viewPositionDetail(symbol) {
    alert(`查看 ${symbol} 的详细信息\n\n此功能将显示：\n- 历史价格走势\n- 技术指标分析\n- 新闻和公告\n- 财务数据`);
}

/**
 * 导出持仓数据
 */
function exportPositions() {
    alert('正在导出持仓数据到Excel...\n\n导出内容包括：\n- 持仓明细\n- 盈亏统计\n- 资产配置\n- 行业分布');
}

// 点击弹窗外部关闭
document.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal')) {
        e.target.classList.remove('active');
    }
});














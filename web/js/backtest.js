/**
 * 策略回测 JavaScript
 */

let equityChart = null;

/**
 * 运行回测
 */
async function runBacktest() {
    const btn = event.target;
    const originalText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 回测中...';
    
    try {
        const params = {
            strategy: document.getElementById('strategy-type').value,
            symbol: document.getElementById('stock-symbol').value,
            start_date: document.getElementById('start-date').value,
            end_date: document.getElementById('end-date').value,
            initial_capital: parseFloat(document.getElementById('initial-capital').value),
            commission_rate: parseFloat(document.getElementById('commission-rate').value) / 100,
            slippage: parseFloat(document.getElementById('slippage').value)
        };
        
        const response = await fetch('/api/v1/backtest/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });
        
        const result = await response.json();
        
        if (result.status === 'completed') {
            displayBacktestResults(result);
        } else {
            alert('回测失败: ' + (result.message || '未知错误'));
        }
        
    } catch (error) {
        console.error('回测失败:', error);
        alert('回测失败，请检查网络连接或稍后重试');
    } finally {
        btn.disabled = false;
        btn.innerHTML = originalText;
    }
}

/**
 * 显示回测结果
 */
function displayBacktestResults(result) {
    // 隐藏空状态，显示结果
    document.getElementById('empty-state').style.display = 'none';
    document.getElementById('results-container').style.display = 'block';
    
    // 更新关键指标
    document.getElementById('total-return').textContent = `${result.total_return.toFixed(2)}%`;
    document.getElementById('total-return').className = `metric-value ${result.total_return >= 0 ? 'positive' : 'negative'}`;
    
    document.getElementById('annual-return').textContent = `${result.annualized_return.toFixed(2)}%`;
    document.getElementById('annual-return').className = `metric-value ${result.annualized_return >= 0 ? 'positive' : 'negative'}`;
    
    document.getElementById('sharpe-ratio').textContent = result.sharpe_ratio.toFixed(2);
    document.getElementById('max-drawdown').textContent = `${result.max_drawdown.toFixed(2)}%`;
    document.getElementById('win-rate').textContent = `${(result.win_rate * 100).toFixed(1)}%`;
    document.getElementById('profit-factor').textContent = result.profit_factor.toFixed(2);
    
    // 更新资金曲线图表
    updateEquityCurveChart(result.equity_curve);
    
    // 更新交易记录
    updateTradesTable(result.trades);
}

/**
 * 更新资金曲线图表
 */
function updateEquityCurveChart(equityCurve) {
    const ctx = document.getElementById('equity-curve-chart');
    
    if (equityChart) {
        equityChart.destroy();
    }
    
    equityChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: equityCurve.map(point => point.date),
            datasets: [{
                label: '账户净值',
                data: equityCurve.map(point => point.value),
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
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
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            return `净值: ¥${context.parsed.y.toLocaleString('zh-CN', {minimumFractionDigits: 2})}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return '¥' + (value / 10000).toFixed(1) + 'w';
                        }
                    }
                },
                x: {
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            }
        }
    });
}

/**
 * 更新交易记录表格
 */
function updateTradesTable(trades) {
    const tbody = document.getElementById('trades-tbody');
    document.getElementById('trade-count').textContent = `共 ${trades.length} 笔交易`;
    
    tbody.innerHTML = trades.map(trade => {
        const actionClass = trade.action === 'BUY' ? 'positive' : 'negative';
        const actionText = trade.action === 'BUY' ? '买入' : '卖出';
        const actionIcon = trade.action === 'BUY' ? 'arrow-up' : 'arrow-down';
        
        return `
            <tr>
                <td>${trade.date}</td>
                <td>
                    <span class="${actionClass}" style="font-weight: 600;">
                        <i class="fas fa-${actionIcon}"></i> ${actionText}
                    </span>
                </td>
                <td>¥${trade.price.toFixed(2)}</td>
                <td>${trade.quantity}</td>
                <td>¥${(trade.price * trade.quantity).toLocaleString('zh-CN', {minimumFractionDigits: 2})}</td>
                <td>¥${(trade.price * trade.quantity * 0.0003).toFixed(2)}</td>
            </tr>
        `;
    }).join('');
}

/**
 * 加载历史回测
 */
function loadHistoryTests() {
    alert('历史回测记录\n\n功能说明：\n- 查看过往回测结果\n- 对比不同策略表现\n- 导出回测报告\n- 策略参数优化历史');
}














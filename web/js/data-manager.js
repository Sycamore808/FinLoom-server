/**
 * 数据管理 JavaScript
 */

document.addEventListener('DOMContentLoaded', () => {
    loadDataOverview();
    loadDataSources();
});

async function loadDataOverview() {
    try {
        const response = await fetch('/api/v1/data/overview');
        if (!response.ok) {
            throw new Error(`API请求失败: ${response.status}`);
        }
        
        const result = await response.json();
        const data = result.data || result;
        
        // 计算数据库大小（模拟值，基于记录数）
        const recordCount = data.total_records || 0;
        const dbSizeMB = Math.round(recordCount * 0.005); // 假设每条记录约5KB
        
        document.getElementById('db-size').textContent = `${dbSizeMB} MB`;
        document.getElementById('stock-count').textContent = data.total_symbols || 0;
        document.getElementById('record-count').textContent = recordCount.toLocaleString('zh-CN');
        document.getElementById('last-update').textContent = data.last_update || new Date().toLocaleString('zh-CN');
    } catch (error) {
        console.error('加载数据概览失败:', error);
        // 显示错误提示
        document.getElementById('db-size').textContent = '-- MB';
        document.getElementById('stock-count').textContent = '0';
        document.getElementById('record-count').textContent = '0';
        document.getElementById('last-update').textContent = '加载失败';
    }
}

async function loadDataSources() {
    try {
        const response = await fetch('/api/v1/data/overview');
        if (!response.ok) {
            throw new Error(`API请求失败: ${response.status}`);
        }
        
        const result = await response.json();
        const data = result.data || result;
        
        const tbody = document.getElementById('datasources-tbody');
        const symbols = data.symbols || [];
        
        if (symbols.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="loading-cell">暂无数据</td></tr>';
            return;
        }
        
        // 将股票数据转换为数据源显示
        tbody.innerHTML = symbols.map(stock => `
            <tr>
                <td><strong>${stock.symbol} - ${stock.name}</strong></td>
                <td>
                    <span style="color: ${stock.records_count > 0 ? '#10b981' : '#6b7280'};">
                        <i class="fas fa-circle" style="font-size: 8px;"></i>
                        ${stock.records_count > 0 ? '活跃' : '无数据'}
                    </span>
                </td>
                <td>${(stock.records_count || 0).toLocaleString('zh-CN')}</td>
                <td>${stock.update_time || new Date().toLocaleString('zh-CN')}</td>
                <td>
                    <button class="btn-action" onclick="updateSource('${stock.symbol}')">
                        <i class="fas fa-sync-alt"></i> 更新
                    </button>
                </td>
            </tr>
        `).join('');
    } catch (error) {
        console.error('加载数据源失败:', error);
        const tbody = document.getElementById('datasources-tbody');
        tbody.innerHTML = '<tr><td colspan="5" style="color: #ef4444; text-align: center; padding: 2rem;">加载失败，请检查后端服务</td></tr>';
    }
}

function refreshDataOverview() {
    const btn = event.target.closest('button');
    const icon = btn.querySelector('i');
    icon.classList.add('fa-spin');
    
    loadDataOverview();
    loadDataSources();
    
    setTimeout(() => {
        icon.classList.remove('fa-spin');
    }, 1000);
}

function showCollectModal() {
    document.getElementById('collect-modal').classList.add('active');
}

function closeModal(modalId) {
    document.getElementById(modalId).classList.remove('active');
}

async function startCollect() {
    const symbolsText = document.getElementById('symbols-input').value.trim();
    if (!symbolsText) {
        alert('请输入股票代码');
        return;
    }
    
    const symbols = symbolsText.split(',').map(s => s.trim()).filter(s => s);
    const checkboxes = document.querySelectorAll('#collect-modal input[type="checkbox"]:checked');
    const dataTypes = Array.from(checkboxes).map(cb => cb.value);
    
    try {
        const response = await fetch('/api/v1/data/collect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbols, data_types: dataTypes })
        });
        
        const result = await response.json();
        
        if (result.status === 'completed') {
            alert(`数据采集完成！\n成功: ${result.total_collected}/${symbols.length}`);
            closeModal('collect-modal');
            loadDataOverview();
            loadDataSources();
        } else {
            alert('采集失败: ' + result.message);
        }
    } catch (error) {
        console.error('数据采集失败:', error);
        alert('采集失败，请重试');
    }
}

function updateSource(sourceName) {
    alert(`更新数据源: ${sourceName}\n\n功能说明：\n- 增量更新最新数据\n- 修复缺失数据\n- 验证数据完整性`);
}

document.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal')) {
        e.target.classList.remove('active');
    }
});








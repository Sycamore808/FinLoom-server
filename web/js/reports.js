/**
 * 报告中心 JavaScript
 */

document.addEventListener('DOMContentLoaded', () => {
    loadReports();
});

async function loadReports() {
    try {
        const response = await fetch('/api/v1/reports/list');
        const reports = await response.json();
        
        document.getElementById('total-reports').textContent = reports.length;
        displayReports(reports);
    } catch (error) {
        console.error('加载报告列表失败:', error);
        document.getElementById('reports-tbody').innerHTML = 
            '<tr><td colspan="6" class="loading-cell">加载失败</td></tr>';
    }
}

function displayReports(reports) {
    const tbody = document.getElementById('reports-tbody');
    
    if (!reports || reports.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="loading-cell">暂无报告</td></tr>';
        return;
    }
    
    tbody.innerHTML = reports.map(report => {
        const typeMap = {
            'portfolio': '投资组合',
            'backtest': '回测报告',
            'quarterly': '季度总结',
            'annual': '年度总结',
            'stock_analysis': '个股分析'
        };
        
        const statusMap = {
            'completed': { text: '已完成', color: '#10b981' },
            'generating': { text: '生成中', color: '#f59e0b' },
            'failed': { text: '失败', color: '#ef4444' }
        };
        
        const status = statusMap[report.status] || statusMap.completed;
        
        return `
            <tr>
                <td><strong>${report.title}</strong></td>
                <td>${typeMap[report.type] || report.type}</td>
                <td>${report.create_date}</td>
                <td>
                    <span style="color: ${status.color};">
                        <i class="fas fa-circle" style="font-size: 8px;"></i>
                        ${status.text}
                    </span>
                </td>
                <td>${(report.size_kb / 1024).toFixed(2)} MB</td>
                <td>
                    <button class="btn-action" onclick="viewReport('${report.id}')">查看</button>
                    <button class="btn-action" onclick="downloadReport('${report.id}')">下载</button>
                </td>
            </tr>
        `;
    }).join('');
}

function showGenerateModal() {
    document.getElementById('generate-modal').classList.add('active');
    
    // 设置默认日期
    const today = new Date();
    const lastMonth = new Date(today.getFullYear(), today.getMonth() - 1, 1);
    document.getElementById('report-start-date').value = lastMonth.toISOString().split('T')[0];
    document.getElementById('report-end-date').value = today.toISOString().split('T')[0];
}

function closeModal(modalId) {
    document.getElementById(modalId).classList.remove('active');
}

async function generateReport() {
    const type = document.getElementById('report-type').value;
    const title = document.getElementById('report-title').value.trim();
    const startDate = document.getElementById('report-start-date').value;
    const endDate = document.getElementById('report-end-date').value;
    
    if (!title) {
        alert('请输入报告标题');
        return;
    }
    
    try {
        const response = await fetch('/api/v1/reports/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                type,
                params: {
                    title,
                    start_date: startDate,
                    end_date: endDate
                }
            })
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            alert('报告生成任务已提交！\n请稍后在报告列表中查看');
            closeModal('generate-modal');
            loadReports();
        } else {
            alert('生成失败: ' + result.message);
        }
    } catch (error) {
        console.error('生成报告失败:', error);
        alert('生成失败，请重试');
    }
}

function refreshReports() {
    const btn = event.target.closest('button');
    const icon = btn.querySelector('i');
    icon.classList.add('fa-spin');
    
    loadReports();
    
    setTimeout(() => {
        icon.classList.remove('fa-spin');
    }, 500);
}

function filterReports(type) {
    alert(`筛选报告类型: ${type}\n\n功能说明：\n- 按类型筛选报告\n- 快速定位目标报告`);
}

function viewReport(reportId) {
    alert(`查看报告: ${reportId}\n\n功能说明：\n- 在线预览报告内容\n- 查看详细分析和图表\n- 导出为PDF或Excel`);
}

function downloadReport(reportId) {
    alert(`下载报告: ${reportId}\n\n正在准备下载...\n格式: PDF / Excel\n大小: 约 2-5 MB`);
}

document.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal')) {
        e.target.classList.remove('active');
    }
});














/**
 * 系统设置 JavaScript
 */

function switchTab(tabName) {
    // 切换导航按钮状态
    document.querySelectorAll('.settings-nav-item').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.closest('button').classList.add('active');
    
    // 切换内容区域
    document.querySelectorAll('.settings-section').forEach(section => {
        section.classList.remove('active');
    });
    document.getElementById(`${tabName}-section`).classList.add('active');
}

function toggleSwitch(element) {
    element.classList.toggle('active');
}

function saveSettings() {
    // 模拟保存设置
    alert('设置已保存！\n\n保存的设置项：\n- 基本设置\n- 模型配置\n- 数据源设置\n- 风险参数\n- 通知设置');
}














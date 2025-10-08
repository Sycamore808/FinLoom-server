/**
 * FIN-R1模型管理器 JavaScript
 * 负责检测、下载和配置FIN-R1模型
 */

// API基础路径
const API_BASE = '/api/v1/model';

// 页面加载时初始化
document.addEventListener('DOMContentLoaded', async () => {
    console.log('[模型管理器] 页面加载完成，开始初始化...');
    
    // 加载所有初始信息
    await Promise.all([
        loadModelStatus(),
        checkSystemRequirements(),
        loadAvailableDisks()
    ]);
    
    console.log('[模型管理器] 初始化完成');
});

/**
 * 加载模型状态
 */
async function loadModelStatus() {
    const container = document.getElementById('model-status');
    
    try {
        const response = await fetch(`${API_BASE}/status`);
        
        if (!response.ok) {
            throw new Error(`API请求失败: ${response.status}`);
        }
        
        const status = await response.json();
        console.log('[模型状态]', status);
        
        // 渲染模型状态
        const html = `
            <div class="status-row">
                <span class="status-label">配置状态</span>
                <span class="status-value">
                    ${status.configured ? 
                        '<span class="status-badge badge-success">✓ 已配置</span>' : 
                        '<span class="status-badge badge-danger">✗ 未配置</span>'}
                </span>
            </div>
            ${status.configured ? `
            <div class="status-row">
                <span class="status-label">模型路径</span>
                <span class="status-value">${status.path || '未知'}</span>
            </div>
            <div class="status-row">
                <span class="status-label">模型存在</span>
                <span class="status-value">
                    ${status.exists ? 
                        '<span class="status-badge badge-success">✓ 是</span>' : 
                        '<span class="status-badge badge-danger">✗ 否</span>'}
                </span>
            </div>
            ${status.exists ? `
            <div class="status-row">
                <span class="status-label">模型大小</span>
                <span class="status-value">${status.size_mb.toFixed(2)} MB</span>
            </div>
            ` : ''}
            ` : ''}
        `;
        
        container.innerHTML = html;
        
        // 如果模型未配置或不存在，显示警告
        if (!status.configured || !status.exists) {
            const alert = document.createElement('div');
            alert.className = 'alert alert-warning';
            alert.style.marginTop = '16px';
            alert.innerHTML = `
                <i class="fas fa-exclamation-triangle"></i>
                <strong>警告：</strong>FIN-R1模型${!status.configured ? '未配置' : '文件不存在'}，对话功能将无法使用。
                请${!status.configured ? '配置' : '重新下载'}模型。
            `;
            container.appendChild(alert);
        }
        
    } catch (error) {
        console.error('[模型状态] 加载失败:', error);
        container.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-circle"></i>
                加载模型状态失败：${error.message}
            </div>
        `;
    }
}

/**
 * 检查系统配置
 */
async function checkSystemRequirements() {
    const container = document.getElementById('system-requirements');
    
    try {
        const response = await fetch(`${API_BASE}/system-requirements`);
        
        if (!response.ok) {
            throw new Error(`API请求失败: ${response.status}`);
        }
        
        const requirements = await response.json();
        console.log('[系统配置]', requirements);
        
        // 渲染系统信息卡片
        const systemInfo = requirements.system_info || {};
        const infoCards = `
            <div class="system-info-grid">
                <div class="info-card">
                    <div class="icon"><i class="fas fa-microchip"></i></div>
                    <div class="label">CPU核心数</div>
                    <div class="value">${systemInfo.cpu_count || 0}</div>
                </div>
                <div class="info-card">
                    <div class="icon"><i class="fas fa-memory"></i></div>
                    <div class="label">内存大小</div>
                    <div class="value">${(systemInfo.memory_gb || 0).toFixed(1)} GB</div>
                </div>
                <div class="info-card">
                    <div class="icon"><i class="fas fa-hdd"></i></div>
                    <div class="label">可用空间</div>
                    <div class="value">${(systemInfo.disk_free_gb || 0).toFixed(1)} GB</div>
                </div>
                <div class="info-card">
                    <div class="icon"><i class="fas fa-microchip"></i></div>
                    <div class="label">GPU状态</div>
                    <div class="value">${systemInfo.gpu_available ? '✓ 可用' : '✗ 不可用'}</div>
                </div>
            </div>
        `;
        
        // 构建完整HTML
        let html = infoCards;
        
        // 如果有问题，显示警告
        if (requirements.issues && requirements.issues.length > 0) {
            const alertClass = requirements.meets_requirements ? 'alert-warning' : 'alert-danger';
            const alertIcon = requirements.meets_requirements ? 'exclamation-triangle' : 'exclamation-circle';
            
            html += `
                <div class="alert ${alertClass}" style="margin-top: 20px;">
                    <i class="fas fa-${alertIcon}"></i>
                    <strong>${requirements.meets_requirements ? '警告' : '错误'}：</strong>
                    系统配置${requirements.meets_requirements ? '基本满足但存在以下建议' : '不满足以下要求'}：
                    <div class="issues-list">
                        ${requirements.issues.map(issue => `
                            <div class="issue-item">
                                <i class="fas fa-circle" style="font-size: 6px;"></i>
                                <span>${issue}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        } else {
            html += `
                <div class="alert alert-info" style="margin-top: 20px; background: #d4edda; color: #155724; border-left-color: #28a745;">
                    <i class="fas fa-check-circle"></i>
                    <strong>系统配置满足要求</strong>，可以正常运行FIN-R1模型。
                </div>
            `;
        }
        
        container.innerHTML = html;
        
    } catch (error) {
        console.error('[系统配置] 检查失败:', error);
        container.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-circle"></i>
                检查系统配置失败：${error.message}
            </div>
        `;
    }
}

/**
 * 加载可用磁盘
 */
async function loadAvailableDisks() {
    const container = document.getElementById('disk-list');
    
    try {
        const response = await fetch(`${API_BASE}/available-disks`);
        
        if (!response.ok) {
            throw new Error(`API请求失败: ${response.status}`);
        }
        
        const disks = await response.json();
        console.log('[可用磁盘]', disks);
        
        if (!Array.isArray(disks) || disks.length === 0) {
            container.innerHTML = `
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle"></i>
                    未检测到可用磁盘
                </div>
            `;
            return;
        }
        
        // 渲染磁盘列表
        const html = disks.map(disk => {
            const usedPercent = disk.percent_used || 0;
            const freeGB = disk.free_gb || 0;
            const hasSpace = freeGB >= 10; // 至少需要10GB
            
            return `
                <div class="disk-item">
                    <div class="disk-header">
                        <span class="disk-label">
                            <i class="fas fa-hdd"></i> ${disk.label}
                        </span>
                        <span class="disk-free">
                            ${freeGB.toFixed(1)} GB 可用 / ${(disk.total_gb || 0).toFixed(1)} GB
                            ${hasSpace ? 
                                '<span class="status-badge badge-success" style="margin-left: 8px;">✓ 空间充足</span>' : 
                                '<span class="status-badge badge-danger" style="margin-left: 8px;">✗ 空间不足</span>'}
                        </span>
                    </div>
                    <div class="disk-progress">
                        <div class="disk-progress-bar" style="width: ${usedPercent}%"></div>
                    </div>
                </div>
            `;
        }).join('');
        
        container.innerHTML = html;
        
        // 更新下载模态框中的磁盘选择器
        const diskSelect = document.getElementById('download-disk');
        if (diskSelect) {
            diskSelect.innerHTML = disks.map(disk => 
                `<option value="${disk.path}" ${disk.free_gb >= 10 ? '' : 'disabled'}>
                    ${disk.label} - ${disk.free_gb.toFixed(1)} GB 可用
                </option>`
            ).join('');
            
            // 自动选择第一个有足够空间的磁盘
            const suitableDisk = disks.find(d => d.free_gb >= 10);
            if (suitableDisk) {
                diskSelect.value = suitableDisk.path;
                // 更新路径输入框
                const pathInput = document.getElementById('download-path');
                if (pathInput) {
                    pathInput.value = `${suitableDisk.path}AI_Models${suitableDisk.path.includes('\\') ? '\\' : '/'}Fin-R1`;
                }
            }
        }
        
    } catch (error) {
        console.error('[可用磁盘] 加载失败:', error);
        container.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-circle"></i>
                加载磁盘信息失败：${error.message}
            </div>
        `;
    }
}

/**
 * 搜索本地模型
 */
async function searchLocalModels() {
    const container = document.getElementById('local-models');
    
    // 显示加载状态
    container.innerHTML = `
        <div class="loading">
            <i class="fas fa-spinner"></i>
            <p>正在搜索本地模型，请稍候...</p>
        </div>
    `;
    
    try {
        const response = await fetch(`${API_BASE}/search-local`);
        
        if (!response.ok) {
            throw new Error(`API请求失败: ${response.status}`);
        }
        
        const models = await response.json();
        console.log('[本地模型]', models);
        
        if (!Array.isArray(models) || models.length === 0) {
            container.innerHTML = `
                <div class="alert alert-warning">
                    <i class="fas fa-info-circle"></i>
                    未找到本地FIN-R1模型。请下载新模型或手动配置模型路径。
                </div>
            `;
            return;
        }
        
        // 渲染模型列表
        const html = models.map(model => `
            <div class="model-item">
                <div class="model-info">
                    <h3>${model.name}</h3>
                    <div class="model-meta">
                        <span><i class="fas fa-folder"></i> ${model.path}</span><br>
                        <span><i class="fas fa-database"></i> 大小: ${model.size_mb.toFixed(2)} MB</span>
                    </div>
                </div>
                <button class="btn btn-primary" onclick="useModel('${model.path}')">
                    <i class="fas fa-check"></i> 使用此模型
                </button>
            </div>
        `).join('');
        
        container.innerHTML = html;
        
    } catch (error) {
        console.error('[本地模型] 搜索失败:', error);
        container.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-circle"></i>
                搜索失败：${error.message}
            </div>
        `;
    }
}

/**
 * 使用指定的模型
 */
async function useModel(modelPath) {
    if (!confirm(`确认使用此模型？\n路径: ${modelPath}`)) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/use-existing`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model_path: modelPath })
        });
        
        if (!response.ok) {
            throw new Error(`API请求失败: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            alert('✓ 模型配置成功！\n\n现在可以使用智能对话和策略制定功能了。');
            // 重新加载模型状态
            await loadModelStatus();
        } else {
            alert('✗ 配置失败：' + result.message);
        }
        
    } catch (error) {
        console.error('[使用模型] 失败:', error);
        alert('配置模型失败：' + error.message);
    }
}

/**
 * 显示下载模态框
 */
function showDownloadModal() {
    document.getElementById('download-modal').classList.add('active');
}

/**
 * 显示手动配置模态框
 */
function showManualConfigModal() {
    document.getElementById('manual-config-modal').classList.add('active');
}

/**
 * 显示自定义搜索模态框
 */
function showCustomSearchModal() {
    document.getElementById('custom-search-modal').classList.add('active');
}

/**
 * 关闭模态框
 */
function closeModal(modalId) {
    document.getElementById(modalId).classList.remove('active');
}

/**
 * 开始下载模型
 */
async function startDownload() {
    const path = document.getElementById('download-path').value;
    const source = document.getElementById('download-source').value;
    
    if (!path) {
        alert('请输入安装路径');
        return;
    }
    
    // 禁用开始按钮
    const button = document.getElementById('btn-start-download');
    button.disabled = true;
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 下载中...';
    
    // 显示进度条
    const progressContainer = document.getElementById('download-progress');
    progressContainer.style.display = 'block';
    
    try {
        // 更新进度
        updateProgress(0, '正在连接服务器...');
        
        const response = await fetch(`${API_BASE}/download`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                target_path: path,
                source: source
            })
        });
        
        updateProgress(10, '开始下载模型...');
        
        if (!response.ok) {
            throw new Error(`下载请求失败: ${response.status}`);
        }
        
        // 模拟进度更新（实际应该从服务器获取实时进度）
        for (let i = 10; i <= 90; i += 10) {
            await new Promise(resolve => setTimeout(resolve, 3000));
            updateProgress(i, `下载中... ${i}%`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            updateProgress(100, '下载完成！');
            setTimeout(() => {
                alert('✓ 模型下载成功！\n\n现在可以使用智能对话和策略制定功能了。');
                closeModal('download-modal');
                // 重新加载模型状态
                loadModelStatus();
            }, 500);
        } else {
            throw new Error(result.message);
        }
        
    } catch (error) {
        console.error('[下载模型] 失败:', error);
        updateProgress(0, '下载失败');
        alert('下载失败：' + error.message + '\n\n请检查网络连接，或尝试手动下载。');
    } finally {
        button.disabled = false;
        button.innerHTML = '<i class="fas fa-download"></i> 开始下载';
    }
}

/**
 * 更新进度条
 */
function updateProgress(percent, text) {
    const progressBar = document.getElementById('download-progress-bar');
    const progressText = document.getElementById('download-progress-text');
    
    if (progressBar) {
        progressBar.style.width = percent + '%';
        progressBar.textContent = percent + '%';
    }
    
    if (progressText) {
        progressText.textContent = text;
    }
}

/**
 * 手动配置模型路径
 */
async function configureManualPath() {
    const path = document.getElementById('manual-model-path').value;
    
    if (!path) {
        alert('请输入模型路径');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/use-existing`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model_path: path })
        });
        
        if (!response.ok) {
            throw new Error(`API请求失败: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            alert('✓ 模型配置成功！');
            closeModal('manual-config-modal');
            await loadModelStatus();
        } else {
            alert('✗ 配置失败：' + result.message);
        }
        
    } catch (error) {
        console.error('[手动配置] 失败:', error);
        alert('配置失败：' + error.message);
    }
}

/**
 * 自定义路径搜索
 */
async function searchCustomPaths() {
    const pathsText = document.getElementById('custom-search-paths').value;
    
    if (!pathsText.trim()) {
        alert('请输入搜索路径');
        return;
    }
    
    const paths = pathsText.split('\n').map(p => p.trim()).filter(p => p);
    
    closeModal('custom-search-modal');
    
    const container = document.getElementById('local-models');
    container.innerHTML = `
        <div class="loading">
            <i class="fas fa-spinner"></i>
            <p>正在搜索指定路径中的模型...</p>
        </div>
    `;
    
    try {
        const response = await fetch(`${API_BASE}/search-local?paths=${encodeURIComponent(JSON.stringify(paths))}`);
        
        if (!response.ok) {
            throw new Error(`API请求失败: ${response.status}`);
        }
        
        const models = await response.json();
        
        if (!Array.isArray(models) || models.length === 0) {
            container.innerHTML = `
                <div class="alert alert-warning">
                    <i class="fas fa-info-circle"></i>
                    在指定路径中未找到FIN-R1模型。
                </div>
            `;
            return;
        }
        
        const html = models.map(model => `
            <div class="model-item">
                <div class="model-info">
                    <h3>${model.name}</h3>
                    <div class="model-meta">
                        <span><i class="fas fa-folder"></i> ${model.path}</span><br>
                        <span><i class="fas fa-database"></i> 大小: ${model.size_mb.toFixed(2)} MB</span>
                    </div>
                </div>
                <button class="btn btn-primary" onclick="useModel('${model.path}')">
                    <i class="fas fa-check"></i> 使用此模型
                </button>
            </div>
        `).join('');
        
        container.innerHTML = html;
        
    } catch (error) {
        console.error('[自定义搜索] 失败:', error);
        container.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-circle"></i>
                搜索失败：${error.message}
            </div>
        `;
    }
}

// 监听磁盘选择变化，自动更新路径
document.addEventListener('DOMContentLoaded', () => {
    const diskSelect = document.getElementById('download-disk');
    const pathInput = document.getElementById('download-path');
    
    if (diskSelect && pathInput) {
        diskSelect.addEventListener('change', (e) => {
            const diskPath = e.target.value;
            if (diskPath) {
                const separator = diskPath.includes('\\') ? '\\' : '/';
                pathInput.value = `${diskPath}AI_Models${separator}Fin-R1`;
            }
        });
    }
});

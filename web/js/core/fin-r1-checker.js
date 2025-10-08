/**
 * FIN-R1模型检测器
 * 在所有需要FIN-R1的页面自动检查模型状态
 */

(function() {
    'use strict';
    
    const FINR1Checker = {
        // 需要FIN-R1的页面
        requiresModel: [
            '/web/pages/chat-mode.html',
            '/web/pages/strategy-mode.html',
            '/chat-mode',
            '/strategy-mode'
        ],
        
        // 检查当前页面是否需要FIN-R1
        isModelRequired() {
            const path = window.location.pathname;
            return this.requiresModel.some(p => path.includes(p));
        },
        
        // 检查模型状态
        async checkModelStatus() {
            try {
                const response = await fetch('/api/v1/model/status', {
                    timeout: 5000
                });
                
                if (!response.ok) {
                    console.warn('[FIN-R1检测器] API响应异常:', response.status);
                    return { configured: false, exists: false, error: true };
                }
                
                const status = await response.json();
                console.log('[FIN-R1检测器] 模型状态:', status);
                return status;
                
            } catch (error) {
                console.error('[FIN-R1检测器] 检查失败:', error);
                return { configured: false, exists: false, error: true };
            }
        },
        
        // 显示模型未配置提示
        showModelNotConfiguredWarning(status) {
            // 创建提示横幅
            const banner = document.createElement('div');
            banner.id = 'fin-r1-warning-banner';
            banner.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                background: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%);
                color: white;
                padding: 16px 24px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                z-index: 9999;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                animation: slideDown 0.3s ease;
            `;
            
            // 添加动画
            const style = document.createElement('style');
            style.textContent = `
                @keyframes slideDown {
                    from { transform: translateY(-100%); }
                    to { transform: translateY(0); }
                }
            `;
            document.head.appendChild(style);
            
            const message = document.createElement('div');
            message.style.cssText = 'display: flex; align-items: center; gap: 12px; flex: 1;';
            message.innerHTML = `
                <i class="fas fa-exclamation-triangle" style="font-size: 24px;"></i>
                <div>
                    <strong style="font-size: 16px;">FIN-R1模型${!status.configured ? '未配置' : '文件不存在'}</strong>
                    <p style="margin: 4px 0 0 0; font-size: 14px; opacity: 0.9;">
                        智能对话和策略制定功能需要FIN-R1模型支持。请先配置模型后再使用这些功能。
                    </p>
                </div>
            `;
            
            const actions = document.createElement('div');
            actions.style.cssText = 'display: flex; gap: 12px;';
            
            const configButton = document.createElement('button');
            configButton.textContent = '立即配置';
            configButton.style.cssText = `
                padding: 10px 20px;
                background: white;
                color: #ef4444;
                border: none;
                border-radius: 8px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s;
            `;
            configButton.onmouseover = () => configButton.style.transform = 'scale(1.05)';
            configButton.onmouseout = () => configButton.style.transform = 'scale(1)';
            configButton.onclick = () => {
                window.location.href = '/web/pages/model-manager.html';
            };
            
            const closeButton = document.createElement('button');
            closeButton.innerHTML = '×';
            closeButton.style.cssText = `
                padding: 8px 16px;
                background: rgba(255,255,255,0.2);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 24px;
                cursor: pointer;
                transition: all 0.2s;
            `;
            closeButton.onclick = () => banner.remove();
            
            actions.appendChild(configButton);
            actions.appendChild(closeButton);
            
            banner.appendChild(message);
            banner.appendChild(actions);
            
            // 插入到页面顶部
            document.body.insertBefore(banner, document.body.firstChild);
            
            // 调整页面内容位置，避免被横幅遮挡
            const mainContent = document.querySelector('.chat-container, .strategy-container, main');
            if (mainContent) {
                mainContent.style.paddingTop = 'calc(80px + 64px)'; // 横幅高度 + 导航栏高度
            }
        },
        
        // 初始化检测
        async init() {
            // 只在需要模型的页面检查
            if (!this.isModelRequired()) {
                console.log('[FIN-R1检测器] 当前页面不需要FIN-R1模型');
                return;
            }
            
            console.log('[FIN-R1检测器] 开始检查模型状态...');
            
            const status = await this.checkModelStatus();
            
            // 如果模型未配置或不存在，显示警告
            if (!status.configured || !status.exists) {
                console.warn('[FIN-R1检测器] 模型未就绪:', status);
                this.showModelNotConfiguredWarning(status);
            } else {
                console.log('[FIN-R1检测器] ✓ 模型已就绪');
            }
        }
    };
    
    // 页面加载完成后自动检查
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => FINR1Checker.init());
    } else {
        FINR1Checker.init();
    }
    
    // 导出到全局
    window.FINR1Checker = FINR1Checker;
})();








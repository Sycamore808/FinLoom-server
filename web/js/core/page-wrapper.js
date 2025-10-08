/**
 * 页面包装器 - 统一的侧边栏管理
 * 用于在所有功能页面中显示侧边栏
 */

(function() {
    'use strict';
    
    // 检查是否需要显示侧边栏（某些页面如登录页不需要）
    const shouldShowSidebar = () => {
        const path = window.location.pathname;
        const noSidebarPages = ['/web/login.html', '/web/splash.html', '/login.html', '/splash.html'];
        return !noSidebarPages.some(page => path.includes(page));
    };
    
    // 创建侧边栏HTML
    const createSidebarHTML = () => {
        return `
<aside class="sidebar">
    <div class="sidebar-header">
        <h1>
            <i class="fas fa-chart-network"></i>
            <span>FinLoom</span>
        </h1>
    </div>
    <nav class="sidebar-menu">
        <a href="/web/index_upgraded.html" class="menu-item">
            <i class="fas fa-tachometer-alt"></i>
            <span>仪表板</span>
        </a>
        <div class="menu-item-group">
            <a href="#" class="menu-item has-submenu" data-menu-id="intelligent-analysis" onclick="toggleSubmenu(event, this)">
                <i class="fas fa-brain"></i>
                <span>智能分析</span>
                <i class="fas fa-chevron-down submenu-arrow"></i>
            </a>
            <div class="submenu">
                <a href="/web/pages/chat-mode.html" class="submenu-item">
                    <i class="fas fa-comments"></i>
                    <span>智能对话</span>
                </a>
                <a href="/web/pages/strategy-mode.html" class="submenu-item">
                    <i class="fas fa-chart-line"></i>
                    <span>策略制定</span>
                </a>
            </div>
        </div>
        <div class="menu-item-group">
            <a href="#" class="menu-item has-submenu" data-menu-id="investment-management" onclick="toggleSubmenu(event, this)">
                <i class="fas fa-briefcase"></i>
                <span>投资管理</span>
                <i class="fas fa-chevron-down submenu-arrow"></i>
            </a>
            <div class="submenu">
                <a href="/web/pages/portfolio.html" class="submenu-item">
                    <i class="fas fa-briefcase"></i>
                    <span>投资组合</span>
                </a>
                <a href="/web/pages/backtest.html" class="submenu-item">
                    <i class="fas fa-history"></i>
                    <span>策略回测</span>
                </a>
            </div>
        </div>
        <a href="/web/pages/data-manager.html" class="menu-item">
            <i class="fas fa-database"></i>
            <span>数据管理</span>
        </a>
        <a href="/web/pages/reports.html" class="menu-item">
            <i class="fas fa-file-alt"></i>
            <span>报告中心</span>
        </a>
        <div class="menu-item-group">
            <a href="#" class="menu-item has-submenu" data-menu-id="system-config" onclick="toggleSubmenu(event, this)">
                <i class="fas fa-cog"></i>
                <span>系统配置</span>
                <i class="fas fa-chevron-down submenu-arrow"></i>
            </a>
            <div class="submenu">
                <a href="/web/pages/settings.html" class="submenu-item">
                    <i class="fas fa-sliders-h"></i>
                    <span>系统设置</span>
                </a>
                <a href="/web/pages/model-manager.html" class="submenu-item">
                    <i class="fas fa-brain"></i>
                    <span>模型管理</span>
                </a>
            </div>
        </div>
    </nav>
</aside>
        `;
    };
    
    // 创建侧边栏样式
    const createSidebarStyles = () => {
        const style = document.createElement('style');
        style.textContent = `
:root {
    --sidebar-width: 260px;
}

.sidebar {
    width: var(--sidebar-width);
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    box-shadow: 4px 0 20px rgba(0,0,0,0.1);
    position: fixed;
    top: 0;
    left: 0;
    height: 100vh;
    overflow-y: auto;
    z-index: 100;
}

.sidebar-header {
    padding: 2rem 1.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}

.sidebar-header h1 {
    font-size: 1.75rem;
    font-weight: 900;
    color: white;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 0;
}

.sidebar-header h1 i {
    color: #3b82f6;
    font-size: 2rem;
}

.sidebar-menu {
    padding: 1.5rem 0;
}

.menu-item {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1rem 1.5rem;
    color: rgba(255,255,255,0.7);
    text-decoration: none;
    transition: all 0.3s;
    position: relative;
    border-left: 3px solid transparent;
}

.menu-item:hover {
    background: rgba(59,130,246,0.1);
    color: white;
    border-left-color: #3b82f6;
}

.menu-item.active {
    background: rgba(59,130,246,0.2);
    color: white;
    border-left-color: #3b82f6;
}

.menu-item i {
    font-size: 1.25rem;
    width: 24px;
}

.menu-item.has-submenu {
    position: relative;
}

.submenu-arrow {
    margin-left: auto;
    font-size: 0.9rem !important;
    transition: transform 0.3s ease;
}

.menu-item.has-submenu.open .submenu-arrow {
    transform: rotate(180deg);
}

.menu-item-group .submenu {
    max-height: 0;
    overflow: hidden;
    transition: max-height 0.3s ease;
    background: rgba(0,0,0,0.2);
}

.menu-item-group:has(.menu-item.has-submenu.open) .submenu {
    max-height: 200px;
}

.submenu-item {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.875rem 1.5rem 0.875rem 3.5rem;
    color: rgba(255,255,255,0.6);
    text-decoration: none;
    transition: all 0.3s;
    font-size: 0.95rem;
}

.submenu-item:hover {
    background: rgba(59,130,246,0.15);
    color: white;
}

.submenu-item.active {
    background: rgba(59,130,246,0.2);
    color: white;
}

.submenu-item i {
    font-size: 1rem;
    width: 20px;
}

/* 为有侧边栏的页面添加左边距 */
body.has-sidebar {
    margin-left: var(--sidebar-width);
}

/* 响应式 */
@media (max-width: 1024px) {
    .sidebar {
        transform: translateX(-100%);
    }
    body.has-sidebar {
        margin-left: 0;
    }
}
        `;
        document.head.appendChild(style);
    };
    
    // 初始化侧边栏
    const initSidebar = () => {
        if (!shouldShowSidebar()) {
            return;
        }
        
        // 添加样式
        createSidebarStyles();
        
        // 添加侧边栏HTML
        const sidebarHTML = createSidebarHTML();
        document.body.insertAdjacentHTML('afterbegin', sidebarHTML);
        document.body.classList.add('has-sidebar');
        
        // 初始化侧边栏状态
        if (window.SidebarState && typeof window.SidebarState.getCurrentPage === 'function') {
            const currentPath = window.location.pathname;
            const menuItems = document.querySelectorAll('.menu-item, .submenu-item');
            
            menuItems.forEach(item => {
                const href = item.getAttribute('href');
                if (href && href !== '#' && currentPath.includes(href)) {
                    item.classList.add('active');
                    
                    // 如果是子菜单项，展开父菜单
                    const parentGroup = item.closest('.menu-item-group');
                    if (parentGroup) {
                        const parentMenu = parentGroup.querySelector('.has-submenu');
                        if (parentMenu) {
                            parentMenu.classList.add('open');
                        }
                    }
                }
            });
        }
    };
    
    // 页面加载时初始化
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initSidebar);
    } else {
        initSidebar();
    }
    
    // 导出全局函数
    window.PageWrapper = {
        initSidebar
    };
})();








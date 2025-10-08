/**
 * 全局侧边栏状态管理
 * 在所有页面保持侧边栏状态一致
 */

(function() {
    'use strict';
    
    // 侧边栏状态管理
    const SidebarState = {
        // 获取展开的菜单
        getExpandedMenus() {
            const saved = localStorage.getItem('fin_expanded_menus');
            return saved ? JSON.parse(saved) : [];
        },
        
        // 保存展开的菜单
        setExpandedMenus(menus) {
            localStorage.setItem('fin_expanded_menus', JSON.stringify(menus));
        },
        
        // 获取当前活动页面
        getCurrentPage() {
            return window.location.pathname;
        },
        
        // 设置活动页面
        setActivePage(path) {
            localStorage.setItem('fin_active_page', path);
        }
    };
    
    // 初始化侧边栏
    function initSidebar() {
        const currentPath = SidebarState.getCurrentPage();
        const expandedMenus = SidebarState.getExpandedMenus();
        
        // 展开保存的菜单
        expandedMenus.forEach(menuId => {
            const menu = document.querySelector(`[data-menu-id="${menuId}"]`);
            if (menu) {
                menu.classList.add('open');
            }
        });
        
        // 设置当前活动项
        const menuItems = document.querySelectorAll('.menu-item, .submenu-item');
        menuItems.forEach(item => {
            const href = item.getAttribute('href');
            if (href && (href === currentPath || currentPath.includes(href))) {
                item.classList.add('active');
                
                // 如果是子菜单项，展开父菜单
                const parentGroup = item.closest('.menu-item-group');
                if (parentGroup) {
                    const parentMenu = parentGroup.querySelector('.has-submenu');
                    if (parentMenu) {
                        parentMenu.classList.add('open');
                    }
                }
            } else {
                item.classList.remove('active');
            }
        });
    }
    
    // 子菜单切换
    window.toggleSubmenu = function(event, element) {
        event.preventDefault();
        event.stopPropagation();
        
        const isOpen = element.classList.contains('open');
        const menuId = element.getAttribute('data-menu-id') || element.textContent.trim();
        
        // 切换状态
        element.classList.toggle('open');
        
        // 更新保存的状态
        const expandedMenus = SidebarState.getExpandedMenus();
        if (!isOpen) {
            // 添加到展开列表
            if (!expandedMenus.includes(menuId)) {
                expandedMenus.push(menuId);
            }
        } else {
            // 从展开列表移除
            const index = expandedMenus.indexOf(menuId);
            if (index > -1) {
                expandedMenus.splice(index, 1);
            }
        }
        
        SidebarState.setExpandedMenus(expandedMenus);
    };
    
    // 页面加载完成后初始化
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initSidebar);
    } else {
        initSidebar();
    }
    
    // 页面卸载时保存当前状态
    window.addEventListener('beforeunload', () => {
        const expandedMenus = [];
        document.querySelectorAll('.menu-item.has-submenu.open').forEach(menu => {
            const menuId = menu.getAttribute('data-menu-id') || menu.textContent.trim();
            expandedMenus.push(menuId);
        });
        SidebarState.setExpandedMenus(expandedMenus);
    });
    
    // 导出到全局
    window.SidebarState = SidebarState;
})();














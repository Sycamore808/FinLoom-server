/**
 * AUE3架构核心 - Actions, UI, Effects 三层架构
 * 
 * Architecture:
 * - Actions层: 处理用户交互和事件调度
 * - UI层: 纯展示逻辑，负责渲染界面
 * - Effects层: 处理副作用（API调用、状态管理、本地存储等）
 */

class AUE3Core {
    constructor() {
        this.state = {};
        this.listeners = new Map();
        this.actions = {};
        this.effects = {};
        this.ui = {};
    }

    /**
     * 注册Actions层
     * @param {Object} actions - 动作处理器对象
     */
    registerActions(actions) {
        this.actions = { ...this.actions, ...actions };
        return this;
    }

    /**
     * 注册Effects层
     * @param {Object} effects - 副作用处理器对象
     */
    registerEffects(effects) {
        this.effects = { ...this.effects, ...effects };
        return this;
    }

    /**
     * 注册UI层
     * @param {Object} ui - UI渲染器对象
     */
    registerUI(ui) {
        this.ui = { ...this.ui, ...ui };
        return this;
    }

    /**
     * 分发动作
     * @param {string} actionName - 动作名称
     * @param {*} payload - 动作载荷
     */
    async dispatch(actionName, payload) {
        const action = this.actions[actionName];
        if (!action) {
            console.error(`Action "${actionName}" not found`);
            return;
        }

        try {
            // 执行action，返回新状态或effect名称
            const result = await action(payload, this.state, this);
            
            // 如果返回状态更新
            if (result && result.state) {
                this.setState(result.state);
            }

            // 如果需要执行副作用
            if (result && result.effect) {
                await this.runEffect(result.effect, result.effectPayload);
            }

            // 如果需要更新UI
            if (result && result.render) {
                this.render(result.render);
            }
        } catch (error) {
            console.error(`Error dispatching action "${actionName}":`, error);
            this.handleError(error);
        }
    }

    /**
     * 执行副作用
     * @param {string} effectName - 副作用名称
     * @param {*} payload - 副作用载荷
     */
    async runEffect(effectName, payload) {
        const effect = this.effects[effectName];
        if (!effect) {
            console.error(`Effect "${effectName}" not found`);
            return;
        }

        try {
            return await effect(payload, this.state, this);
        } catch (error) {
            console.error(`Error running effect "${effectName}":`, error);
            throw error;
        }
    }

    /**
     * 渲染UI
     * @param {string} uiName - UI名称
     * @param {*} data - 渲染数据
     */
    render(uiName, data) {
        const uiRenderer = this.ui[uiName];
        if (!uiRenderer) {
            console.error(`UI renderer "${uiName}" not found`);
            return;
        }

        try {
            uiRenderer(data || this.state, this);
        } catch (error) {
            console.error(`Error rendering UI "${uiName}":`, error);
            this.handleError(error);
        }
    }

    /**
     * 更新状态
     * @param {Object} newState - 新状态
     */
    setState(newState) {
        const oldState = { ...this.state };
        this.state = { ...this.state, ...newState };
        
        // 触发状态变更监听器
        this.notifyListeners('stateChange', { oldState, newState: this.state });
    }

    /**
     * 获取状态
     * @param {string} key - 状态键
     */
    getState(key) {
        return key ? this.state[key] : this.state;
    }

    /**
     * 添加事件监听器
     * @param {string} event - 事件名称
     * @param {Function} callback - 回调函数
     */
    on(event, callback) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, []);
        }
        this.listeners.get(event).push(callback);
        return () => this.off(event, callback);
    }

    /**
     * 移除事件监听器
     * @param {string} event - 事件名称
     * @param {Function} callback - 回调函数
     */
    off(event, callback) {
        if (!this.listeners.has(event)) return;
        const callbacks = this.listeners.get(event);
        const index = callbacks.indexOf(callback);
        if (index > -1) {
            callbacks.splice(index, 1);
        }
    }

    /**
     * 通知监听器
     * @param {string} event - 事件名称
     * @param {*} data - 事件数据
     */
    notifyListeners(event, data) {
        if (!this.listeners.has(event)) return;
        this.listeners.get(event).forEach(callback => {
            try {
                callback(data);
            } catch (error) {
                console.error(`Error in listener for "${event}":`, error);
            }
        });
    }

    /**
     * 错误处理
     * @param {Error} error - 错误对象
     */
    handleError(error) {
        this.notifyListeners('error', error);
        // 可以在这里添加全局错误处理逻辑
    }

    /**
     * 初始化应用
     * @param {Object} initialState - 初始状态
     */
    init(initialState = {}) {
        this.state = initialState;
        this.notifyListeners('init', this.state);
        return this;
    }
}

// 工具函数
const AUE3Utils = {
    /**
     * 创建元素
     * @param {string} tag - 标签名
     * @param {Object} attrs - 属性
     * @param {string|Array} children - 子元素
     */
    createElement(tag, attrs = {}, children = null) {
        const element = document.createElement(tag);
        
        // 设置属性
        Object.entries(attrs).forEach(([key, value]) => {
            if (key === 'className') {
                element.className = value;
            } else if (key === 'style' && typeof value === 'object') {
                Object.assign(element.style, value);
            } else if (key.startsWith('on') && typeof value === 'function') {
                const event = key.substring(2).toLowerCase();
                element.addEventListener(event, value);
            } else {
                element.setAttribute(key, value);
            }
        });

        // 添加子元素
        if (children) {
            if (Array.isArray(children)) {
                children.forEach(child => {
                    if (typeof child === 'string') {
                        element.appendChild(document.createTextNode(child));
                    } else if (child instanceof Node) {
                        element.appendChild(child);
                    }
                });
            } else if (typeof children === 'string') {
                element.textContent = children;
            } else if (children instanceof Node) {
                element.appendChild(children);
            }
        }

        return element;
    },

    /**
     * 查询元素
     * @param {string} selector - 选择器
     * @param {Element} parent - 父元素
     */
    $(selector, parent = document) {
        return parent.querySelector(selector);
    },

    /**
     * 查询所有元素
     * @param {string} selector - 选择器
     * @param {Element} parent - 父元素
     */
    $$(selector, parent = document) {
        return Array.from(parent.querySelectorAll(selector));
    },

    /**
     * 防抖函数
     * @param {Function} func - 函数
     * @param {number} wait - 等待时间
     */
    debounce(func, wait = 300) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    /**
     * 节流函数
     * @param {Function} func - 函数
     * @param {number} limit - 限制时间
     */
    throttle(func, limit = 300) {
        let inThrottle;
        return function executedFunction(...args) {
            if (!inThrottle) {
                func(...args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    },

    /**
     * 格式化日期
     * @param {Date|string} date - 日期
     * @param {string} format - 格式
     */
    formatDate(date, format = 'YYYY-MM-DD HH:mm:ss') {
        const d = new Date(date);
        const year = d.getFullYear();
        const month = String(d.getMonth() + 1).padStart(2, '0');
        const day = String(d.getDate()).padStart(2, '0');
        const hours = String(d.getHours()).padStart(2, '0');
        const minutes = String(d.getMinutes()).padStart(2, '0');
        const seconds = String(d.getSeconds()).padStart(2, '0');

        return format
            .replace('YYYY', year)
            .replace('MM', month)
            .replace('DD', day)
            .replace('HH', hours)
            .replace('mm', minutes)
            .replace('ss', seconds);
    },

    /**
     * 转义HTML
     * @param {string} str - 字符串
     */
    escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    },

    /**
     * 深拷贝
     * @param {*} obj - 对象
     */
    deepClone(obj) {
        return JSON.parse(JSON.stringify(obj));
    },

    /**
     * 合并对象
     * @param {Object} target - 目标对象
     * @param {Object} source - 源对象
     */
    merge(target, source) {
        return Object.assign({}, target, source);
    },

    /**
     * 生成唯一ID
     */
    uuid() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0;
            const v = c === 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    },

    /**
     * 本地存储
     */
    storage: {
        set(key, value) {
            try {
                localStorage.setItem(key, JSON.stringify(value));
                return true;
            } catch (error) {
                console.error('Storage set error:', error);
                return false;
            }
        },
        get(key, defaultValue = null) {
            try {
                const value = localStorage.getItem(key);
                return value ? JSON.parse(value) : defaultValue;
            } catch (error) {
                console.error('Storage get error:', error);
                return defaultValue;
            }
        },
        remove(key) {
            localStorage.removeItem(key);
        },
        clear() {
            localStorage.clear();
        }
    }
};

// 导出
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { AUE3Core, AUE3Utils };
} else {
    window.AUE3Core = AUE3Core;
    window.AUE3Utils = AUE3Utils;
}











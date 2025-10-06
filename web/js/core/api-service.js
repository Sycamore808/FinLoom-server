/**
 * API服务 - 统一的API调用接口
 */

class APIService {
    constructor(baseURL = '') {
        this.baseURL = baseURL || window.location.origin;
        this.defaultHeaders = {
            'Content-Type': 'application/json',
        };
        this.requestInterceptors = [];
        this.responseInterceptors = [];
    }

    /**
     * 添加请求拦截器
     * @param {Function} interceptor - 拦截器函数
     */
    addRequestInterceptor(interceptor) {
        this.requestInterceptors.push(interceptor);
    }

    /**
     * 添加响应拦截器
     * @param {Function} interceptor - 拦截器函数
     */
    addResponseInterceptor(interceptor) {
        this.responseInterceptors.push(interceptor);
    }

    /**
     * 处理请求
     * @param {string} url - URL
     * @param {Object} options - 选项
     */
    async request(url, options = {}) {
        // 构建完整URL
        const fullURL = url.startsWith('http') ? url : `${this.baseURL}${url}`;

        // 合并默认headers
        options.headers = {
            ...this.defaultHeaders,
            ...options.headers
        };

        // 执行请求拦截器
        let config = { url: fullURL, ...options };
        for (const interceptor of this.requestInterceptors) {
            config = await interceptor(config);
        }

        try {
            // 发送请求
            const response = await fetch(config.url, config);

            // 执行响应拦截器
            let result = response;
            for (const interceptor of this.responseInterceptors) {
                result = await interceptor(result);
            }

            // 解析响应
            const contentType = result.headers.get('content-type');
            let data;
            if (contentType && contentType.includes('application/json')) {
                data = await result.json();
            } else {
                data = await result.text();
            }

            // 检查HTTP状态
            if (!result.ok) {
                throw new APIError(
                    data.message || 'Request failed',
                    result.status,
                    data
                );
            }

            return data;
        } catch (error) {
            // 错误处理
            if (error instanceof APIError) {
                throw error;
            }
            throw new APIError(
                error.message || 'Network error',
                0,
                null
            );
        }
    }

    /**
     * GET请求
     * @param {string} url - URL
     * @param {Object} params - 查询参数
     * @param {Object} options - 选项
     */
    async get(url, params = {}, options = {}) {
        // 构建查询字符串
        const queryString = new URLSearchParams(params).toString();
        const fullURL = queryString ? `${url}?${queryString}` : url;

        return this.request(fullURL, {
            method: 'GET',
            ...options
        });
    }

    /**
     * POST请求
     * @param {string} url - URL
     * @param {Object} data - 数据
     * @param {Object} options - 选项
     */
    async post(url, data = {}, options = {}) {
        return this.request(url, {
            method: 'POST',
            body: JSON.stringify(data),
            ...options
        });
    }

    /**
     * PUT请求
     * @param {string} url - URL
     * @param {Object} data - 数据
     * @param {Object} options - 选项
     */
    async put(url, data = {}, options = {}) {
        return this.request(url, {
            method: 'PUT',
            body: JSON.stringify(data),
            ...options
        });
    }

    /**
     * DELETE请求
     * @param {string} url - URL
     * @param {Object} options - 选项
     */
    async delete(url, options = {}) {
        return this.request(url, {
            method: 'DELETE',
            ...options
        });
    }

    /**
     * PATCH请求
     * @param {string} url - URL
     * @param {Object} data - 数据
     * @param {Object} options - 选项
     */
    async patch(url, data = {}, options = {}) {
        return this.request(url, {
            method: 'PATCH',
            body: JSON.stringify(data),
            ...options
        });
    }

    /**
     * 上传文件
     * @param {string} url - URL
     * @param {FormData} formData - 表单数据
     * @param {Object} options - 选项
     */
    async upload(url, formData, options = {}) {
        return this.request(url, {
            method: 'POST',
            body: formData,
            headers: {
                // 不设置Content-Type，让浏览器自动设置
                ...options.headers
            },
            ...options
        });
    }
}

/**
 * API错误类
 */
class APIError extends Error {
    constructor(message, status, data) {
        super(message);
        this.name = 'APIError';
        this.status = status;
        this.data = data;
    }
}

// FinLoom API服务实例
const finloomAPI = new APIService();

// 添加通用错误处理
finloomAPI.addResponseInterceptor(async (response) => {
    if (!response.ok && response.status === 401) {
        // 未授权，跳转到登录页
        console.warn('Unauthorized, redirecting to login...');
        // window.location.href = '/web/login.html';
    }
    return response;
});

// FinLoom API端点
const FinLoomAPI = {
    // 对话相关
    chat: {
        send: (message, conversationId) => 
            finloomAPI.post('/api/chat', { message, conversation_id: conversationId }),
        history: (conversationId) => 
            finloomAPI.get(`/api/chat/history/${conversationId}`),
        conversations: () => 
            finloomAPI.get('/api/chat/conversations'),
        newConversation: () => 
            finloomAPI.post('/api/chat/conversation'),
    },

    // 策略相关
    strategy: {
        generate: (requirements) => 
            finloomAPI.post('/api/strategy/generate', requirements),
        backtest: (strategyId, params) => 
            finloomAPI.post(`/api/strategy/${strategyId}/backtest`, params),
        optimize: (strategyId, params) => 
            finloomAPI.post(`/api/strategy/${strategyId}/optimize`, params),
        code: (strategyId) => 
            finloomAPI.get(`/api/strategy/${strategyId}/code`),
        save: (strategyData) => 
            finloomAPI.post('/api/strategy/save', strategyData),
        list: () => 
            finloomAPI.get('/api/strategy/list'),
    },

    // 市场数据
    market: {
        indices: () => 
            finloomAPI.get('/api/market/indices'),
        quote: (symbol) => 
            finloomAPI.get(`/api/market/quote/${symbol}`),
        news: (limit = 10) => 
            finloomAPI.get('/api/market/news', { limit }),
        sectors: () => 
            finloomAPI.get('/api/market/sectors'),
    },

    // 组合管理
    portfolio: {
        summary: () => 
            finloomAPI.get('/api/portfolio/summary'),
        positions: () => 
            finloomAPI.get('/api/portfolio/positions'),
        performance: (period) => 
            finloomAPI.get('/api/portfolio/performance', { period }),
        trades: (limit = 20) => 
            finloomAPI.get('/api/portfolio/trades', { limit }),
    },

    // 用户相关
    user: {
        profile: () => 
            finloomAPI.get('/api/user/profile'),
        preferences: () => 
            finloomAPI.get('/api/user/preferences'),
        updatePreferences: (preferences) => 
            finloomAPI.put('/api/user/preferences', preferences),
    }
};

// 导出
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { APIService, APIError, FinLoomAPI };
} else {
    window.APIService = APIService;
    window.APIError = APIError;
    window.FinLoomAPI = FinLoomAPI;
}











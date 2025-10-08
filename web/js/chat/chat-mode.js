/**
 * 聊天模式模块 - AUE3架构实现
 */

// ==================== Effects 副作用层 ====================
const ChatEffects = {
    /**
     * 发送消息到服务器
     */
    async sendMessage(payload, state, app) {
        const { message, conversationId } = payload;
        
        // 显示分析中提示
        let statusMsg = null;
        try {
            statusMsg = document.createElement('div');
            statusMsg.className = 'system-status-message';
            statusMsg.innerHTML = `
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 12px; margin: 16px 0; text-align: center; animation: pulse 2s infinite;">
                    <i class="fas fa-brain fa-spin" style="font-size: 28px; margin-bottom: 8px;"></i>
                    <div style="font-size: 18px; font-weight: 600;">FIN-R1 正在分析中...</div>
                    <div style="font-size: 14px; opacity: 0.9; margin-top: 4px;">预计需要 15-30 秒，请稍候</div>
                </div>
            `;
            const messagesContainer = document.querySelector('.chat-messages') || document.querySelector('.messages-container');
            if (messagesContainer) {
                messagesContainer.appendChild(statusMsg);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
        } catch (e) {
            console.log('无法显示状态提示');
        }
        
        try {
            // 调用真实的后端API
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message,
                    conversation_id: conversationId || ''
                })
            });
            
            // 移除状态提示
            if (statusMsg) statusMsg.remove();
            
            if (!response.ok) {
                throw new Error(`API请求失败: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.status === 'success') {
                return {
                    success: true,
                    reply: result.response,
                    timestamp: new Date().toISOString()
                };
            } else {
                throw new Error(result.response || '分析失败');
            }
        } catch (error) {
            console.error('发送消息失败:', error);
            // 降级到模拟回复
            const fallbackResponses = [
                '抱歉，我暂时无法连接到分析服务。根据市场情况，建议您关注蓝筹股并保持谨慎。',
                '系统正在处理您的请求，请稍后再试。',
                '网络连接异常，请检查后端服务是否正常运行。'
            ];
            return {
                success: false,
                reply: fallbackResponses[0] + '\n\n错误详情: ' + error.message,
                timestamp: new Date().toISOString()
            };
        }
    },

    /**
     * 加载对话历史
     */
    async loadConversations(payload, state, app) {
        // 从本地存储加载
        const conversations = AUE3Utils.storage.get('chat_conversations', [
            {
                id: 'conv_1',
                title: '市场分析讨论',
                lastMessage: '您好，请问有什么可以帮助您的？',
                timestamp: new Date(Date.now() - 5 * 60 * 1000).toISOString(),
                active: true
            },
            {
                id: 'conv_2',
                title: '个股分析：贵州茅台',
                lastMessage: '茅台的估值处于历史中位水平',
                timestamp: new Date(Date.now() - 60 * 60 * 1000).toISOString(),
                active: false
            },
            {
                id: 'conv_3',
                title: '投资策略咨询',
                lastMessage: '建议采用价值投资策略',
                timestamp: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
                active: false
            }
        ]);
        
        return conversations;
    },

    /**
     * 保存对话
     */
    async saveConversation(payload, state, app) {
        const { conversations } = payload;
        AUE3Utils.storage.set('chat_conversations', conversations);
        return { success: true };
    }
};

// ==================== Actions 动作层 ====================
const ChatActions = {
    /**
     * 初始化聊天
     */
    async initialize(payload, state, app) {
        try {
            const conversations = await app.runEffect('loadConversations');
            
            return {
                state: {
                    conversations,
                    currentConversation: conversations.find(c => c.active) || conversations[0],
                    messages: [],
                    loading: false,
                    error: null
                },
                render: 'conversations'
            };
        } catch (error) {
            console.error('初始化失败:', error);
            return {
                state: {
                    loading: false,
                    error: error.message
                }
            };
        }
    },

    /**
     * 发送消息
     */
    async sendMessage(payload, state, app) {
        const { message } = payload;
        
        if (!message || !message.trim()) {
            Components.toast('请输入消息内容', 'warning');
            return;
        }

        // 添加用户消息到界面
        const userMessage = {
            id: AUE3Utils.uuid(),
            type: 'user',
            content: message.trim(),
            timestamp: new Date().toISOString()
        };

        const newMessages = [...state.messages, userMessage];

        // 更新状态并渲染
        app.setState({ 
            messages: newMessages,
            isTyping: true 
        });
        app.render('messages');

        try {
            // 发送到服务器
            const result = await app.runEffect('sendMessage', {
                message: message.trim(),
                conversationId: state.currentConversation?.id
            });

            // 添加AI回复
            const aiMessage = {
                id: AUE3Utils.uuid(),
                type: 'assistant',
                content: result.reply,
                timestamp: result.timestamp
            };

            return {
                state: {
                    messages: [...newMessages, aiMessage],
                    isTyping: false
                },
                render: 'messages'
            };
        } catch (error) {
            console.error('发送消息失败:', error);
            Components.toast('发送失败，请重试', 'error');
            
            return {
                state: {
                    isTyping: false,
                    error: error.message
                }
            };
        }
    },

    /**
     * 新建对话
     */
    newConversation(payload, state, app) {
        const newConv = {
            id: 'conv_' + Date.now(),
            title: '新对话',
            lastMessage: '',
            timestamp: new Date().toISOString(),
            active: true
        };

        const conversations = state.conversations.map(c => ({ ...c, active: false }));
        conversations.unshift(newConv);

        return {
            state: {
                conversations,
                currentConversation: newConv,
                messages: []
            },
            render: 'conversations'
        };
    },

    /**
     * 切换对话
     */
    switchConversation(payload, state, app) {
        const { conversationId } = payload;
        
        const conversations = state.conversations.map(c => ({
            ...c,
            active: c.id === conversationId
        }));

        const currentConversation = conversations.find(c => c.id === conversationId);

        return {
            state: {
                conversations,
                currentConversation,
                messages: [] // 实际应该加载该对话的历史消息
            },
            render: 'conversations'
        };
    },

    /**
     * 快速提问
     */
    quickAsk(payload, state, app) {
        const { question } = payload;
        return ChatActions.sendMessage({ message: question }, state, app);
    }
};

// ==================== UI 界面层 ====================
const ChatUI = {
    /**
     * 渲染对话列表
     */
    conversations(state, app) {
        const chatList = AUE3Utils.$('.chat-list');
        if (!chatList || !state.conversations) return;

        chatList.innerHTML = '';

        state.conversations.forEach(conv => {
            const timeAgo = this.formatTimeAgo(conv.timestamp);
            
            const item = AUE3Utils.createElement('div', {
                className: `chat-item ${conv.active ? 'active' : ''}`,
                onClick: () => app.dispatch('switchConversation', { conversationId: conv.id })
            }, [
                AUE3Utils.createElement('div', {
                    className: 'chat-item-icon'
                }, AUE3Utils.createElement('i', {
                    className: 'fas fa-comment-dots'
                })),
                AUE3Utils.createElement('div', {
                    className: 'chat-item-content'
                }, [
                    AUE3Utils.createElement('div', {
                        className: 'chat-item-title'
                    }, conv.title),
                    AUE3Utils.createElement('div', {
                        className: 'chat-item-time'
                    }, timeAgo)
                ])
            ]);

            chatList.appendChild(item);
        });
    },

    /**
     * 渲染消息列表
     */
    messages(state, app) {
        const messagesContainer = AUE3Utils.$('.messages-container');
        const welcomeScreen = AUE3Utils.$('#welcomeScreen');
        
        if (!messagesContainer) return;

        // 如果有消息，隐藏欢迎屏幕
        if (state.messages && state.messages.length > 0) {
            if (welcomeScreen) welcomeScreen.style.display = 'none';
            messagesContainer.style.display = 'block';
            messagesContainer.innerHTML = '';

            state.messages.forEach(msg => {
                const messageEl = this.createMessageElement(msg);
                messagesContainer.appendChild(messageEl);
            });

            // 显示输入中动画
            if (state.isTyping) {
                messagesContainer.appendChild(this.createTypingIndicator());
            }

            // 滚动到底部
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        } else {
            if (welcomeScreen) welcomeScreen.style.display = 'flex';
            messagesContainer.style.display = 'none';
        }
    },

    /**
     * 创建消息元素
     */
    createMessageElement(message) {
        const time = AUE3Utils.formatDate(message.timestamp, 'HH:mm');
        
        return AUE3Utils.createElement('div', {
            className: `message ${message.type}`,
            style: {
                display: 'flex',
                gap: '1rem',
                marginBottom: '1.5rem',
                animation: 'fadeInUp 0.3s ease'
            }
        }, [
            AUE3Utils.createElement('div', {
                className: 'message-avatar',
                style: {
                    width: '40px',
                    height: '40px',
                    borderRadius: '50%',
                    background: message.type === 'user' ? 
                        'linear-gradient(135deg, #3b82f6, #8b5cf6)' : 
                        'linear-gradient(135deg, #10b981, #06b6d4)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'white',
                    flexShrink: '0'
                }
            }, AUE3Utils.createElement('i', {
                className: `fas fa-${message.type === 'user' ? 'user' : 'robot'}`
            })),
            AUE3Utils.createElement('div', {
                className: 'message-content',
                style: {
                    flex: '1',
                    minWidth: '0'
                }
            }, [
                AUE3Utils.createElement('div', {
                    className: 'message-bubble',
                    style: {
                        padding: '1rem 1.25rem',
                        borderRadius: '12px',
                        background: message.type === 'user' ? '#3b82f6' : 'white',
                        color: message.type === 'user' ? 'white' : '#0f172a',
                        boxShadow: message.type === 'user' ? 
                            'none' : '0 2px 8px rgba(0,0,0,0.08)',
                        maxWidth: '80%',
                        wordWrap: 'break-word',
                        lineHeight: '1.6'
                    }
                }, message.content),
                AUE3Utils.createElement('div', {
                    className: 'message-time',
                    style: {
                        fontSize: '0.8rem',
                        color: '#94a3b8',
                        marginTop: '0.5rem'
                    }
                }, time)
            ])
        ]);
    },

    /**
     * 创建输入中指示器
     */
    createTypingIndicator() {
        return AUE3Utils.createElement('div', {
            className: 'message assistant',
            style: {
                display: 'flex',
                gap: '1rem',
                marginBottom: '1.5rem'
            }
        }, [
            AUE3Utils.createElement('div', {
                style: {
                    width: '40px',
                    height: '40px',
                    borderRadius: '50%',
                    background: 'linear-gradient(135deg, #10b981, #06b6d4)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'white'
                }
            }, AUE3Utils.createElement('i', {
                className: 'fas fa-robot'
            })),
            AUE3Utils.createElement('div', {
                style: {
                    padding: '1rem 1.25rem',
                    borderRadius: '12px',
                    background: 'white',
                    boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
                    display: 'flex',
                    gap: '0.5rem',
                    alignItems: 'center'
                }
            }, [
                AUE3Utils.createElement('div', {
                    style: {
                        width: '8px',
                        height: '8px',
                        borderRadius: '50%',
                        background: '#94a3b8',
                        animation: 'pulse 1.5s ease-in-out infinite'
                    }
                }),
                AUE3Utils.createElement('div', {
                    style: {
                        width: '8px',
                        height: '8px',
                        borderRadius: '50%',
                        background: '#94a3b8',
                        animation: 'pulse 1.5s ease-in-out 0.2s infinite'
                    }
                }),
                AUE3Utils.createElement('div', {
                    style: {
                        width: '8px',
                        height: '8px',
                        borderRadius: '50%',
                        background: '#94a3b8',
                        animation: 'pulse 1.5s ease-in-out 0.4s infinite'
                    }
                })
            ])
        ]);
    },

    /**
     * 格式化时间
     */
    formatTimeAgo(timestamp) {
        const now = new Date();
        const time = new Date(timestamp);
        const diff = now - time;
        const minutes = Math.floor(diff / 60000);
        const hours = Math.floor(diff / 3600000);
        const days = Math.floor(diff / 86400000);

        if (minutes < 1) return '刚刚';
        if (minutes < 60) return `${minutes}分钟前`;
        if (hours < 24) return `${hours}小时前`;
        if (days === 1) return '昨天';
        if (days < 7) return `${days}天前`;
        return AUE3Utils.formatDate(time, 'MM-DD');
    }
};

// ==================== 初始化应用 ====================
const chatApp = new AUE3Core();

// 注册各层
chatApp
    .registerEffects(ChatEffects)
    .registerActions(ChatActions)
    .registerUI(ChatUI);

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    // 初始化状态
    chatApp.init({
        conversations: [],
        currentConversation: null,
        messages: [],
        isTyping: false,
        loading: true,
        error: null
    });

    // 加载数据
    chatApp.dispatch('initialize');

    // 绑定发送按钮
    const sendButton = AUE3Utils.$('#sendButton');
    const messageInput = AUE3Utils.$('#messageInput');

    if (sendButton && messageInput) {
        sendButton.addEventListener('click', () => {
            const message = messageInput.value;
            if (message.trim()) {
                chatApp.dispatch('sendMessage', { message });
                messageInput.value = '';
                messageInput.style.height = 'auto';
            }
        });

        // 回车发送
        messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendButton.click();
            }
        });

        // 自动调整高度
        messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        });
    }

    // 新建对话按钮
    const newChatBtn = AUE3Utils.$('.btn-new-chat');
    if (newChatBtn) {
        newChatBtn.addEventListener('click', () => {
            chatApp.dispatch('newConversation');
        });
    }

    // 快速操作卡片
    AUE3Utils.$$('.action-card').forEach(card => {
        card.addEventListener('click', function() {
            const text = this.querySelector('span').textContent;
            const questions = {
                '市场分析': '请分析一下当前A股市场的整体走势',
                '个股研究': '能帮我分析一下贵州茅台这只股票吗？',
                '投资理念': '如何构建一个稳健的投资组合？',
                '组合优化': '我的投资组合应该如何优化配置？'
            };
            const question = questions[text];
            if (question) {
                chatApp.dispatch('quickAsk', { question });
            }
        });
    });
});

// 导出全局
window.chatApp = chatApp;
window.sendQuickMessage = (message) => {
    chatApp.dispatch('quickAsk', { question: message });
};

















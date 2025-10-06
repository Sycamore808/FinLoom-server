/**
 * 共享UI组件库
 */

const Components = {
    /**
     * 加载动画
     */
    loading: (text = '加载中...') => {
        return AUE3Utils.createElement('div', {
            className: 'loading-container',
            style: {
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                padding: '3rem',
                gap: '1rem'
            }
        }, [
            AUE3Utils.createElement('div', {
                className: 'spinner',
                style: {
                    width: '40px',
                    height: '40px',
                    border: '4px solid #e2e8f0',
                    borderTopColor: '#3b82f6',
                    borderRadius: '50%',
                    animation: 'spin 1s linear infinite'
                }
            }),
            AUE3Utils.createElement('div', {
                style: {
                    color: '#64748b',
                    fontSize: '1rem'
                }
            }, text)
        ]);
    },

    /**
     * Toast提示
     */
    toast: (message, type = 'info', duration = 3000) => {
        const colors = {
            success: '#10b981',
            error: '#ef4444',
            warning: '#f59e0b',
            info: '#3b82f6'
        };

        const toast = AUE3Utils.createElement('div', {
            className: 'toast',
            style: {
                position: 'fixed',
                top: '2rem',
                right: '2rem',
                background: 'white',
                padding: '1rem 1.5rem',
                borderRadius: '12px',
                boxShadow: '0 10px 40px rgba(0,0,0,0.15)',
                borderLeft: `4px solid ${colors[type]}`,
                zIndex: '10000',
                animation: 'slideInRight 0.3s ease',
                maxWidth: '400px',
                display: 'flex',
                alignItems: 'center',
                gap: '1rem'
            }
        }, [
            AUE3Utils.createElement('i', {
                className: `fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'times-circle' : type === 'warning' ? 'exclamation-triangle' : 'info-circle'}`,
                style: {
                    color: colors[type],
                    fontSize: '1.25rem'
                }
            }),
            AUE3Utils.createElement('span', {
                style: {
                    color: '#0f172a',
                    fontWeight: '600'
                }
            }, message)
        ]);

        document.body.appendChild(toast);

        setTimeout(() => {
            toast.style.animation = 'slideOutRight 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, duration);

        return toast;
    },

    /**
     * 确认对话框
     */
    confirm: (title, message, onConfirm, onCancel) => {
        const overlay = AUE3Utils.createElement('div', {
            className: 'modal-overlay',
            style: {
                position: 'fixed',
                top: '0',
                left: '0',
                right: '0',
                bottom: '0',
                background: 'rgba(0,0,0,0.5)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                zIndex: '10000',
                animation: 'fadeIn 0.3s ease'
            }
        });

        const modal = AUE3Utils.createElement('div', {
            className: 'modal',
            style: {
                background: 'white',
                borderRadius: '20px',
                padding: '2rem',
                maxWidth: '500px',
                width: '90%',
                boxShadow: '0 20px 60px rgba(0,0,0,0.2)',
                animation: 'scaleIn 0.3s ease'
            }
        }, [
            AUE3Utils.createElement('h3', {
                style: {
                    fontSize: '1.5rem',
                    fontWeight: '800',
                    color: '#0f172a',
                    marginBottom: '1rem'
                }
            }, title),
            AUE3Utils.createElement('p', {
                style: {
                    fontSize: '1rem',
                    color: '#64748b',
                    marginBottom: '2rem',
                    lineHeight: '1.6'
                }
            }, message),
            AUE3Utils.createElement('div', {
                style: {
                    display: 'flex',
                    gap: '1rem',
                    justifyContent: 'flex-end'
                }
            }, [
                AUE3Utils.createElement('button', {
                    className: 'btn-secondary',
                    style: {
                        padding: '0.75rem 1.5rem',
                        border: '2px solid #e2e8f0',
                        background: 'white',
                        borderRadius: '10px',
                        fontWeight: '600',
                        cursor: 'pointer',
                        transition: 'all 0.2s'
                    },
                    onClick: () => {
                        overlay.remove();
                        if (onCancel) onCancel();
                    }
                }, '取消'),
                AUE3Utils.createElement('button', {
                    className: 'btn-primary',
                    style: {
                        padding: '0.75rem 1.5rem',
                        border: 'none',
                        background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
                        color: 'white',
                        borderRadius: '10px',
                        fontWeight: '600',
                        cursor: 'pointer',
                        transition: 'all 0.2s'
                    },
                    onClick: () => {
                        overlay.remove();
                        if (onConfirm) onConfirm();
                    }
                }, '确认')
            ])
        ]);

        overlay.appendChild(modal);
        document.body.appendChild(overlay);

        return overlay;
    },

    /**
     * 模态框
     */
    modal: (content, options = {}) => {
        const {
            title = '',
            width = '600px',
            closable = true,
            onClose = null
        } = options;

        const overlay = AUE3Utils.createElement('div', {
            className: 'modal-overlay',
            style: {
                position: 'fixed',
                top: '0',
                left: '0',
                right: '0',
                bottom: '0',
                background: 'rgba(0,0,0,0.5)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                zIndex: '10000',
                animation: 'fadeIn 0.3s ease'
            },
            onClick: (e) => {
                if (closable && e.target === overlay) {
                    overlay.remove();
                    if (onClose) onClose();
                }
            }
        });

        const modalContent = AUE3Utils.createElement('div', {
            className: 'modal-content',
            style: {
                background: 'white',
                borderRadius: '20px',
                padding: '2rem',
                maxWidth: width,
                width: '90%',
                maxHeight: '90vh',
                overflow: 'auto',
                boxShadow: '0 20px 60px rgba(0,0,0,0.2)',
                animation: 'scaleIn 0.3s ease'
            }
        });

        if (title) {
            const header = AUE3Utils.createElement('div', {
                style: {
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: '1.5rem',
                    paddingBottom: '1rem',
                    borderBottom: '1px solid #e2e8f0'
                }
            }, [
                AUE3Utils.createElement('h3', {
                    style: {
                        fontSize: '1.5rem',
                        fontWeight: '800',
                        color: '#0f172a'
                    }
                }, title),
                closable ? AUE3Utils.createElement('button', {
                    style: {
                        border: 'none',
                        background: 'none',
                        fontSize: '1.5rem',
                        color: '#94a3b8',
                        cursor: 'pointer',
                        padding: '0.25rem',
                        lineHeight: '1'
                    },
                    onClick: () => {
                        overlay.remove();
                        if (onClose) onClose();
                    }
                }, '×') : null
            ]);
            modalContent.appendChild(header);
        }

        if (typeof content === 'string') {
            modalContent.appendChild(AUE3Utils.createElement('div', {}, content));
        } else {
            modalContent.appendChild(content);
        }

        overlay.appendChild(modalContent);
        document.body.appendChild(overlay);

        return {
            element: overlay,
            close: () => {
                overlay.remove();
                if (onClose) onClose();
            }
        };
    },

    /**
     * 空状态
     */
    empty: (message = '暂无数据', icon = 'inbox') => {
        return AUE3Utils.createElement('div', {
            className: 'empty-state',
            style: {
                textAlign: 'center',
                padding: '4rem 2rem',
                color: '#94a3b8'
            }
        }, [
            AUE3Utils.createElement('i', {
                className: `fas fa-${icon}`,
                style: {
                    fontSize: '4rem',
                    marginBottom: '1.5rem',
                    opacity: '0.5'
                }
            }),
            AUE3Utils.createElement('p', {
                style: {
                    fontSize: '1.1rem',
                    fontWeight: '600'
                }
            }, message)
        ]);
    },

    /**
     * 徽章
     */
    badge: (text, variant = 'default') => {
        const colors = {
            default: { bg: '#f1f5f9', color: '#64748b' },
            primary: { bg: 'rgba(59, 130, 246, 0.1)', color: '#3b82f6' },
            success: { bg: 'rgba(16, 185, 129, 0.1)', color: '#10b981' },
            warning: { bg: 'rgba(245, 158, 11, 0.1)', color: '#f59e0b' },
            danger: { bg: 'rgba(239, 68, 68, 0.1)', color: '#ef4444' }
        };

        const style = colors[variant] || colors.default;

        return AUE3Utils.createElement('span', {
            className: `badge badge-${variant}`,
            style: {
                padding: '0.25rem 0.75rem',
                borderRadius: '12px',
                fontSize: '0.85rem',
                fontWeight: '600',
                background: style.bg,
                color: style.color,
                display: 'inline-block'
            }
        }, text);
    },

    /**
     * 进度条
     */
    progressBar: (percentage, options = {}) => {
        const {
            height = '8px',
            color = 'linear-gradient(90deg, #3b82f6, #8b5cf6)',
            showLabel = false
        } = options;

        const container = AUE3Utils.createElement('div', {
            style: {
                width: '100%',
                height: height,
                background: '#e2e8f0',
                borderRadius: '10px',
                overflow: 'hidden',
                position: 'relative'
            }
        });

        const fill = AUE3Utils.createElement('div', {
            style: {
                width: `${Math.min(100, Math.max(0, percentage))}%`,
                height: '100%',
                background: color,
                borderRadius: '10px',
                transition: 'width 0.5s ease'
            }
        });

        container.appendChild(fill);

        if (showLabel) {
            const label = AUE3Utils.createElement('div', {
                style: {
                    textAlign: 'center',
                    marginTop: '0.5rem',
                    fontSize: '0.85rem',
                    fontWeight: '600',
                    color: '#64748b'
                }
            }, `${percentage}%`);
            
            const wrapper = AUE3Utils.createElement('div');
            wrapper.appendChild(container);
            wrapper.appendChild(label);
            return wrapper;
        }

        return container;
    }
};

// 添加必要的CSS动画
if (!document.getElementById('aue3-animations')) {
    const style = document.createElement('style');
    style.id = 'aue3-animations';
    style.textContent = `
        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes fadeOut {
            from { opacity: 1; }
            to { opacity: 0; }
        }
        @keyframes scaleIn {
            from { transform: scale(0.9); opacity: 0; }
            to { transform: scale(1); opacity: 1; }
        }
        @keyframes slideInRight {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        @keyframes slideOutRight {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
        }
    `;
    document.head.appendChild(style);
}

// 导出
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { Components };
} else {
    window.Components = Components;
}











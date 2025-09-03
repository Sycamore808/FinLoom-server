"""
响应生成器模块
负责生成系统响应
"""

import json
import random
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from common.logging_system import setup_logger
from common.exceptions import QuantSystemError

logger = setup_logger("response_generator")

@dataclass
class ResponseTemplate:
    """响应模板数据结构"""
    template_id: str
    intent: str
    template_text: str
    variables: List[str]
    tone: str  # 'formal', 'casual', 'friendly'
    context_requirements: List[str]

class ResponseGenerator:
    """响应生成器类"""
    
    # 响应模板库
    RESPONSE_TEMPLATES = {
        'greeting': [
            {
                'text': "您好！我是您的智能投资顾问，很高兴为您服务。请问有什么可以帮助您的吗？",
                'tone': 'friendly',
                'variables': []
            },
            {
                'text': "欢迎使用智能投资系统！我可以为您提供投资建议、市场分析和风险评估等服务。",
                'tone': 'formal',
                'variables': []
            }
        ],
        'investment_inquiry': [
            {
                'text': "根据您的风险偏好({risk_level})和投资期限({horizon})，我建议您考虑以下投资方案：{recommendations}",
                'tone': 'formal',
                'variables': ['risk_level', 'horizon', 'recommendations']
            },
            {
                'text': "基于当前市场状况，{investment_amount}元的投资建议如下：{strategy_description}",
                'tone': 'formal',
                'variables': ['investment_amount', 'strategy_description']
            }
        ],
        'performance_query': [
            {
                'text': "您的投资组合{time_period}的表现如下：\n收益率：{return_rate}\n最大回撤：{max_drawdown}\n夏普比率：{sharpe_ratio}",
                'tone': 'formal',
                'variables': ['time_period', 'return_rate', 'max_drawdown', 'sharpe_ratio']
            },
            {
                'text': "{time_period}您的投资{performance_summary}。详细数据：{detailed_metrics}",
                'tone': 'casual',
                'variables': ['time_period', 'performance_summary', 'detailed_metrics']
            }
        ],
        'risk_assessment': [
            {
                'text': "当前投资组合的风险评估：\n风险等级：{risk_level}\n主要风险因素：{risk_factors}\n建议措施：{recommendations}",
                'tone': 'formal',
                'variables': ['risk_level', 'risk_factors', 'recommendations']
            }
        ],
        'market_analysis': [
            {
                'text': "市场分析：{market_summary}\n关键指标：{key_indicators}\n投资建议：{investment_advice}",
                'tone': 'formal',
                'variables': ['market_summary', 'key_indicators', 'investment_advice']
            }
        ],
        'error': [
            {
                'text': "抱歉，我没有完全理解您的问题。您可以换一种方式描述吗？",
                'tone': 'friendly',
                'variables': []
            },
            {
                'text': "对不起，{error_message}。请稍后再试或联系客服。",
                'tone': 'formal',
                'variables': ['error_message']
            }
        ],
        'clarification': [
            {
                'text': "为了更好地帮助您，我需要了解：{clarification_questions}",
                'tone': 'friendly',
                'variables': ['clarification_questions']
            }
        ],
        'confirmation': [
            {
                'text': "请确认以下信息是否正确：\n{confirmation_details}\n如果正确，请回复"确认"。",
                'tone': 'formal',
                'variables': ['confirmation_details']
            }
        ],
        'goodbye': [
            {
                'text': "感谢您的使用！祝您投资顺利！如有需要随时联系我。",
                'tone': 'friendly',
                'variables': []
            },
            {
                'text': "再见！期待下次为您服务。",
                'tone': 'casual',
                'variables': []
            }
        ]
    }
    
    # 动态内容生成规则
    DYNAMIC_CONTENT = {
        'market_condition': {
            'bullish': '市场处于上升趋势，投资者情绪乐观',
            'bearish': '市场处于下跌趋势，建议谨慎操作',
            'sideways': '市场震荡整理，适合区间交易',
            'volatile': '市场波动较大，注意风险控制'
        },
        'risk_level': {
            'conservative': '保守型',
            'moderate': '稳健型',
            'aggressive': '激进型',
            'very_aggressive': '非常激进型'
        },
        'time_horizon': {
            'short_term': '短期（1年以内）',
            'medium_term': '中期（1-3年）',
            'long_term': '长期（3-5年）',
            'very_long_term': '超长期（5年以上）'
        }
    }
    
    def __init__(self, tone: str = 'formal'):
        """初始化响应生成器
        
        Args:
            tone: 默认语气
        """
        self.default_tone = tone
        self.context_memory = {}
        
    def generate_response(
        self,
        intent: str,
        entities: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        tone: Optional[str] = None
    ) -> str:
        """生成响应
        
        Args:
            intent: 意图
            entities: 实体信息
            context: 上下文
            tone: 语气（可选）
            
        Returns:
            生成的响应文本
        """
        tone = tone or self.default_tone
        
        # 获取模板
        templates = self.RESPONSE_TEMPLATES.get(intent, self.RESPONSE_TEMPLATES['error'])
        
        # 选择合适的模板
        suitable_templates = [t for t in templates if t.get('tone', 'formal') == tone]
        if not suitable_templates:
            suitable_templates = templates
        
        template = random.choice(suitable_templates)
        
        # 准备变量
        variables = self._prepare_variables(template['variables'], entities, context)
        
        # 生成响应
        try:
            response = template['text'].format(**variables)
        except KeyError as e:
            logger.error(f"Missing variable in template: {e}")
            response = self._generate_fallback_response(intent)
        
        # 后处理
        response = self._post_process_response(response, context)
        
        return response
    
    def generate_structured_response(
        self,
        intent: str,
        data: Dict[str, Any],
        format: str = 'text'
    ) -> str:
        """生成结构化响应
        
        Args:
            intent: 意图
            data: 数据
            format: 格式 ('text', 'markdown', 'html')
            
        Returns:
            格式化的响应
        """
        if format == 'markdown':
            return self._generate_markdown_response(intent, data)
        elif format == 'html':
            return self._generate_html_response(intent, data)
        else:
            return self._generate_text_response(intent, data)
    
    def generate_error_response(
        self,
        error_type: str,
        error_details: Optional[str] = None
    ) -> str:
        """生成错误响应
        
        Args:
            error_type: 错误类型
            error_details: 错误详情
            
        Returns:
            错误响应文本
        """
        error_messages = {
            'understanding': '抱歉，我没有理解您的意思。请您换一种方式描述。',
            'processing': '处理您的请求时出现了问题，请稍后再试。',
            'data_unavailable': '暂时无法获取相关数据，请稍后查询。',
            'permission': '您没有权限执行此操作。',
            'invalid_input': '您提供的信息格式不正确，请检查后重试。',
            'system_error': '系统出现异常，我们正在处理，请稍后再试。'
        }
        
        base_message = error_messages.get(error_type, '出现了未知错误。')
        
        if error_details:
            return f"{base_message} 详情：{error_details}"
        return base_message
    
    def generate_clarification_response(
        self,
        missing_info: List[str]
    ) -> str:
        """生成澄清请求响应
        
        Args:
            missing_info: 缺失信息列表
            
        Returns:
            澄清请求文本
        """
        clarification_templates = {
            'investment_amount': '请问您计划投资多少资金？',
            'risk_tolerance': '请问您的风险承受能力如何？（保守/稳健/激进）',
            'investment_horizon': '请问您的投资期限是多久？',
            'investment_goals': '请问您的主要投资目标是什么？',
            'preferred_assets': '请问您偏好哪些类型的资产？'
        }
        
        questions = []
        for info in missing_info:
            question = clarification_templates.get(info, f"请提供{info}的信息")
            questions.append(question)
        
        if len(questions) == 1:
            return questions[0]
        else:
            intro = "为了给您更准确的建议，我需要了解以下信息：\n"
            return intro + "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    
    def generate_confirmation_response(
        self,
        action: str,
        details: Dict[str, Any]
    ) -> str:
        """生成确认响应
        
        Args:
            action: 动作类型
            details: 详细信息
            
        Returns:
            确认响应文本
        """
        confirmation_intros = {
            'investment': '请确认您的投资方案：',
            'adjustment': '请确认调整方案：',
            'execution': '请确认执行以下操作：'
        }
        
        intro = confirmation_intros.get(action, '请确认以下信息：')
        
        # 格式化详情
        detail_lines = []
        for key, value in details.items():
            # 将key转换为中文
            key_mapping = {
                'amount': '投资金额',
                'strategy': '投资策略',
                'risk_level': '风险等级',
                'horizon': '投资期限',
                'expected_return': '预期收益'
            }
            display_key = key_mapping.get(key, key)
            detail_lines.append(f"• {display_key}：{value}")
        
        confirmation_text = f"{intro}\n" + "\n".join(detail_lines)
        confirmation_text += "\n\n如果信息正确，请回复"确认"；如需修改，请说明需要调整的内容。"
        
        return confirmation_text
    
    def generate_suggestion_response(
        self,
        suggestions: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """生成建议响应
        
        Args:
            suggestions: 建议列表
            context: 上下文
            
        Returns:
            建议响应文本
        """
        if not suggestions:
            return "暂时没有特别的建议。"
        
        intro = "基于您的情况，我有以下建议：\n"
        
        suggestion_texts = []
        for i, suggestion in enumerate(suggestions, 1):
            title = suggestion.get('title', f'建议{i}')
            description = suggestion.get('description', '')
            reason = suggestion.get('reason', '')
            
            text = f"{i}. **{title}**\n   {description}"
            if reason:
                text += f"\n   理由：{reason}"
            
            suggestion_texts.append(text)
        
        return intro + "\n\n".join(suggestion_texts)
    
    def _prepare_variables(
        self,
        required_vars: List[str],
        entities: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """准备模板变量
        
        Args:
            required_vars: 需要的变量列表
            entities: 实体信息
            context: 上下文
            
        Returns:
            变量字典
        """
        variables = {}
        
        for var in required_vars:
            # 先从实体中查找
            if var in entities:
                variables[var] = entities[var]
            # 再从上下文中查找
            elif context and var in context:
                variables[var] = context[var]
            # 生成默认值
            else:
                variables[var] = self._generate_default_value(var)
        
        return variables
    
    def _generate_default_value(self, var_name: str) -> str:
        """生成默认值
        
        Args:
            var_name: 变量名
            
        Returns:
            默认值
        """
        defaults = {
            'time_period': '最近一个月',
            'return_rate': '暂无数据',
            'risk_level': '中等',
            'market_summary': '市场表现正常',
            'recommendations': '请提供更多信息以获得个性化建议'
        }
        
        return defaults.get(var_name, f'[{var_name}]')
    
    def _post_process_response(
        self,
        response: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """后处理响应
        
        Args:
            response: 原始响应
            context: 上下文
            
        Returns:
            处理后的响应
        """
        # 添加个性化元素
        if context and context.get('user_name'):
            response = response.replace('您', f"{context['user_name']}您")
        
        # 添加时间信息
        if '{current_time}' in response:
            response = response.replace(
                '{current_time}',
                datetime.now().strftime('%Y-%m-%d %H:%M')
            )
        
        # 确保响应不会太长
        max_length = 500
        if len(response) > max_length:
            response = response[:max_length-3] + '...'
        
        return response
    
    def _generate_text_response(
        self,
        intent: str,
        data: Dict[str, Any]
    ) -> str:
        """生成纯文本响应
        
        Args:
            intent: 意图
            data: 数据
            
        Returns:
            文本响应
        """
        lines = []
        
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{key}:")
                for sub_key, sub_value in value.items():
                    lines.append(f"  {sub_key}: {sub_value}")
            elif isinstance(value, list):
                lines.append(f"{key}:")
                for item in value:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"{key}: {value}")
        
        return "\n".join(lines)
    
    def _generate_markdown_response(
        self,
        intent: str,
        data: Dict[str, Any]
    ) -> str:
        """生成Markdown格式响应
        
        Args:
            intent: 意图
            data: 数据
            
        Returns:
            Markdown响应
        """
        lines = []
        
        # 添加标题
        title = data.get('title', '查询结果')
        lines.append(f"# {title}\n")
        
        for key, value in data.items():
            if key == 'title':
                continue
                
            if isinstance(value, dict):
                lines.append(f"## {key}\n")
                for sub_key, sub_value in value.items():
                    lines.append(f"- **{sub_key}**: {sub_value}")
            elif isinstance(value, list):
                lines.append(f"## {key}\n")
                for item in value:
                    lines.append(f"- {item}")
            else:
                lines.append(f"**{key}**: {value}\n")
        
        return "\n".join(lines)
    
    def _generate_html_response(
        self,
        intent: str,
        data: Dict[str, Any]
    ) -> str:
        """生成HTML格式响应
        
        Args:
            intent: 意图
            data: 数据
            
        Returns:
            HTML响应
        """
        html_parts = ['<div class="response">']
        
        # 添加标题
        title = data.get('title', '查询结果')
        html_parts.append(f'<h2>{title}</h2>')
        
        for key, value in data.items():
            if key == 'title':
                continue
                
            if isinstance(value, dict):
                html_parts.append(f'<h3>{key}</h3>')
                html_parts.append('<ul>')
                for sub_key, sub_value in value.items():
                    html_parts.append(f'<li><strong>{sub_key}</strong>: {sub_value}</li>')
                html_parts.append('</ul>')
            elif isinstance(value, list):
                html_parts.append(f'<h3>{key}</h3>')
                html_parts.append('<ul>')
                for item in value:
                    html_parts.append(f'<li>{item}</li>')
                html_parts.append('</ul>')
            else:
                html_parts.append(f'<p><strong>{key}</strong>: {value}</p>')
        
        html_parts.append('</div>')
        
        return '\n'.join(html_parts)
    
    def _generate_fallback_response(self, intent: str) -> str:
        """生成后备响应
        
        Args:
            intent: 意图
            
        Returns:
            后备响应文本
        """
        fallbacks = {
            'investment_inquiry': '我正在为您准备投资建议，请稍候...',
            'performance_query': '正在查询您的投资表现...',
            'risk_assessment': '正在进行风险评估...',
            'market_analysis': '正在分析市场状况...',
            'default': '正在处理您的请求，请稍候...'
        }
        
        return fallbacks.get(intent, fallbacks['default'])

# 模块级别函数
def create_response_generator(
    tone: str = 'formal'
) -> ResponseGenerator:
    """创建响应生成器实例
    
    Args:
        tone: 默认语气
        
    Returns:
        响应生成器实例
    """
    return ResponseGenerator(tone=tone)

def generate_quick_response(
    intent: str,
    entities: Dict[str, Any] = None
) -> str:
    """生成快速响应的便捷函数
    
    Args:
        intent: 意图
        entities: 实体信息
        
    Returns:
        响应文本
    """
    generator = ResponseGenerator()
    return generator.generate_response(intent, entities or {})
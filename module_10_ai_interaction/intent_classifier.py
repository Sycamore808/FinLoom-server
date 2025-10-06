"""
意图分类器模块
负责识别用户输入的意图
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from common.logging_system import setup_logger

logger = setup_logger("intent_classifier")


class IntentClassifier:
    """意图分类器类"""

    # 意图定义和关键词映射
    INTENT_PATTERNS = {
        # 问候和告别
        "greeting": {
            "keywords": [
                "你好",
                "您好",
                "hi",
                "hello",
                "嗨",
                "早上好",
                "下午好",
                "晚上好",
            ],
            "patterns": [r"^(你好|您好|hi|hello)"],
            "priority": 10,
        },
        "goodbye": {
            "keywords": ["再见", "拜拜", "谢谢", "bye", "goodbye", "结束", "退出"],
            "patterns": [r"(再见|拜拜|bye|goodbye|谢谢.*再见)"],
            "priority": 10,
        },
        # 投资需求相关
        "create_strategy": {
            "keywords": [
                "策略",
                "投资方案",
                "帮我设计",
                "创建",
                "建立",
                "制定",
                "想要",
                "需要",
                "希望",
                "打算",
                "计划",
            ],
            "patterns": [
                r"(帮我|为我|给我).*(设计|制定|建立|创建).*(策略|方案)",
                r"我想.*(投资|买|购买)",
            ],
            "priority": 9,
        },
        "modify_strategy": {
            "keywords": ["修改", "调整", "改变", "更换", "换一个"],
            "patterns": [r"(修改|调整|改变).*(策略|方案|配置)"],
            "priority": 8,
        },
        # 查询相关
        "query_price": {
            "keywords": ["价格", "多少钱", "股价", "现价", "市价"],
            "patterns": [r"(价格|股价|市价).*(多少|几)"],
            "priority": 7,
        },
        "query_performance": {
            "keywords": ["收益", "业绩", "表现", "赚了多少", "亏了多少", "回报"],
            "patterns": [r"(收益|业绩|表现|回报).*(怎么样|如何|多少)"],
            "priority": 7,
        },
        "query_risk": {
            "keywords": ["风险", "安全", "稳定", "波动", "回撤", "风险评估"],
            "patterns": [r"(风险|安全|稳定).*(怎么样|如何|大不大)"],
            "priority": 7,
        },
        "query_holdings": {
            "keywords": ["持仓", "仓位", "持股", "组合", "资产配置"],
            "patterns": [r"(持仓|仓位|组合).*(什么|哪些|怎么样)"],
            "priority": 7,
        },
        # 操作相关
        "buy_signal": {
            "keywords": ["买", "买入", "建仓", "开仓", "加仓", "抄底"],
            "patterns": [r"(买入|建仓|开仓).*(什么|哪个|推荐)"],
            "priority": 8,
        },
        "sell_signal": {
            "keywords": ["卖", "卖出", "平仓", "清仓", "减仓", "止损", "止盈"],
            "patterns": [r"(卖出|平仓|清仓).*(什么|哪个|建议)"],
            "priority": 8,
        },
        # 分析相关
        "market_analysis": {
            "keywords": ["市场", "行情", "趋势", "分析", "走势", "大盘"],
            "patterns": [r"(市场|行情|大盘).*(怎么样|如何|分析)"],
            "priority": 7,
        },
        "stock_analysis": {
            "keywords": ["分析", "研究", "评价", "怎么样", "如何"],
            "patterns": [r"([\u4e00-\u9fa5]+|[0-9]{6}).*(怎么样|如何|值得|推荐)"],
            "priority": 6,
        },
        # 建议相关
        "ask_advice": {
            "keywords": ["建议", "意见", "看法", "怎么办", "该怎么", "应该"],
            "patterns": [r"(有什么|给点|给我).*(建议|意见)", r"(该怎么|应该).*"],
            "priority": 7,
        },
        # 回测和优化
        "backtest": {
            "keywords": ["回测", "历史表现", "过去", "测试"],
            "patterns": [r"(回测|测试).*(策略|方案)"],
            "priority": 7,
        },
        "optimize": {
            "keywords": ["优化", "改进", "提升", "增强"],
            "patterns": [r"(优化|改进|提升).*(策略|收益|性能)"],
            "priority": 7,
        },
        # 确认和修改
        "confirm": {
            "keywords": ["确认", "同意", "好的", "可以", "行", "就这样"],
            "patterns": [r"^(确认|同意|好的|可以|行)$"],
            "priority": 9,
        },
        "reject": {
            "keywords": ["不", "不要", "不行", "不可以", "取消"],
            "patterns": [r"^(不|不要|不行|取消)"],
            "priority": 9,
        },
        # 帮助和说明
        "help": {
            "keywords": ["帮助", "help", "怎么用", "如何使用", "功能"],
            "patterns": [r"(帮助|help|怎么用)"],
            "priority": 8,
        },
        "explain": {
            "keywords": ["什么是", "解释", "说明", "含义"],
            "patterns": [r"(什么是|解释|说明).+"],
            "priority": 6,
        },
    }

    # 实体提取规则
    ENTITY_PATTERNS = {
        "investment_amount": [
            r"(\d+(?:\.\d+)?)\s*(?:元|万元|万|亿|千|百)",
            r"￥\s*(\d+(?:\.\d+)?)",
        ],
        "risk_tolerance": [
            r"(保守|稳健|激进|进取|非常激进|低风险|中等风险|高风险)",
        ],
        "investment_horizon": [
            r"(\d+)\s*(?:年|个月|月|天)",
            r"(短期|中期|长期|超长期)",
        ],
        "stock_code": [
            r"([0-9]{6})",  # A股代码
            r"([A-Z]{2,5})",  # 美股代码
        ],
        "percentage": [
            r"(\d+(?:\.\d+)?)\s*%",
            r"百分之\s*(\d+(?:\.\d+)?)",
        ],
    }

    def __init__(self):
        """初始化意图分类器"""
        self.history: List[str] = []
        self.context_window = 3

    def classify(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, float, Dict[str, Any]]:
        """分类用户输入的意图

        Args:
            text: 用户输入文本
            context: 上下文信息

        Returns:
            (意图, 置信度, 提取的实体)
        """
        text = text.strip()

        # 提取实体
        entities = self.extract_entities(text)

        # 意图识别
        intent_scores = {}

        for intent_name, intent_config in self.INTENT_PATTERNS.items():
            score = self._calculate_intent_score(text, intent_config)
            if score > 0:
                intent_scores[intent_name] = score

        # 如果没有匹配的意图，返回默认
        if not intent_scores:
            return "unknown", 0.3, entities

        # 选择得分最高的意图
        best_intent = max(intent_scores, key=intent_scores.get)
        confidence = min(1.0, intent_scores[best_intent])

        # 更新历史
        self.history.append(best_intent)
        if len(self.history) > self.context_window:
            self.history.pop(0)

        logger.info(f"Classified intent: {best_intent} (confidence: {confidence:.2f})")

        return best_intent, confidence, entities

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """提取实体

        Args:
            text: 输入文本

        Returns:
            实体字典
        """
        entities = {}

        for entity_type, patterns in self.ENTITY_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    value = match.group(1) if match.groups() else match.group(0)
                    entities[entity_type] = self._normalize_entity(entity_type, value)
                    break

        return entities

    def get_intent_history(self) -> List[str]:
        """获取意图历史

        Returns:
            意图历史列表
        """
        return self.history.copy()

    def _calculate_intent_score(
        self, text: str, intent_config: Dict[str, Any]
    ) -> float:
        """计算意图分数

        Args:
            text: 输入文本
            intent_config: 意图配置

        Returns:
            意图分数
        """
        score = 0.0
        text_lower = text.lower()

        # 关键词匹配
        keywords = intent_config.get("keywords", [])
        keyword_matches = sum(1 for kw in keywords if kw in text_lower)
        if keyword_matches > 0:
            score += keyword_matches * 0.3

        # 正则模式匹配
        patterns = intent_config.get("patterns", [])
        for pattern in patterns:
            if re.search(pattern, text):
                score += 0.5
                break

        # 优先级加权
        priority = intent_config.get("priority", 5)
        score *= priority / 10

        return min(1.0, score)

    def _normalize_entity(self, entity_type: str, value: str) -> Any:
        """归一化实体值

        Args:
            entity_type: 实体类型
            value: 实体值

        Returns:
            归一化后的值
        """
        if entity_type == "investment_amount":
            # 提取数字
            num_match = re.search(r"(\d+(?:\.\d+)?)", value)
            if num_match:
                amount = float(num_match.group(1))
                # 转换单位
                if "万" in value:
                    amount *= 10000
                elif "亿" in value:
                    amount *= 100000000
                elif "千" in value:
                    amount *= 1000
                elif "百" in value:
                    amount *= 100
                return amount
            return value

        elif entity_type == "risk_tolerance":
            risk_map = {
                "保守": "conservative",
                "低风险": "conservative",
                "稳健": "moderate",
                "中等风险": "moderate",
                "激进": "aggressive",
                "进取": "aggressive",
                "高风险": "aggressive",
                "非常激进": "very_aggressive",
            }
            return risk_map.get(value, value)

        elif entity_type == "investment_horizon":
            horizon_map = {
                "短期": "short_term",
                "中期": "medium_term",
                "长期": "long_term",
                "超长期": "very_long_term",
            }
            if value in horizon_map:
                return horizon_map[value]

            # 转换具体时间
            time_match = re.search(r"(\d+)\s*(年|月|天)", value)
            if time_match:
                num = int(time_match.group(1))
                unit = time_match.group(2)

                if unit == "年":
                    if num < 1:
                        return "short_term"
                    elif num <= 3:
                        return "medium_term"
                    elif num <= 5:
                        return "long_term"
                    else:
                        return "very_long_term"
                elif unit == "月":
                    if num <= 12:
                        return "short_term"
                    elif num <= 36:
                        return "medium_term"
                    else:
                        return "long_term"

            return value

        elif entity_type == "percentage":
            num_match = re.search(r"(\d+(?:\.\d+)?)", value)
            if num_match:
                return float(num_match.group(1)) / 100
            return value

        else:
            return value


# 模块级别函数
def create_intent_classifier() -> IntentClassifier:
    """创建意图分类器实例

    Returns:
        意图分类器实例
    """
    return IntentClassifier()


def classify_user_intent(text: str) -> Tuple[str, float, Dict[str, Any]]:
    """分类用户意图的便捷函数

    Args:
        text: 用户输入文本

    Returns:
        (意图, 置信度, 实体)
    """
    classifier = IntentClassifier()
    return classifier.classify(text)

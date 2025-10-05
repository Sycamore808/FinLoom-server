"""
对话管理器模块
负责管理多轮对话的状态和流程
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from common.exceptions import QuantSystemError
from common.logging_system import setup_logger
from module_10_ai_interaction.conversation_history import (
    ConversationHistoryManager,
    ConversationRecord,
)
from module_10_ai_interaction.intent_classifier import IntentClassifier
from module_10_ai_interaction.response_generator import ResponseGenerator

logger = setup_logger("dialogue_manager")


class DialogueState(Enum):
    """对话状态枚举"""

    INIT = "init"  # 初始化
    GREETING = "greeting"  # 问候
    REQUIREMENT_GATHERING = "requirement_gathering"  # 需求收集
    CLARIFICATION = "clarification"  # 澄清
    RECOMMENDATION = "recommendation"  # 推荐
    CONFIRMATION = "confirmation"  # 确认
    EXECUTION = "execution"  # 执行
    FEEDBACK = "feedback"  # 反馈
    FAREWELL = "farewell"  # 告别


@dataclass
class DialogueContext:
    """对话上下文数据结构"""

    session_id: str
    user_id: str
    current_state: DialogueState
    state_history: List[DialogueState] = field(default_factory=list)
    collected_info: Dict[str, Any] = field(default_factory=dict)
    pending_clarifications: List[str] = field(default_factory=list)
    last_intent: Optional[str] = None
    turn_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DialogueManager:
    """对话管理器类"""

    # 状态转换规则
    STATE_TRANSITIONS = {
        DialogueState.INIT: [DialogueState.GREETING],
        DialogueState.GREETING: [DialogueState.REQUIREMENT_GATHERING],
        DialogueState.REQUIREMENT_GATHERING: [
            DialogueState.CLARIFICATION,
            DialogueState.RECOMMENDATION,
        ],
        DialogueState.CLARIFICATION: [
            DialogueState.REQUIREMENT_GATHERING,
            DialogueState.RECOMMENDATION,
        ],
        DialogueState.RECOMMENDATION: [
            DialogueState.CONFIRMATION,
            DialogueState.CLARIFICATION,
            DialogueState.REQUIREMENT_GATHERING,
        ],
        DialogueState.CONFIRMATION: [
            DialogueState.EXECUTION,
            DialogueState.RECOMMENDATION,
        ],
        DialogueState.EXECUTION: [DialogueState.FEEDBACK],
        DialogueState.FEEDBACK: [
            DialogueState.FAREWELL,
            DialogueState.REQUIREMENT_GATHERING,
        ],
        DialogueState.FAREWELL: [],
    }

    # 必需的信息字段
    REQUIRED_FIELDS = [
        "investment_amount",
        "risk_tolerance",
        "investment_horizon",
        "investment_goals",
    ]

    def __init__(
        self,
        history_manager: Optional[ConversationHistoryManager] = None,
        intent_classifier: Optional[IntentClassifier] = None,
        response_generator: Optional[ResponseGenerator] = None,
    ):
        """初始化对话管理器

        Args:
            history_manager: 对话历史管理器
            intent_classifier: 意图分类器
            response_generator: 响应生成器
        """
        self.history_manager = history_manager or ConversationHistoryManager()
        self.intent_classifier = intent_classifier or IntentClassifier()
        self.response_generator = response_generator or ResponseGenerator()

        # 活跃会话
        self.active_sessions: Dict[str, DialogueContext] = {}

    def start_conversation(self, user_id: str) -> DialogueContext:
        """启动新对话

        Args:
            user_id: 用户ID

        Returns:
            对话上下文
        """
        session_id = str(uuid.uuid4())

        context = DialogueContext(
            session_id=session_id,
            user_id=user_id,
            current_state=DialogueState.INIT,
        )

        # 转换到问候状态
        self._transition_state(context, DialogueState.GREETING)

        # 保存到活跃会话
        self.active_sessions[session_id] = context

        logger.info(f"Started conversation {session_id} for user {user_id}")

        return context

    def process_user_input(self, session_id: str, user_input: str) -> Dict[str, Any]:
        """处理用户输入

        Args:
            session_id: 会话ID
            user_input: 用户输入

        Returns:
            处理结果字典
        """
        # 获取会话上下文
        context = self.active_sessions.get(session_id)
        if not context:
            raise QuantSystemError(f"Session {session_id} not found")

        # 更新回合计数
        context.turn_count += 1
        context.updated_at = datetime.now()

        # 意图识别
        intent, confidence, entities = self.intent_classifier.classify(user_input)
        context.last_intent = intent

        # 更新收集的信息
        self._update_collected_info(context, entities)

        # 根据当前状态和意图处理
        response_data = self._handle_state_transition(
            context, intent, user_input, entities
        )

        # 生成响应
        response_text = self._generate_response(
            context, intent, entities, response_data
        )

        # 保存对话记录
        self._save_conversation_turn(
            context, user_input, response_text, intent, entities, confidence
        )

        # 检查是否需要澄清
        if context.pending_clarifications:
            clarification = self.response_generator.generate_clarification_response(
                context.pending_clarifications
            )
            response_text += "\n\n" + clarification

        return {
            "session_id": session_id,
            "response": response_text,
            "state": context.current_state.value,
            "intent": intent,
            "confidence": confidence,
            "turn_count": context.turn_count,
            "needs_clarification": len(context.pending_clarifications) > 0,
            "collected_info": context.collected_info,
        }

    def get_conversation_context(self, session_id: str) -> Optional[DialogueContext]:
        """获取对话上下文

        Args:
            session_id: 会话ID

        Returns:
            对话上下文
        """
        return self.active_sessions.get(session_id)

    def end_conversation(self, session_id: str) -> bool:
        """结束对话

        Args:
            session_id: 会话ID

        Returns:
            是否成功结束
        """
        context = self.active_sessions.get(session_id)
        if not context:
            return False

        # 转换到告别状态
        self._transition_state(context, DialogueState.FAREWELL)

        # 从活跃会话中移除
        del self.active_sessions[session_id]

        logger.info(f"Ended conversation {session_id}")

        return True

    def get_active_sessions_count(self) -> int:
        """获取活跃会话数量

        Returns:
            活跃会话数量
        """
        return len(self.active_sessions)

    def _transition_state(
        self, context: DialogueContext, new_state: DialogueState
    ) -> bool:
        """状态转换

        Args:
            context: 对话上下文
            new_state: 新状态

        Returns:
            是否成功转换
        """
        # 检查状态转换是否合法
        allowed_states = self.STATE_TRANSITIONS.get(context.current_state, [])

        if new_state not in allowed_states:
            logger.warning(
                f"Invalid state transition: {context.current_state} -> {new_state}"
            )
            return False

        # 保存历史状态
        context.state_history.append(context.current_state)

        # 更新状态
        context.current_state = new_state
        context.updated_at = datetime.now()

        logger.info(f"State transition: {context.state_history[-1]} -> {new_state}")

        return True

    def _update_collected_info(
        self, context: DialogueContext, entities: Dict[str, Any]
    ):
        """更新收集的信息

        Args:
            context: 对话上下文
            entities: 提取的实体
        """
        for key, value in entities.items():
            if value is not None:
                context.collected_info[key] = value

        # 检查是否有需要澄清的信息
        context.pending_clarifications = []
        for field in self.REQUIRED_FIELDS:
            if field not in context.collected_info:
                context.pending_clarifications.append(field)

    def _handle_state_transition(
        self,
        context: DialogueContext,
        intent: str,
        user_input: str,
        entities: Dict[str, Any],
    ) -> Dict[str, Any]:
        """处理状态转换

        Args:
            context: 对话上下文
            intent: 用户意图
            user_input: 用户输入
            entities: 提取的实体

        Returns:
            响应数据
        """
        current_state = context.current_state
        response_data = {}

        if current_state == DialogueState.GREETING:
            # 问候 -> 需求收集
            self._transition_state(context, DialogueState.REQUIREMENT_GATHERING)
            response_data["type"] = "greeting"

        elif current_state == DialogueState.REQUIREMENT_GATHERING:
            # 检查是否需要澄清
            if context.pending_clarifications:
                self._transition_state(context, DialogueState.CLARIFICATION)
                response_data["type"] = "clarification"
            else:
                # 进入推荐
                self._transition_state(context, DialogueState.RECOMMENDATION)
                response_data["type"] = "recommendation"

        elif current_state == DialogueState.CLARIFICATION:
            # 检查信息是否完整
            if not context.pending_clarifications:
                self._transition_state(context, DialogueState.RECOMMENDATION)
                response_data["type"] = "recommendation"
            else:
                response_data["type"] = "clarification"

        elif current_state == DialogueState.RECOMMENDATION:
            # 检查用户意图
            if intent == "confirm":
                self._transition_state(context, DialogueState.CONFIRMATION)
                response_data["type"] = "confirmation"
            elif intent == "modify":
                self._transition_state(context, DialogueState.REQUIREMENT_GATHERING)
                response_data["type"] = "modification"
            else:
                response_data["type"] = "recommendation"

        elif current_state == DialogueState.CONFIRMATION:
            if intent == "confirm":
                self._transition_state(context, DialogueState.EXECUTION)
                response_data["type"] = "execution"
            else:
                self._transition_state(context, DialogueState.RECOMMENDATION)
                response_data["type"] = "recommendation"

        elif current_state == DialogueState.EXECUTION:
            self._transition_state(context, DialogueState.FEEDBACK)
            response_data["type"] = "feedback"

        elif current_state == DialogueState.FEEDBACK:
            if intent == "goodbye":
                self._transition_state(context, DialogueState.FAREWELL)
                response_data["type"] = "farewell"
            else:
                self._transition_state(context, DialogueState.REQUIREMENT_GATHERING)
                response_data["type"] = "new_requirement"

        return response_data

    def _generate_response(
        self,
        context: DialogueContext,
        intent: str,
        entities: Dict[str, Any],
        response_data: Dict[str, Any],
    ) -> str:
        """生成响应

        Args:
            context: 对话上下文
            intent: 意图
            entities: 实体
            response_data: 响应数据

        Returns:
            响应文本
        """
        response_type = response_data.get("type", "default")

        if response_type == "greeting":
            return self.response_generator.generate_response("greeting", entities)
        elif response_type == "clarification":
            return self.response_generator.generate_clarification_response(
                context.pending_clarifications
            )
        elif response_type == "recommendation":
            return self._generate_recommendation_response(context, entities)
        elif response_type == "confirmation":
            return self.response_generator.generate_confirmation_response(
                "investment", context.collected_info
            )
        elif response_type == "execution":
            return "正在执行您的投资策略，请稍候..."
        elif response_type == "feedback":
            return "策略执行完成！您对这次服务满意吗？有什么建议或需要调整的地方吗？"
        elif response_type == "farewell":
            return self.response_generator.generate_response("goodbye", {})
        else:
            return self.response_generator.generate_response(intent, entities)

    def _generate_recommendation_response(
        self, context: DialogueContext, entities: Dict[str, Any]
    ) -> str:
        """生成推荐响应

        Args:
            context: 对话上下文
            entities: 实体

        Returns:
            推荐响应文本
        """
        info = context.collected_info

        response = f"根据您的投资需求：\n"
        response += f"• 投资金额：{info.get('investment_amount', '未指定')}\n"
        response += f"• 风险偏好：{info.get('risk_tolerance', '未指定')}\n"
        response += f"• 投资期限：{info.get('investment_horizon', '未指定')}\n"
        response += f"• 投资目标：{info.get('investment_goals', '未指定')}\n\n"
        response += "我为您推荐以下投资策略...\n"
        response += "(此处应调用推荐引擎生成具体推荐)\n\n"
        response += "您对这个方案满意吗？"

        return response

    def _save_conversation_turn(
        self,
        context: DialogueContext,
        user_input: str,
        system_response: str,
        intent: str,
        entities: Dict[str, Any],
        confidence: float,
    ):
        """保存对话回合

        Args:
            context: 对话上下文
            user_input: 用户输入
            system_response: 系统响应
            intent: 意图
            entities: 实体
            confidence: 置信度
        """
        record = ConversationRecord(
            session_id=context.session_id,
            user_id=context.user_id,
            turn_id=f"{context.session_id}_{context.turn_count}",
            timestamp=datetime.now(),
            user_input=user_input,
            system_response=system_response,
            intent=intent,
            entities=entities,
            confidence=confidence,
            context_state=context.current_state.value,
            metadata={
                "turn_count": context.turn_count,
                "collected_info": context.collected_info,
            },
        )

        self.history_manager.save_conversation_turn(record)


# 模块级别函数
def create_dialogue_manager() -> DialogueManager:
    """创建对话管理器实例

    Returns:
        对话管理器实例
    """
    return DialogueManager()

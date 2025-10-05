"""
模块间通信协议
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum

from .exceptions import QuantSystemError
from .logging_system import setup_logger

logger = setup_logger("communication_protocol")

class MessageType(Enum):
    """消息类型枚举"""
    DATA = "DATA"
    COMMAND = "COMMAND"
    QUERY = "QUERY"
    RESPONSE = "RESPONSE"
    HEARTBEAT = "HEARTBEAT"
    ERROR = "ERROR"

class MessagePriority(Enum):
    """消息优先级枚举"""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10

@dataclass
class InterModuleMessage:
    """模块间消息标准格式"""
    message_id: str
    timestamp: datetime
    source_module: str
    target_module: str
    message_type: MessageType
    priority: MessagePriority
    payload: Dict[str, Any]
    require_ack: bool = True
    timeout_ms: int = 5000
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['message_type'] = self.message_type.value
        data['priority'] = self.priority.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InterModuleMessage':
        """从字典创建消息对象"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['message_type'] = MessageType(data['message_type'])
        data['priority'] = MessagePriority(data['priority'])
        return cls(**data)

class MessageQueue:
    """消息队列类"""
    
    def __init__(self, max_size: int = 10000):
        """初始化消息队列
        
        Args:
            max_size: 最大队列大小
        """
        self.max_size = max_size
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self.subscribers: Dict[str, List[Callable]] = {}
        self.running = False
        
    async def put(self, message: InterModuleMessage) -> bool:
        """添加消息到队列
        
        Args:
            message: 消息对象
            
        Returns:
            是否添加成功
        """
        try:
            await self.queue.put(message)
            logger.debug(f"Message {message.message_id} added to queue")
            return True
        except asyncio.QueueFull:
            logger.error(f"Queue is full, message {message.message_id} dropped")
            return False
    
    async def get(self) -> InterModuleMessage:
        """从队列获取消息
        
        Returns:
            消息对象
        """
        return await self.queue.get()
    
    def subscribe(self, message_type: MessageType, callback: Callable):
        """订阅特定类型的消息
        
        Args:
            message_type: 消息类型
            callback: 回调函数
        """
        if message_type.value not in self.subscribers:
            self.subscribers[message_type.value] = []
        self.subscribers[message_type.value].append(callback)
        logger.info(f"Subscribed to {message_type.value} messages")
    
    def unsubscribe(self, message_type: MessageType, callback: Callable):
        """取消订阅
        
        Args:
            message_type: 消息类型
            callback: 回调函数
        """
        if message_type.value in self.subscribers:
            try:
                self.subscribers[message_type.value].remove(callback)
                logger.info(f"Unsubscribed from {message_type.value} messages")
            except ValueError:
                logger.warning(f"Callback not found in subscribers for {message_type.value}")
    
    async def start_processing(self):
        """开始处理消息"""
        self.running = True
        logger.info("Message queue processing started")
        
        while self.running:
            try:
                message = await self.get()
                await self._process_message(message)
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    async def stop_processing(self):
        """停止处理消息"""
        self.running = False
        logger.info("Message queue processing stopped")
    
    async def _process_message(self, message: InterModuleMessage):
        """处理消息
        
        Args:
            message: 消息对象
        """
        message_type = message.message_type.value
        
        if message_type in self.subscribers:
            for callback in self.subscribers[message_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(message)
                    else:
                        callback(message)
                except Exception as e:
                    logger.error(f"Error in message callback: {e}")

class ModuleCommunicator:
    """模块通信器类"""
    
    def __init__(self, module_name: str, message_queue: MessageQueue):
        """初始化模块通信器
        
        Args:
            module_name: 模块名称
            message_queue: 消息队列
        """
        self.module_name = module_name
        self.message_queue = message_queue
        self.pending_responses: Dict[str, asyncio.Future] = {}
        self.message_counter = 0
        
    def _generate_message_id(self) -> str:
        """生成消息ID"""
        self.message_counter += 1
        return f"{self.module_name}_{self.message_counter}_{int(datetime.now().timestamp() * 1000)}"
    
    async def send_message(
        self,
        target_module: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        require_ack: bool = True,
        timeout_ms: int = 5000
    ) -> Optional[InterModuleMessage]:
        """发送消息
        
        Args:
            target_module: 目标模块
            message_type: 消息类型
            payload: 消息载荷
            priority: 优先级
            require_ack: 是否需要确认
            timeout_ms: 超时时间（毫秒）
            
        Returns:
            响应消息（如果需要确认）
        """
        message_id = self._generate_message_id()
        
        message = InterModuleMessage(
            message_id=message_id,
            timestamp=datetime.now(),
            source_module=self.module_name,
            target_module=target_module,
            message_type=message_type,
            priority=priority,
            payload=payload,
            require_ack=require_ack,
            timeout_ms=timeout_ms
        )
        
        # 添加到队列
        success = await self.message_queue.put(message)
        if not success:
            raise QuantSystemError(f"Failed to send message {message_id}")
        
        logger.info(f"Message {message_id} sent to {target_module}")
        
        # 如果需要确认，等待响应
        if require_ack:
            return await self._wait_for_response(message_id, timeout_ms)
        
        return None
    
    async def send_data(
        self,
        target_module: str,
        data: Any,
        data_type: str = "market_data",
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> Optional[InterModuleMessage]:
        """发送数据消息
        
        Args:
            target_module: 目标模块
            data: 数据
            data_type: 数据类型
            priority: 优先级
            
        Returns:
            响应消息
        """
        payload = {
            "data_type": data_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        return await self.send_message(
            target_module=target_module,
            message_type=MessageType.DATA,
            payload=payload,
            priority=priority
        )
    
    async def send_command(
        self,
        target_module: str,
        command: str,
        parameters: Dict[str, Any] = None,
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> Optional[InterModuleMessage]:
        """发送命令消息
        
        Args:
            target_module: 目标模块
            command: 命令
            parameters: 参数
            priority: 优先级
            
        Returns:
            响应消息
        """
        payload = {
            "command": command,
            "parameters": parameters or {},
            "timestamp": datetime.now().isoformat()
        }
        
        return await self.send_message(
            target_module=target_module,
            message_type=MessageType.COMMAND,
            payload=payload,
            priority=priority
        )
    
    async def send_query(
        self,
        target_module: str,
        query: str,
        parameters: Dict[str, Any] = None,
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> Optional[InterModuleMessage]:
        """发送查询消息
        
        Args:
            target_module: 目标模块
            query: 查询
            parameters: 参数
            priority: 优先级
            
        Returns:
            响应消息
        """
        payload = {
            "query": query,
            "parameters": parameters or {},
            "timestamp": datetime.now().isoformat()
        }
        
        return await self.send_message(
            target_module=target_module,
            message_type=MessageType.QUERY,
            payload=payload,
            priority=priority
        )
    
    async def _wait_for_response(self, message_id: str, timeout_ms: int) -> Optional[InterModuleMessage]:
        """等待响应消息
        
        Args:
            message_id: 消息ID
            timeout_ms: 超时时间
            
        Returns:
            响应消息
        """
        future = asyncio.Future()
        self.pending_responses[message_id] = future
        
        try:
            response = await asyncio.wait_for(future, timeout=timeout_ms / 1000)
            return response
        except asyncio.TimeoutError:
            logger.error(f"Timeout waiting for response to message {message_id}")
            return None
        finally:
            self.pending_responses.pop(message_id, None)
    
    def handle_response(self, message: InterModuleMessage):
        """处理响应消息
        
        Args:
            message: 响应消息
        """
        correlation_id = message.correlation_id
        if correlation_id and correlation_id in self.pending_responses:
            future = self.pending_responses[correlation_id]
            if not future.done():
                future.set_result(message)
    
    def subscribe_to_messages(self, message_type: MessageType, callback: Callable):
        """订阅消息
        
        Args:
            message_type: 消息类型
            callback: 回调函数
        """
        self.message_queue.subscribe(message_type, callback)

# 全局消息队列实例
_global_message_queue: Optional[MessageQueue] = None

def create_message_queue(max_size: int = 10000) -> MessageQueue:
    """创建消息队列
    
    Args:
        max_size: 最大队列大小
        
    Returns:
        消息队列实例
    """
    global _global_message_queue
    if _global_message_queue is None:
        _global_message_queue = MessageQueue(max_size)
    return _global_message_queue

def create_module_communicator(module_name: str) -> ModuleCommunicator:
    """创建模块通信器
    
    Args:
        module_name: 模块名称
        
    Returns:
        模块通信器实例
    """
    message_queue = create_message_queue()
    return ModuleCommunicator(module_name, message_queue)

async def send_data_message(
    source_module: str,
    target_module: str,
    data: Any,
    data_type: str = "market_data"
) -> Optional[InterModuleMessage]:
    """发送数据消息的便捷函数
    
    Args:
        source_module: 源模块
        target_module: 目标模块
        data: 数据
        data_type: 数据类型
        
    Returns:
        响应消息
    """
    communicator = create_module_communicator(source_module)
    return await communicator.send_data(target_module, data, data_type)
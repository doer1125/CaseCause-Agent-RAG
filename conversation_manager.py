import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from langchain_core.documents import Document
from reasoning_trace import trace_manager

class Message:
    def __init__(self, role: str, content: str):
        self.role = role  # "user" or "assistant"
        self.content = content
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        message = cls(role=data["role"], content=data["content"])
        message.timestamp = datetime.fromisoformat(data["timestamp"])
        return message

class Conversation:
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.messages: List[Message] = []
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        # 对话状态管理
        self.dialogue_state = {
            "current_task": "initial",  # initial, complaint_description, points_extraction, points_confirmation, case_retrieval, legal_retrieval, document_generation
            "complaint_points": None,   # 提取的投诉要点
            "retrieved_cases": None,    # 检索到的案由
            "retrieved_legal": None,    # 检索到的法律法规
            "document_type": None,      # 要生成的文书类型
            "document_params": None,    # 文书生成参数
            "confirmed": False          # 投诉要点是否已确认
        }
        # 对话意图历史
        self.intent_history = []

    def add_message(self, role: str, content: str) -> Message:
        message = Message(role=role, content=content)
        self.messages.append(message)
        self.updated_at = datetime.now()
        return message

    def get_history(self, max_rounds: Optional[int] = None) -> List[Message]:
        if max_rounds:
            # 按轮次计算，每两轮(用户+助手)为一个对话轮次
            start_index = max(0, len(self.messages) - 2 * max_rounds)
            return self.messages[start_index:]
        return self.messages

    def update_dialogue_state(self, updates: Dict[str, Any]):
        """
        更新对话状态
        
        Args:
            updates: 要更新的对话状态字段
        """
        self.dialogue_state.update(updates)
        self.updated_at = datetime.now()
    
    def set_current_task(self, task: str):
        """
        设置当前对话任务
        
        Args:
            task: 任务类型字符串
        """
        self.dialogue_state["current_task"] = task
        self.updated_at = datetime.now()
    
    def set_complaint_points(self, points: Dict[str, Any]):
        """
        设置投诉要点
        
        Args:
            points: 投诉要点字典
        """
        self.dialogue_state["complaint_points"] = points
        self.set_current_task("points_extraction")
    
    def confirm_complaint_points(self, confirmed: bool = True):
        """
        确认投诉要点
        
        Args:
            confirmed: 是否确认
        """
        self.dialogue_state["confirmed"] = confirmed
        if confirmed:
            self.set_current_task("case_retrieval")
    
    def add_intent(self, intent: str):
        """
        添加对话意图到历史记录
        
        Args:
            intent: 意图类型字符串
        """
        self.intent_history.append({
            "intent": intent,
            "timestamp": datetime.now()
        })
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "dialogue_state": self.dialogue_state,
            "intent_history": [
                {
                    "intent": item["intent"],
                    "timestamp": item["timestamp"].isoformat()
                } for item in self.intent_history
            ],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        conversation = cls(session_id=data["session_id"])
        conversation.messages = [Message.from_dict(msg) for msg in data["messages"]]
        conversation.dialogue_state = data.get("dialogue_state", {
            "current_task": "initial",
            "complaint_points": None,
            "retrieved_cases": None,
            "retrieved_legal": None,
            "document_type": None,
            "document_params": None,
            "confirmed": False
        })
        conversation.intent_history = [
            {
                "intent": item["intent"],
                "timestamp": datetime.fromisoformat(item["timestamp"])
            } for item in data.get("intent_history", [])
        ]
        conversation.created_at = datetime.fromisoformat(data["created_at"])
        conversation.updated_at = datetime.fromisoformat(data["updated_at"])
        return conversation

class ConversationManager:
    def __init__(self, max_conversations: int = 100, max_history_rounds: int = 10):
        self.conversations: Dict[str, Conversation] = {}
        self.max_conversations = max_conversations
        self.max_history_rounds = max_history_rounds

    def create_conversation(self) -> Conversation:
        """创建新的对话会话"""
        # 清理过期会话
        self._cleanup_old_conversations()
        
        # 如果超过最大会话数，清理最旧的会话
        if len(self.conversations) >= self.max_conversations:
            oldest_session_id = min(
                self.conversations.keys(),
                key=lambda k: self.conversations[k].updated_at
            )
            del self.conversations[oldest_session_id]
        
        conversation = Conversation()
        self.conversations[conversation.session_id] = conversation
        return conversation

    def get_conversation(self, session_id: str) -> Optional[Conversation]:
        """获取指定会话"""
        return self.conversations.get(session_id)

    def delete_conversation(self, session_id: str) -> bool:
        """删除会话"""
        if session_id in self.conversations:
            del self.conversations[session_id]
            return True
        return False

    def add_message(self, session_id: str, role: str, content: str) -> Optional[Message]:
        """添加消息到会话"""
        conversation = self.get_conversation(session_id)
        if conversation:
            return conversation.add_message(role, content)
        return None

    def _cleanup_old_conversations(self, inactive_time: timedelta = timedelta(hours=24)):
        """清理长时间不活跃的会话"""
        current_time = datetime.now()
        expired_sessions = [
            session_id for session_id, conv in self.conversations.items()
            if current_time - conv.updated_at > inactive_time
        ]
        
        for session_id in expired_sessions:
            del self.conversations[session_id]

class ContextFuser:
    def __init__(self, max_context_length: int = 2000, compression_strategy: str = "hybrid", summary_model=None):
        self.max_context_length = max_context_length
        self.compression_strategy = compression_strategy
        self.summary_model = summary_model
        # 重要性权重配置
        self.role_weights = {
            "user": 1.0,
            "assistant": 0.8
        }
        # 轮次衰减权重（越新的消息权重越高）
        self.round_decay = 0.9
        
    def _calculate_importance(self, msg: Message, index: int, total_messages: int) -> float:
        """计算消息的重要性分数"""
        # 角色权重
        role_weight = self.role_weights.get(msg.role, 0.5)
        # 位置权重（越新的消息权重越高）
        position_weight = (self.round_decay ** (total_messages - index - 1))
        # 内容长度权重（内容越长，可能越重要）
        length_weight = min(1.0, len(msg.content) / 100)
        # 综合权重
        importance = role_weight * position_weight * (1 + length_weight)
        return importance
        
    def _fuse_with_importance(self, query: str, history: List[Message], session_id: Optional[str] = None) -> str:
        """基于重要性的上下文压缩"""
        if session_id:
            trace_manager.add_step(
                session_id=session_id,
                step_type="context_fusion",
                description="使用基于重要性的上下文压缩",
                data={"history_length": len(history)}
            )
        
        if not history:
            return f"当前查询：\n{query}\n"
        
        # 计算每条消息的重要性
        total_messages = len(history)
        messages_with_importance = [
            (msg, self._calculate_importance(msg, i, total_messages))
            for i, msg in enumerate(history)
        ]
        
        if session_id:
            trace_manager.add_step(
                session_id=session_id,
                step_type="context_fusion",
                description="计算消息重要性完成",
                data={
                    "message_importance_scores": [
                        {"role": msg.role, "score": importance, "length": len(msg.content)}
                        for msg, importance in messages_with_importance
                    ]
                }
            )
        
        # 按重要性排序（从高到低）
        sorted_messages = sorted(
            messages_with_importance,
            key=lambda x: x[1],
            reverse=True
        )
        
        # 选择重要性高的消息，直到达到长度限制
        compressed_history = []
        temp_length = len(f"当前查询：\n{query}\n")
        
        for msg, importance in sorted_messages:
            msg_length = len(f"{msg.role}: {msg.content}\n")
            if temp_length + msg_length <= self.max_context_length:
                compressed_history.append(msg)
                temp_length += msg_length
            else:
                break
        
        if session_id:
            trace_manager.add_step(
                session_id=session_id,
                step_type="context_fusion",
                description="基于重要性选择消息完成",
                data={
                    "selected_messages_count": len(compressed_history),
                    "total_messages": total_messages,
                    "compressed_length": temp_length
                }
            )
        
        # 按原始顺序排序
        compressed_history.sort(key=lambda msg: history.index(msg))
        
        # 构建压缩上下文
        compressed_history_str = ""
        for msg in compressed_history:
            compressed_history_str += f"{msg.role}: {msg.content}\n"
        
        fused_context = f"""对话历史：
{compressed_history_str}
当前查询：
{query}
"""
        
        return fused_context
        
    def _fuse_with_summary(self, query: str, history: List[Message], session_id: Optional[str] = None) -> str:
        """基于摘要的上下文压缩"""
        if session_id:
            trace_manager.add_step(
                session_id=session_id,
                step_type="context_fusion",
                description="使用基于摘要的上下文压缩",
                data={"history_length": len(history)}
            )
        
        if not history:
            return f"当前查询：\n{query}\n"
        
        # 构建完整对话历史
        full_history = ""
        for msg in history:
            full_history += f"{msg.role}: {msg.content}\n"
        
        # 生成对话摘要
        summary = self._generate_summary(full_history)
        
        if session_id:
            trace_manager.add_step(
                session_id=session_id,
                step_type="context_fusion",
                description="生成对话摘要完成",
                data={
                    "original_history_length": len(full_history),
                    "summary_length": len(summary),
                    "summary": summary
                }
            )
        
        # 构建压缩上下文
        fused_context = f"""对话摘要：
{summary}
当前查询：
{query}
"""
        
        # 如果仍然过长，进一步压缩
        if len(fused_context) > self.max_context_length:
            # 只保留摘要的核心部分
            summary = summary[:self.max_context_length - len(f"当前查询：\n{query}\n") - len("对话摘要：\n\n")]
            fused_context = f"""对话摘要：
{summary}...
当前查询：
{query}
"""
        
        return fused_context
        
    def _generate_summary(self, text: str) -> str:
        """生成文本摘要"""
        # 如果没有提供摘要模型，使用简单的摘要方法
        if not self.summary_model:
            # 简单摘要：取文本的前200个字符
            return text[:200] + "..." if len(text) > 200 else text
        
        # 使用提供的摘要模型生成摘要
        # 这里需要根据具体的模型实现摘要生成逻辑
        try:
            summary = self.summary_model.generate(text)
            return summary
        except Exception as e:
            # 模型调用失败，使用简单摘要
            return text[:200] + "..." if len(text) > 200 else text
        
    def _fuse_with_semantic(self, query: str, history: List[Message], session_id: Optional[str] = None) -> str:
        """基于语义相关性的上下文压缩"""
        if session_id:
            trace_manager.add_step(
                session_id=session_id,
                step_type="context_fusion",
                description="使用基于语义相关性的上下文压缩",
                data={"history_length": len(history)}
            )
        
        if not history:
            return f"当前查询：\n{query}\n"
        
        # 简单实现：保留最新的N轮对话，同时保留与当前查询最相关的早期对话
        # 这里可以结合语义相似度计算，选择与当前查询最相关的消息
        
        # 首先保留最新的2轮对话
        recent_messages = history[-4:]  # 2轮对话，每轮包括用户和助手消息
        
        # 计算早期消息与当前查询的相关性（这里使用简单的关键词匹配）
        query_keywords = set(query.split())
        relevant_early_messages = []
        
        for msg in history[:-4]:
            msg_keywords = set(msg.content.split())
            # 计算关键词重叠率
            overlap = len(query_keywords.intersection(msg_keywords)) / len(query_keywords) if query_keywords else 0
            if overlap > 0.2:  # 关键词重叠率大于20%，认为相关
                relevant_early_messages.append(msg)
        
        if session_id:
            trace_manager.add_step(
                session_id=session_id,
                step_type="context_fusion",
                description="语义相关性分析完成",
                data={
                    "recent_messages_count": len(recent_messages),
                    "relevant_early_messages_count": len(relevant_early_messages),
                    "query_keywords": list(query_keywords)
                }
            )
        
        # 合并相关消息和最新消息
        compressed_history = relevant_early_messages + recent_messages
        
        # 构建压缩上下文
        compressed_history_str = ""
        for msg in compressed_history:
            compressed_history_str += f"{msg.role}: {msg.content}\n"
        
        fused_context = f"""对话历史：
{compressed_history_str}
当前查询：
{query}\n"""
        
        # 如果仍然过长，进行进一步压缩
        if len(fused_context) > self.max_context_length:
            # 只保留最新的对话轮次
            return self._fuse_with_original(query, history, session_id)
        
        return fused_context
        
    def _fuse_with_hybrid(self, query: str, history: List[Message], session_id: Optional[str] = None) -> str:
        """混合上下文压缩策略"""
        if session_id:
            trace_manager.add_step(
                session_id=session_id,
                step_type="context_fusion",
                description="使用混合上下文压缩策略",
                data={"history_length": len(history)}
            )
        
        if not history:
            return f"当前查询：\n{query}\n"
        
        # 混合策略：结合摘要和最新消息
        # 1. 生成对话历史的摘要
        full_history = ""
        for msg in history[:-2]:  # 排除最新的1轮对话
            full_history += f"{msg.role}: {msg.content}\n"
        
        summary = self._generate_summary(full_history)
        
        if session_id:
            trace_manager.add_step(
                session_id=session_id,
                step_type="context_fusion",
                description="生成历史对话摘要完成",
                data={
                    "history_to_summarize_length": len(full_history),
                    "summary_length": len(summary)
                }
            )
        
        # 2. 保留最新的1轮对话
        recent_messages = history[-2:] if len(history) >= 2 else history
        
        # 构建压缩上下文
        recent_messages_str = ""
        for msg in recent_messages:
            recent_messages_str += f"{msg.role}: {msg.content}\n"
        
        fused_context = f"""对话摘要：
{summary}
最新对话：
{recent_messages_str}
当前查询：
{query}\n"""
        
        # 如果仍然过长，使用原始压缩方法
        if len(fused_context) > self.max_context_length:
            return self._fuse_with_original(query, history, session_id)
        
        return fused_context
        
    def _fuse_with_original(self, query: str, history: List[Message], session_id: Optional[str] = None) -> str:
        """原始的上下文压缩方法"""
        if session_id:
            trace_manager.add_step(
                session_id=session_id,
                step_type="context_fusion",
                description="使用原始上下文压缩方法",
                data={"history_length": len(history)}
            )
        
        # 构建对话历史字符串
        history_str = ""
        for msg in history:
            history_str += f"{msg.role}: {msg.content}\n"
        
        # 融合查询和历史
        fused_context = f"""对话历史：
{history_str}
当前查询：
{query}
"""
        
        # 如果上下文过长，进行压缩
        if len(fused_context) > self.max_context_length:
            # 保留最新的对话轮次，直到上下文长度合适
            compressed_history = []
            temp_length = len(f"当前查询：\n{query}\n")
            
            for msg in reversed(history):
                msg_length = len(f"{msg.role}: {msg.content}\n")
                if temp_length + msg_length <= self.max_context_length:
                    compressed_history.insert(0, msg)
                    temp_length += msg_length
                else:
                    break
            
            if session_id:
                trace_manager.add_step(
                    session_id=session_id,
                    step_type="context_fusion",
                    description="上下文压缩完成",
                    data={
                        "original_length": len(fused_context),
                        "compressed_length": temp_length,
                        "retained_messages_count": len(compressed_history),
                        "total_messages": len(history)
                    }
                )
            
            # 重新构建压缩后的上下文
            compressed_history_str = ""
            for msg in compressed_history:
                compressed_history_str += f"{msg.role}: {msg.content}\n"
            
            fused_context = f"""对话历史：
{compressed_history_str}
当前查询：
{query}
"""
        
        return fused_context
        
    def fuse_context(self, query: str, history: List[Message], session_id: Optional[str] = None, dialogue_state: Optional[Dict[str, Any]] = None) -> str:
        """
        融合对话历史和当前查询
        
        Args:
            query: 当前查询文本
            history: 对话历史消息列表
            session_id: 会话ID，用于记录推理轨迹
            dialogue_state: 当前对话状态，用于动态调整融合策略
            
        Returns:
            融合后的查询文本
        """
        # 记录上下文融合开始
        if session_id:
            trace_manager.add_step(
                session_id=session_id,
                step_type="context_fusion",
                description="开始上下文融合",
                data={
                    "input_query": query,
                    "history_length": len(history),
                    "compression_strategy": self.compression_strategy,
                    "dialogue_state": dialogue_state
                }
            )
        
        # 根据对话状态动态选择压缩策略
        strategy = self.compression_strategy
        if dialogue_state:
            current_task = dialogue_state.get("current_task", "initial")
            
            # 根据当前任务调整压缩策略
            if current_task in ["initial", "complaint_description"]:
                # 初始阶段和投诉描述阶段，保留更多上下文
                strategy = "hybrid"
            elif current_task in ["points_extraction", "points_confirmation"]:
                # 要点提取和确认阶段，重点关注投诉描述和提取结果
                strategy = "importance"
            elif current_task in ["case_retrieval", "legal_retrieval"]:
                # 检索阶段，重点关注已确认的投诉要点
                strategy = "semantic"
            elif current_task == "document_generation":
                # 文书生成阶段，需要完整的上下文信息
                strategy = "hybrid"
        
        # 根据选择的策略进行上下文融合
        if strategy == "importance":
            fused_query = self._fuse_with_importance(query, history, session_id)
        elif strategy == "summary":
            fused_query = self._fuse_with_summary(query, history, session_id)
        elif strategy == "semantic":
            fused_query = self._fuse_with_semantic(query, history, session_id)
        elif strategy == "hybrid":
            fused_query = self._fuse_with_hybrid(query, history, session_id)
        else:
            # 默认使用原始压缩方法
            fused_query = self._fuse_with_original(query, history, session_id)
        
        # 如果对话状态中有已确认的投诉要点，将其加入融合查询
        if dialogue_state and dialogue_state.get("confirmed") and dialogue_state.get("complaint_points"):
            complaint_points = dialogue_state["complaint_points"]
            points_text = "已确认的投诉要点：\n"
            
            # 添加投诉类型
            if complaint_points.get("complaint_type"):
                points_text += f"- 投诉类型：{', '.join(complaint_points['complaint_type'])}\n"
            
            # 添加具体问题
            if complaint_points.get("issues"):
                points_text += f"- 具体问题：{', '.join(complaint_points['issues'])}\n"
            
            # 添加涉及的机构和人员
            if complaint_points.get("institutions"):
                points_text += f"- 涉及机构：{', '.join(complaint_points['institutions'])}\n"
            if complaint_points.get("persons"):
                points_text += f"- 涉及人员：{', '.join(complaint_points['persons'])}\n"
            
            # 将投诉要点添加到融合查询
            fused_query = points_text + "\n" + fused_query
        
        # 记录上下文融合结果
        if session_id:
            trace_manager.add_step(
                session_id=session_id,
                step_type="context_fusion",
                description="上下文融合完成",
                data={
                    "fused_query": fused_query,
                    "original_query_length": len(query),
                    "fused_query_length": len(fused_query),
                    "selected_strategy": strategy
                }
            )
        
        return fused_query

    def extract_context(self, query: str, history: List[Message], retrieved_docs: List[Document]) -> List[Document]:
        """根据对话历史和当前查询，从检索结果中提取最相关的上下文"""
        # 这里可以实现更复杂的上下文筛选逻辑
        # 目前简单返回所有检索到的文档
        return retrieved_docs

# 全局对话管理器实例
conversation_manager = ConversationManager()
context_fuser = ContextFuser()

from datetime import datetime
from typing import List, Dict, Any, Optional

class ReasoningStep:
    """
    推理步骤类，用于记录Agent推理过程中的每一个步骤
    """
    def __init__(self, step_type: str, description: str, data: Dict[str, Any] = None):
        """
        初始化推理步骤
        
        Args:
            step_type: 步骤类型，如context_fusion, retrieval, reranking, response_generation
            description: 步骤描述，简要说明该步骤的作用
            data: 步骤详细数据，包含该步骤的输入输出和中间结果
        """
        self.step_type = step_type
        self.description = description
        self.data = data or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将推理步骤转换为字典格式，便于序列化和传输
        
        Returns:
            推理步骤的字典表示
        """
        return {
            "step_type": self.step_type,
            "description": self.description,
            "data": self.data,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningStep":
        """
        从字典创建推理步骤实例
        
        Args:
            data: 推理步骤的字典表示
            
        Returns:
            推理步骤实例
        """
        step = cls(
            step_type=data["step_type"],
            description=data["description"],
            data=data["data"]
        )
        step.timestamp = datetime.fromisoformat(data["timestamp"])
        return step

class ReasoningTrace:
    """
    推理轨迹类，用于记录和管理完整的Agent推理过程
    """
    def __init__(self, session_id: str, query: str, fused_query: Optional[str] = None):
        """
        初始化推理轨迹
        
        Args:
            session_id: 会话ID，关联到具体的对话
            query: 原始用户查询
            fused_query: 融合上下文后的查询（可选）
        """
        self.session_id = session_id
        self.query = query
        self.fused_query = fused_query
        self.steps: List[ReasoningStep] = []
        self.created_at = datetime.now()
    
    def add_step(self, step_type: str, description: str, data: Dict[str, Any] = None) -> ReasoningStep:
        """
        添加推理步骤到轨迹中
        
        Args:
            step_type: 步骤类型
            description: 步骤描述
            data: 步骤详细数据
            
        Returns:
            添加的推理步骤实例
        """
        step = ReasoningStep(step_type, description, data)
        self.steps.append(step)
        return step
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将推理轨迹转换为字典格式
        
        Returns:
            推理轨迹的字典表示
        """
        return {
            "session_id": self.session_id,
            "query": self.query,
            "fused_query": self.fused_query,
            "steps": [step.to_dict() for step in self.steps],
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningTrace":
        """
        从字典创建推理轨迹实例
        
        Args:
            data: 推理轨迹的字典表示
            
        Returns:
            推理轨迹实例
        """
        trace = cls(
            session_id=data["session_id"],
            query=data["query"],
            fused_query=data.get("fused_query")
        )
        trace.created_at = datetime.fromisoformat(data["created_at"])
        trace.steps = [ReasoningStep.from_dict(step_data) for step_data in data["steps"]]
        return trace
    
    def get_steps_by_type(self, step_type: str) -> List[ReasoningStep]:
        """
        根据步骤类型获取特定类型的推理步骤
        
        Args:
            step_type: 步骤类型
            
        Returns:
            该类型的推理步骤列表
        """
        return [step for step in self.steps if step.step_type == step_type]

class ReasoningTraceManager:
    """
    推理轨迹管理器，用于管理和存储推理轨迹
    """
    def __init__(self):
        """
        初始化推理轨迹管理器
        """
        self.traces: Dict[str, ReasoningTrace] = {}  # 会话ID到推理轨迹的映射
    
    def create_trace(self, session_id: str, query: str, fused_query: Optional[str] = None) -> ReasoningTrace:
        """
        创建新的推理轨迹
        
        Args:
            session_id: 会话ID
            query: 原始用户查询
            fused_query: 融合上下文后的查询（可选）
            
        Returns:
            新创建的推理轨迹实例
        """
        trace = ReasoningTrace(session_id, query, fused_query)
        self.traces[session_id] = trace
        return trace
    
    def get_trace(self, session_id: str) -> Optional[ReasoningTrace]:
        """
        获取指定会话ID的推理轨迹
        
        Args:
            session_id: 会话ID
            
        Returns:
            推理轨迹实例，如果不存在则返回None
        """
        return self.traces.get(session_id)
    
    def add_step(self, session_id: str, step_type: str, description: str, data: Dict[str, Any] = None) -> Optional[ReasoningStep]:
        """
        向指定会话ID的推理轨迹添加推理步骤
        
        Args:
            session_id: 会话ID
            step_type: 步骤类型
            description: 步骤描述
            data: 步骤详细数据
            
        Returns:
            添加的推理步骤实例，如果会话不存在则返回None
        """
        trace = self.get_trace(session_id)
        if trace:
            return trace.add_step(step_type, description, data)
        return None
    
    def remove_trace(self, session_id: str) -> bool:
        """
        删除指定会话ID的推理轨迹
        
        Args:
            session_id: 会话ID
            
        Returns:
            删除成功返回True，否则返回False
        """
        if session_id in self.traces:
            del self.traces[session_id]
            return True
        return False
    
    def cleanup_old_traces(self, max_age_seconds: int = 3600) -> int:
        """
        清理指定时间前的旧推理轨迹
        
        Args:
            max_age_seconds: 最大保留时间，单位为秒
            
        Returns:
            清理的轨迹数量
        """
        current_time = datetime.now()
        old_session_ids = []
        
        for session_id, trace in self.traces.items():
            age_seconds = (current_time - trace.created_at).total_seconds()
            if age_seconds > max_age_seconds:
                old_session_ids.append(session_id)
        
        for session_id in old_session_ids:
            self.remove_trace(session_id)
        
        return len(old_session_ids)

# 全局推理轨迹管理器实例
trace_manager = ReasoningTraceManager()
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
import os

# 导入对话管理模块
from conversation_manager import conversation_manager, context_fuser

# 导入意图识别模块
from intent_recognizer import IntentRecognizer

# 导入投诉分析模块
from complaint_analyzer import ComplaintAnalyzer

# 导入文书生成模块
from document_generator import DocumentGenerator

# 导入RAG检索模块
from Rag_Retrieved import (
    create_hybrid_retriever,
    hybrid_search
)
from reasoning_trace import trace_manager
from langchain_core.documents import Document

# 加载环境变量
load_dotenv()

# 创建FastAPI应用
app = FastAPI(
    title="CaseCause RAG Agent API",
    description="多轮对话RAG检索系统API",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该配置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("正在初始化RAG组件...")
# 初始化RAG组件
semantic_retriever, bm25_retriever, reranker = create_hybrid_retriever()
logger.info("RAG组件初始化完成")

logger.info("正在初始化意图识别器...")
# 初始化意图识别器
intent_recognizer = IntentRecognizer()
logger.info("意图识别器初始化完成")

logger.info("正在初始化投诉分析器...")
# 初始化投诉分析器
complaint_analyzer = ComplaintAnalyzer()
logger.info("投诉分析器初始化完成")

logger.info("正在初始化文书生成器...")
# 初始化文书生成器
document_generator = DocumentGenerator()
logger.info("文书生成器初始化完成")

# 数据模型
class Message(BaseModel):
    role: str = Field(..., description="消息角色，'user' 或 'assistant'")
    content: str = Field(..., description="消息内容")
    timestamp: datetime = Field(default_factory=datetime.now, description="消息时间戳")

class DialogueState(BaseModel):
    current_task: str = Field(..., description="当前对话任务")
    complaint_points: Optional[Dict[str, Any]] = Field(None, description="提取的投诉要点")
    retrieved_cases: Optional[List[Dict[str, Any]]] = Field(None, description="检索到的案由")
    retrieved_legal: Optional[List[Dict[str, Any]]] = Field(None, description="检索到的法律法规")
    document_type: Optional[str] = Field(None, description="要生成的文书类型")
    document_params: Optional[Dict[str, Any]] = Field(None, description="文书生成参数")
    confirmed: bool = Field(False, description="投诉要点是否已确认")

class ChatRequest(BaseModel):
    message: str = Field(..., description="用户消息内容")
    session_id: Optional[str] = Field(None, description="对话会话ID，不提供则创建新会话")
    metadata_filters: Optional[Dict[str, Any]] = Field(None, description="元数据过滤条件")
    max_rounds: Optional[int] = Field(5, description="最大对话轮次，用于上下文融合")
    dialogue_state: Optional[DialogueState] = Field(None, description="当前对话状态")

class DocumentResponse(BaseModel):
    content: str = Field(..., description="文档内容")
    metadata: Dict[str, Any] = Field(..., description="文档元数据")

    @classmethod
    def from_document(cls, doc: Document) -> "DocumentResponse":
        return cls(
            content=doc.page_content,
            metadata=doc.metadata
        )

class ReasoningStepResponse(BaseModel):
    step_type: str = Field(..., description="步骤类型")
    description: str = Field(..., description="步骤描述")
    data: Dict[str, Any] = Field(..., description="步骤详细数据")
    timestamp: datetime = Field(..., description="步骤时间戳")

class ReasoningTraceResponse(BaseModel):
    session_id: str = Field(..., description="会话ID")
    query: str = Field(..., description="原始用户查询")
    fused_query: Optional[str] = Field(None, description="融合上下文后的查询")
    steps: List[ReasoningStepResponse] = Field(..., description="推理步骤列表")
    created_at: datetime = Field(..., description="推理轨迹创建时间")

class IntentResult(BaseModel):
    intent: str = Field(..., description="识别到的意图")
    confidence: float = Field(..., description="意图置信度")

class ChatResponse(BaseModel):
    session_id: str = Field(..., description="对话会话ID")
    message: Message = Field(..., description="AI响应消息")
    context: List[DocumentResponse] = Field(..., description="检索到的上下文文档")
    reasoning_trace: Optional[ReasoningTraceResponse] = Field(None, description="推理轨迹")
    dialogue_state: DialogueState = Field(..., description="当前对话状态")
    intent_result: IntentResult = Field(..., description="意图识别结果")

class ConversationInfo(BaseModel):
    session_id: str = Field(..., description="对话会话ID")
    created_at: datetime = Field(..., description="会话创建时间")
    updated_at: datetime = Field(..., description="会话更新时间")
    message_count: int = Field(..., description="会话消息数量")

# API端点
@app.post("/api/conversations", response_model=ConversationInfo, summary="创建新的对话会话")
async def create_conversation():
    """创建新的对话会话，返回会话信息"""
    conversation = conversation_manager.create_conversation()
    return ConversationInfo(
        session_id=conversation.session_id,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        message_count=len(conversation.messages)
    )

@app.post("/api/chat", response_model=ChatResponse, summary="发送消息并获取AI响应")
async def chat(chat_request: ChatRequest):
    """
    发送用户消息，返回AI响应
    """
    # 获取或创建对话会话
    if chat_request.session_id:
        conversation = conversation_manager.get_conversation(chat_request.session_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="会话不存在")
    else:
        conversation = conversation_manager.create_conversation()
    
    # 更新会话的对话状态（如果提供）
    if chat_request.dialogue_state:
        conversation.update_dialogue_state(chat_request.dialogue_state.model_dump())
    
    # 添加用户消息到会话
    conversation.add_message("user", chat_request.message)
    
    # 获取对话历史
    history = conversation.get_history(max_rounds=chat_request.max_rounds)
    
    # 创建推理轨迹
    trace = trace_manager.create_trace(
        session_id=conversation.session_id,
        query=chat_request.message
    )
    
    # 意图识别
    intent_result = intent_recognizer.recognize_intent(chat_request.message)
    conversation.add_intent(intent_result["intent"])
    
    # 记录意图识别结果
    trace_manager.add_step(
        session_id=conversation.session_id,
        step_type="intent_recognition",
        description="识别用户意图",
        data={
            "intent": intent_result["intent"],
            "confidence": intent_result["confidence"]
        }
    )
    
    # 获取当前对话状态
    dialogue_state = conversation.dialogue_state
    
    # 融合上下文，考虑对话状态
    fused_query = context_fuser.fuse_context(
        chat_request.message, 
        history, 
        conversation.session_id, 
        dialogue_state
    )
    trace.fused_query = fused_query
    
    # 根据意图和对话状态执行不同的处理流程
    retrieved_docs = []
    response_content = ""
    
    current_task = dialogue_state.get("current_task", "initial")
    intent = intent_result["intent"]
    
    # 1. 初始阶段或投诉描述阶段
    if current_task in ["initial", "complaint_description"] or intent == "complaint_description":
        # 提取投诉要点
        complaint_points = complaint_analyzer.extract_complaint_points(chat_request.message)
        conversation.set_complaint_points(complaint_points)
        
        # 生成响应
        response_content = "我已提取到以下投诉要点：\n\n"
        if complaint_points["complaint_type"]:
            response_content += f"• 投诉类型：{', '.join(complaint_points['complaint_type'])}\n"
        if complaint_points["issues"]:
            response_content += f"• 具体问题：{', '.join(complaint_points['issues'])}\n"
        if complaint_points["persons"]:
            response_content += f"• 涉及人员：{', '.join(complaint_points['persons'])}\n"
        if complaint_points["institutions"]:
            response_content += f"• 涉及机构：{', '.join(complaint_points['institutions'])}\n"
        if complaint_points["times"]:
            response_content += f"• 涉及时间：{', '.join(complaint_points['times'])}\n"
        if complaint_points["locations"]:
            response_content += f"• 涉及地点：{', '.join(complaint_points['locations'])}\n"
        
        response_content += "\n请问这些投诉要点是否正确？您可以回复'确认'或提出修改意见。"
    
    # 2. 投诉要点确认阶段
    elif current_task in ["points_extraction", "points_confirmation"] or intent == "points_confirmation":
        if "确认" in chat_request.message or "是的" in chat_request.message or "没错" in chat_request.message:
            # 用户确认投诉要点
            conversation.confirm_complaint_points(confirmed=True)
            response_content = "投诉要点已确认。我将为您检索相关案由。\n\n"
            
            # 自动检索相关案由
            complaint_points = conversation.dialogue_state["complaint_points"]
            retrieved_cases = complaint_analyzer.retrieve_case_causes(complaint_points, k=3)
            conversation.dialogue_state["retrieved_cases"] = [{
                "page_content": doc.page_content,
                "metadata": doc.metadata
            } for doc in retrieved_cases]
            retrieved_docs = retrieved_cases
            
            # 添加案由检索结果到响应
            for i, doc in enumerate(retrieved_cases):
                response_content += f"{i+1}. 案由名称：{doc.metadata.get('案由名称', '未知')}\n"
                response_content += f"   所属专业：{doc.metadata.get('所属专业', '未知')}\n"
                response_content += f"   内容摘要：{doc.page_content[:150]}...\n\n"
            
            response_content += "您可以回复'检索法律'获取相关法律法规，或'生成文书'请求生成相关文书。"
        else:
            # 用户需要修改投诉要点
            response_content = "请您详细说明需要修改或补充的内容，我将重新提取投诉要点。"
            conversation.set_current_task("points_extraction")
    
    # 3. 案由检索阶段
    elif current_task == "case_retrieval" or intent == "case_retrieval":
        if dialogue_state.get("complaint_points"):
            retrieved_cases = complaint_analyzer.retrieve_case_causes(dialogue_state["complaint_points"], k=3)
            conversation.dialogue_state["retrieved_cases"] = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in retrieved_cases]
            retrieved_docs = retrieved_cases
            
            response_content = "以下是根据您的投诉要点检索到的相关案由：\n\n"
            for i, doc in enumerate(retrieved_cases):
                response_content += f"{i+1}. 案由名称：{doc.metadata.get('案由名称', '未知')}\n"
                response_content += f"   所属专业：{doc.metadata.get('所属专业', '未知')}\n"
                response_content += f"   内容摘要：{doc.page_content[:150]}...\n\n"
            
            response_content += "您可以回复'检索法律'获取相关法律法规，或'生成文书'请求生成相关文书。"
        else:
            response_content = "请先描述您的投诉内容，我将提取投诉要点后再为您检索案由。"
            conversation.set_current_task("complaint_description")
    
    # 4. 法律法规检索阶段
    elif current_task == "legal_retrieval" or intent == "legal_retrieval":
        # 这里可以添加法律法规检索逻辑
        response_content = "法律法规检索功能正在开发中，敬请期待！"
    
    # 5. 文书生成阶段
    elif current_task == "document_generation" or intent == "document_generation":
        if dialogue_state.get("document_type"):
            # 已指定文书类型，开始生成文书
            document_type = dialogue_state["document_type"]
            complaint_points = dialogue_state["complaint_points"]
            additional_params = dialogue_state.get("document_params", {})
            
            # 生成文书
            document_content = document_generator.generate_document(document_type, complaint_points, additional_params)
            
            # 保存文书（可选）
            # document_path = document_generator.save_document(document_content, f"document_{conversation.session_id}.txt")
            
            response_content = f"已为您生成{document_type}：\n\n"
            response_content += document_content
            response_content += "\n\n您可以回复'生成PDF'将文书转换为PDF格式，或'确认'完成流程。"
        else:
            # 未指定文书类型，询问用户
            response_content = "请告诉我您需要生成哪种类型的文书？（如：投诉书、举报信等）"
            conversation.set_current_task("document_generation")
    
    # 6. 澄清请求
    elif intent == "clarification":
        response_content = "请问您需要澄清什么问题？我将为您详细解答。"
    
    # 7. 退出请求
    elif intent == "exit":
        response_content = "感谢您的使用，祝您工作顺利！"
    
    # 8. 默认处理
    else:
        # 执行混合检索
        retrieved_docs = hybrid_search(
            query=fused_query,
            semantic_retriever=semantic_retriever,
            bm25_retriever=bm25_retriever,
            reranker=reranker,
            metadata_filters=chat_request.metadata_filters,
            k=5,
            session_id=conversation.session_id
        )
        
        # 生成响应
        response_content = "基于您的查询，我为您提供以下信息：\n\n"
        for i, doc in enumerate(retrieved_docs):
            response_content += f"{i+1}. {doc.page_content[:100]}...\n"
            response_content += f"   来源：{doc.metadata.get('source', '未知')}\n\n"
        
        # 根据当前任务提供后续操作建议
        if current_task == "case_retrieval":
            response_content += "您可以回复'检索法律'获取相关法律法规，或'生成文书'请求生成相关文书。"
    
    # 记录响应生成
    trace_manager.add_step(
        session_id=conversation.session_id,
        step_type="response_generation",
        description="生成AI响应",
        data={
            "response_content": response_content,
            "retrieved_documents_count": len(retrieved_docs)
        }
    )
    
    # 添加AI响应到会话
    conversation.add_message("assistant", response_content)
    
    # 构建推理轨迹响应
    reasoning_steps = [
        ReasoningStepResponse(
            step_type=step.step_type,
            description=step.description,
            data=step.data,
            timestamp=step.timestamp
        )
        for step in trace.steps
    ]
    
    reasoning_trace_response = ReasoningTraceResponse(
        session_id=trace.session_id,
        query=trace.query,
        fused_query=trace.fused_query,
        steps=reasoning_steps,
        created_at=trace.created_at
    )
    
    # 构建响应
    return ChatResponse(
        session_id=conversation.session_id,
        message=Message(
            role="assistant",
            content=response_content,
            timestamp=datetime.now()
        ),
        context=[DocumentResponse.from_document(doc) for doc in retrieved_docs],
        reasoning_trace=reasoning_trace_response,
        dialogue_state=DialogueState(**conversation.dialogue_state),
        intent_result=IntentResult(**intent_result)
    )

@app.get("/api/conversations/{session_id}/messages", response_model=List[Message], summary="获取对话历史")
async def get_conversation_history(session_id: str):
    """获取指定会话的对话历史"""
    conversation = conversation_manager.get_conversation(session_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    return [
        Message(
            role=msg.role,
            content=msg.content,
            timestamp=msg.timestamp
        ) for msg in conversation.messages
    ]

@app.delete("/api/conversations/{session_id}", summary="结束会话")
async def delete_conversation(session_id: str):
    """
    结束并清理对话会话
    """
    success = conversation_manager.delete_conversation(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    return {"detail": "会话已成功结束"}

@app.post("/api/complaint/points/confirm", summary="确认或修改投诉要点")
async def confirm_complaint_points(session_id: str, confirmed: bool, complaint_points: Optional[Dict[str, Any]] = None):
    """
    确认或修改提取的投诉要点
    
    Args:
        session_id: 对话会话ID
        confirmed: 是否确认投诉要点
        complaint_points: 修改后的投诉要点（可选）
    
    Returns:
        操作结果
    """
    # 获取对话会话
    conversation = conversation_manager.get_conversation(session_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    # 更新投诉要点（如果提供）
    if complaint_points:
        conversation.set_complaint_points(complaint_points)
    
    # 确认投诉要点
    conversation.confirm_complaint_points(confirmed)
    
    return {
        "detail": "投诉要点已成功确认" if confirmed else "投诉要点已成功更新",
        "dialogue_state": conversation.dialogue_state
    }

@app.get("/api/conversations", response_model=List[ConversationInfo], summary="获取所有会话列表")
async def get_all_conversations():
    """获取所有活跃的对话会话列表"""
    conversations = []
    for conv in conversation_manager.conversations.values():
        conversations.append(
            ConversationInfo(
                session_id=conv.session_id,
                created_at=conv.created_at,
                updated_at=conv.updated_at,
                message_count=len(conv.messages)
            )
        )
    
    # 按更新时间倒序排序
    conversations.sort(key=lambda x: x.updated_at, reverse=True)
    
    return conversations

@app.get("/api/conversations/{session_id}/reasoning", response_model=ReasoningTraceResponse, summary="获取会话推理轨迹")
async def get_conversation_reasoning(session_id: str):
    """获取指定会话的完整推理轨迹"""
    # 获取推理轨迹
    trace = trace_manager.get_trace(session_id)
    if not trace:
        raise HTTPException(status_code=404, detail="推理轨迹不存在")
    
    # 构建推理轨迹响应
    reasoning_steps = [
        ReasoningStepResponse(
            step_type=step.step_type,
            description=step.description,
            data=step.data,
            timestamp=step.timestamp
        )
        for step in trace.steps
    ]
    
    return ReasoningTraceResponse(
        session_id=trace.session_id,
        query=trace.query,
        fused_query=trace.fused_query,
        steps=reasoning_steps,
        created_at=trace.created_at
    )

# 根路径，用于健康检查
@app.get("/")
async def root():
    return {
        "message": "CaseCause RAG Agent API is running",
        "version": "1.0.0",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }

# 运行应用
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

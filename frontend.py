import streamlit as st
import requests
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

# API配置
API_BASE_URL = "http://localhost:8000"

# 页面配置
st.set_page_config(
    page_title="CaseCause RAG Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 样式定制
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        max-width: 80%;
    }
    .user-message {
        background-color: #3b82f6;
        color: white;
        margin-left: auto;
    }
    .assistant-message {
        background-color: #f3f4f6;
        color: black;
        margin-right: auto;
    }
    .message-content {
        margin-bottom: 0.5rem;
    }
    .message-timestamp {
        font-size: 0.75rem;
        opacity: 0.7;
    }
    .sidebar-section {
        margin-bottom: 2rem;
    }
    .context-card {
        background-color: #f9fafb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #3b82f6;
    }
    .context-metadata {
        font-size: 0.875rem;
        color: #6b7280;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# 辅助函数
def create_conversation() -> Dict[str, Any]:
    """创建新的对话会话"""
    response = requests.post(f"{API_BASE_URL}/api/conversations")
    response.raise_for_status()
    return response.json()

def send_message(session_id: str, message: str, metadata_filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """发送消息到指定会话"""
    payload = {
        "message": message,
        "session_id": session_id,
        "metadata_filters": metadata_filters
    }
    response = requests.post(f"{API_BASE_URL}/api/chat", json=payload)
    response.raise_for_status()
    return response.json()

def get_conversation_history(session_id: str) -> List[Dict[str, Any]]:
    """获取对话历史"""
    response = requests.get(f"{API_BASE_URL}/api/conversations/{session_id}/messages")
    response.raise_for_status()
    return response.json()

def get_all_conversations() -> List[Dict[str, Any]]:
    """获取所有对话会话"""
    response = requests.get(f"{API_BASE_URL}/api/conversations")
    response.raise_for_status()
    return response.json()

def delete_conversation(session_id: str) -> bool:
    """删除对话会话"""
    response = requests.delete(f"{API_BASE_URL}/api/conversations/{session_id}")
    return response.status_code == 200

def format_timestamp(timestamp_str: str) -> str:
    """格式化时间戳"""
    timestamp = datetime.fromisoformat(timestamp_str)
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

# 主应用
def main():
    st.title("🤖 CaseCause RAG Agent")
    st.subheader("多轮对话式法律检索系统")
    
    # 初始化会话状态
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "conversations" not in st.session_state:
        st.session_state.conversations = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "context" not in st.session_state:
        st.session_state.context = []
    if "metadata_filters" not in st.session_state:
        st.session_state.metadata_filters = {}
    
    # 侧边栏
    with st.sidebar:
        st.header("会话管理")
        
        # 刷新会话列表
        if st.button("刷新会话列表"):
            st.session_state.conversations = get_all_conversations()
        
        # 会话列表
        st.subheader("对话会话")
        for conv in st.session_state.conversations:
            is_active = conv["session_id"] == st.session_state.session_id
            btn_label = f"📌 {conv['session_id'][:8]}..." if is_active else f"{conv['session_id'][:8]}..."
            
            col1, col2 = st.columns([3, 1])
            with col1:
                if col1.button(btn_label, key=conv["session_id"], use_container_width=True):
                    st.session_state.session_id = conv["session_id"]
                    # 加载对话历史
                    st.session_state.messages = get_conversation_history(conv["session_id"])
                    st.session_state.context = []
                    st.rerun()
            with col2:
                if col2.button("🗑️", key=f"delete_{conv['session_id']}", use_container_width=True):
                    delete_conversation(conv["session_id"])
                    st.session_state.conversations = get_all_conversations()
                    # 如果删除的是当前会话，清除当前会话
                    if conv["session_id"] == st.session_state.session_id:
                        st.session_state.session_id = None
                        st.session_state.messages = []
                        st.session_state.context = []
                    st.rerun()
        
        # 创建新会话
        if st.button("➕ 创建新会话", use_container_width=True):
            new_conv = create_conversation()
            st.session_state.session_id = new_conv["session_id"]
            st.session_state.conversations = get_all_conversations()
            st.session_state.messages = []
            st.session_state.context = []
            st.rerun()
        
        # 设置
        st.header("检索设置")
        
        # 元数据过滤
        st.subheader("元数据过滤")
        show_metadata_filters = st.checkbox("启用元数据过滤")
        if show_metadata_filters:
            # 这里可以根据实际情况添加具体的元数据过滤选项
            # 例如：所属专业、类型等
            major = st.text_input("所属专业")
            doc_type = st.text_input("文档类型")
            
            st.session_state.metadata_filters = {}
            if major:
                st.session_state.metadata_filters["所属专业"] = major
            if doc_type:
                st.session_state.metadata_filters["类型"] = doc_type
        else:
            st.session_state.metadata_filters = {}
    
    # 主内容区
    col1, col2 = st.columns([3, 2])
    
    # 对话区
    with col1:
        st.header("对话")
        
        # 显示对话历史
        for msg in st.session_state.messages:
            role = msg["role"]
            content = msg["content"]
            timestamp = format_timestamp(msg["timestamp"])
            
            if role == "user":
                st.markdown(
                    f'''<div class="chat-message user-message">
                        <div class="message-content">{content}</div>
                        <div class="message-timestamp">{timestamp}</div>
                    </div>''',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'''<div class="chat-message assistant-message">
                        <div class="message-content">{content}</div>
                        <div class="message-timestamp">{timestamp}</div>
                    </div>''',
                    unsafe_allow_html=True
                )
        
        # 消息输入
        st.write("\n")
        with st.form(key="chat_form"):
            user_input = st.text_area("请输入您的问题：", height=100)
            col1, col2 = st.columns([1, 3])
            with col1:
                submit_button = st.form_submit_button("发送", use_container_width=True)
            with col2:
                clear_button = st.form_submit_button("清除对话", use_container_width=True)
        
        # 处理用户输入
        if submit_button and user_input.strip():
            # 确保有活跃会话
            if not st.session_state.session_id:
                new_conv = create_conversation()
                st.session_state.session_id = new_conv["session_id"]
                st.session_state.conversations = get_all_conversations()
            
            # 添加用户消息到界面
            user_msg = {
                "role": "user",
                "content": user_input.strip(),
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(user_msg)
            st.rerun()
            
            # 发送消息到API
            with st.spinner("正在处理您的请求..."):
                try:
                    response = send_message(
                        st.session_state.session_id,
                        user_input.strip(),
                        st.session_state.metadata_filters
                    )
                    
                    # 添加AI响应到界面
                    assistant_msg = response["message"]
                    st.session_state.messages.append(assistant_msg)
                    
                    # 保存检索到的上下文
                    st.session_state.context = response["context"]
                    
                    # 刷新会话列表
                    st.session_state.conversations = get_all_conversations()
                    
                    st.rerun()
                except Exception as e:
                    st.error(f"发送消息失败：{str(e)}")
        
        # 清除对话
        if clear_button:
            if st.session_state.session_id:
                # 结束当前会话
                delete_conversation(st.session_state.session_id)
            # 创建新会话
            new_conv = create_conversation()
            st.session_state.session_id = new_conv["session_id"]
            st.session_state.conversations = get_all_conversations()
            st.session_state.messages = []
            st.session_state.context = []
            st.rerun()
    
    # 上下文展示区
    with col2:
        st.header("检索上下文")
        
        if st.session_state.context:
            st.write(f"共检索到 {len(st.session_state.context)} 个相关文档：")
            
            for i, context_doc in enumerate(st.session_state.context):
                with st.expander(f"文档 {i+1}", expanded=True):
                    st.markdown(f'''<div class='context-card'>
                        <strong>内容：</strong>
                        <p>{context_doc['content']}</p>
                        <div class='context-metadata'>
                            <strong>元数据：</strong>
                            {json.dumps(context_doc['metadata'], ensure_ascii=False, indent=2)}
                        </div>
                    </div>''', unsafe_allow_html=True)
        else:
            st.info("暂无检索上下文，发送消息后将显示相关文档")

# 运行应用
if __name__ == "__main__":
    main()

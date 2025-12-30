import streamlit as st
import requests
import json

# 设置页面标题和样式
st.set_page_config(
    page_title="多轮对话RAG系统",
    page_icon="💬",
    layout="wide"
)

# API地址配置
API_URL = "http://localhost:8000"

# 初始化会话状态
if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "dialogue_state" not in st.session_state:
    st.session_state.dialogue_state = None

# 侧边栏配置
with st.sidebar:
    st.title("💬 多轮对话RAG系统")
    st.write("基于规则+AI的多轮对话系统，支持投诉要点提取、案由检索和文书生成")
    
    # 会话管理
    st.subheader("会话管理")
    
    if st.button("创建新会话"):
        st.session_state.session_id = None
        st.session_state.messages = []
        st.session_state.dialogue_state = None
        st.success("已创建新会话")
    
    if "session_id" in st.session_state and st.session_state.session_id:
        st.write(f"当前会话ID: {st.session_state.session_id}")
    
    # API状态测试
    st.subheader("API状态")
    try:
        response = requests.get(f"{API_URL}")
        if response.status_code == 200:
            st.success("API服务正常运行")
        else:
            st.error(f"API服务异常，状态码: {response.status_code}")
    except Exception as e:
        st.error(f"无法连接到API服务: {e}")

# 主对话界面
def send_message(message):
    """发送消息到API"""
    if not message:
        return
    
    # 添加用户消息到对话历史
    st.session_state.messages.append({
        "role": "user",
        "content": message
    })
    
    # 构建API请求
    payload = {
        "message": message,
        "session_id": st.session_state.session_id,
        "max_rounds": 5
    }
    
    # 如果有对话状态，添加到请求中
    if st.session_state.dialogue_state:
        payload["dialogue_state"] = st.session_state.dialogue_state
    
    try:
        # 发送请求到API
        response = requests.post(
            f"{API_URL}/api/chat",
            json=payload
        )
        
        if response.status_code == 200:
            # 解析响应
            result = response.json()
            
            # 更新会话ID
            st.session_state.session_id = result["session_id"]
            
            # 更新对话状态
            st.session_state.dialogue_state = result["dialogue_state"]
            
            # 添加助手响应到对话历史
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["message"]["content"]
            })
            
            # 显示对话状态
            with st.expander("查看对话状态", expanded=False):
                st.json(result["dialogue_state"])
            
            # 显示意图识别结果
            with st.expander("查看意图识别结果", expanded=False):
                st.json(result["intent_result"])
            
            # 显示检索到的上下文
            if result["context"]:
                with st.expander("查看检索上下文", expanded=False):
                    for i, doc in enumerate(result["context"]):
                        st.subheader(f"文档 {i+1}")
                        st.write(doc["content"])
                        st.caption(f"来源: {doc['metadata'].get('source', '未知')}")
        else:
            st.error(f"API请求失败: {response.text}")
            # 添加错误消息到对话历史
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"抱歉，请求失败: {response.text}"
            })
    except Exception as e:
        st.error(f"网络请求失败: {e}")
        # 添加错误消息到对话历史
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"抱歉，无法连接到服务器: {e}"
        })
    
    # 清空输入框
    # st.session_state.user_input = ""

# 显示对话历史
st.title("💬 多轮对话界面")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 输入框
user_input = st.text_input(
    "输入您的消息",
    placeholder="请输入您的投诉描述或问题...",
    label_visibility="hidden"
)

if st.button("发送"):
    if user_input:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        send_message(user_input)

# 示例问题
st.divider()
st.subheader("示例问题")
example_questions = [
    "我要投诉某医院非法行医",
    "确认",
    "检索相关案由",
    "生成投诉书"
]

def send_example_message(question):
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })
    # 构建API请求
    payload = {
        "message": question,
        "session_id": st.session_state.session_id,
        "max_rounds": 5
    }
    
    # 如果有对话状态，添加到请求中
    if st.session_state.dialogue_state:
        payload["dialogue_state"] = st.session_state.dialogue_state
    
    try:
        # 发送请求到API
        response = requests.post(
            f"{API_URL}/api/chat",
            json=payload
        )
        
        if response.status_code == 200:
            # 解析响应
            result = response.json()
            
            # 更新会话ID
            st.session_state.session_id = result["session_id"]
            
            # 更新对话状态
            st.session_state.dialogue_state = result["dialogue_state"]
            
            # 添加助手响应到对话历史
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["message"]["content"]
            })
            
            # 显示对话状态
            with st.expander("查看对话状态", expanded=False):
                st.json(result["dialogue_state"])
            
            # 显示意图识别结果
            with st.expander("查看意图识别结果", expanded=False):
                st.json(result["intent_result"])
            
            # 显示检索到的上下文
            if result["context"]:
                with st.expander("查看检索上下文", expanded=False):
                    for i, doc in enumerate(result["context"]):
                        st.subheader(f"文档 {i+1}")
                        st.write(doc["content"])
                        st.caption(f"来源: {doc['metadata'].get('source', '未知')}")
        else:
            st.error(f"API请求失败: {response.text}")
            # 添加错误消息到对话历史
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"抱歉，请求失败: {response.text}"
            })
    except Exception as e:
        st.error(f"网络请求失败: {e}")
        # 添加错误消息到对话历史
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"抱歉，无法连接到服务器: {e}"
        })



for question in example_questions:
    if st.button(question):
        send_example_message(question)
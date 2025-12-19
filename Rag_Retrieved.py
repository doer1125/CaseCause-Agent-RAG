from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import Docx2txtLoader
from pathlib import Path
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import pandas as pd
from dotenv import load_dotenv

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter  import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from helper_functions import num_tokens_from_string, replace_t_with_space, replace_double_lines_with_one_line, split_into_chapters,\
analyse_metric_results, escape_quotes, text_wrap,extract_book_quotes_as_documents

load_dotenv()
os.environ["DEEPSEEK_API_KEY"] = os.getenv('DEEPSEEK_API_KEY')

def encode_excel(path, chunk_size=1000, chunk_overlap=200):
    """
    利用OpenAI嵌入将Excel文件编码到矢量存储中。

    Args：
        path：指向Excel文件的路径。
        chunk_size：每个文本块的期望大小。
        chunk_overlap：连续区块之间的重叠程度。

    Returns：
        一个包含编码CSV内容的FAISS向量存储器。
    """

    # Load Excel documents
    # 使用pandas读取Excel文件
    df = pd.read_excel(path)

    # 将DataFrame转换为文档
    documents = []
    for index, row in df.iterrows():
        # 将每一行转换为文本
        row_text = ""
        for col_name, value in row.items():
            if pd.notna(value):  # 跳过空值
                row_text += f"{col_name}: {value}\n"

        if row_text.strip():  # 只添加非空行
            doc = Document(
                page_content=row_text.strip(),
                metadata={"row_index": index, "source": path}
            )
            documents.append(doc)

    document_cleaned = replace_t_with_space(documents)

    # Create embeddings and vector store
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # 本地模型路径
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh")   
    vectorstore = FAISS.from_documents(document_cleaned, embeddings)

    return vectorstore


def encode_word(path, chunk_size=1000, chunk_overlap=200):
    """
    利用嵌入将Word文件编码到矢量存储中。

    Args：
        path：指向Word文件的路径。
        chunk_size：每个文本块的期望大小。
        chunk_overlap：连续区块之间的重叠程度。

    Returns：
        一个包含编码Word内容的FAISS向量存储器。
    """

    print(f"使用Docx2txtLoader读取: {path}")

    # 加载文档 - 使用Docx2txtLoader替代UnstructuredWordDocumentLoader
    loader = Docx2txtLoader(path)
    documents = loader.load()

    if not documents:
        raise ValueError("文档内容为空")

    print(f"成功读取Word文档，内容长度: {len(documents[0].page_content)} 字符")

    # 创建embeddings和vector store
    # 创建文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""]
    )

    # 分割文档
    chunks = text_splitter.split_documents(documents)
    print(f"文档分割为 {len(chunks)} 个块")

    # 重新创建向量存储
    # 本地模型路径
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh")   
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore

def Creating_Vector_Store():
    """
    创建并保存向量存储。
    """
    file_path = ('data/案由清单.xlsx') # insert the path of the excel file
    word_path = ('data/最新版《行政处罚法》.docx')
    vectorstore = encode_excel(file_path)
    word_vectorstore = encode_word(word_path)
    vectorstore.save_local("case_causes_vector_store")
    word_vectorstore.save_local("legal_vector_store")

def create_retrievers():
    # 本地模型路径
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh")   
    case_causes_vector_store = FAISS.load_local("case_causes_vector_store", embeddings, allow_dangerous_deserialization=True)
    legal_vector_store = FAISS.load_local("legal_vector_store", embeddings, allow_dangerous_deserialization=True)

    case_causes_retriever = case_causes_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    legal_retriever = legal_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})

    return case_causes_retriever, legal_retriever

def retrieve_context_per_question(state):
    """
    根据问题从向量存储中检索上下文。

    Args：
        state：包含问题的状态字典。

    Returns：
        更新后的状态字典，包含检索到的上下文。
    """
    question = state["question"]
    retriever = state["retriever"]
    if "retrieved_context" not in state:
        state["retrieved_context"] = []
    if retriever == case_causes_retriever:
        case_causes_context = retriever.get_relevant_documents(question)
        state["retrieved_context"] = case_causes_context
    elif retriever == legal_retriever:
        legal_context = retriever.get_relevant_documents(question)
        state["retrieved_context"] = legal_context
    return state


if __name__=="__main__":
    # Creating_Vector_Store()
    case_causes_retriever, legal_retriever = create_retrievers()
    init_state = {"question": "行政机关对当事人进行处罚不使用罚款、没收财物单据或者使用非法定部门制发的罚款、没收财物单据的，当事人是否有权利拒绝？",
                  "retriever":legal_retriever}
    state = retrieve_context_per_question(init_state)
    print(state['retrieved_context'])



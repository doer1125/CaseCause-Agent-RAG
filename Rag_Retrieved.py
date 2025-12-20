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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from helper_functions import (
    num_tokens_from_string, replace_t_with_space, replace_double_lines_with_one_line, 
    split_into_chapters, analyse_metric_results, escape_quotes, text_wrap, 
    extract_book_quotes_as_documents
)
from bm25_retriever import BM25Retriever
from reranker import Reranker

load_dotenv()
os.environ["DEEPSEEK_API_KEY"] = os.getenv('DEEPSEEK_API_KEY')

def encode_excel(path, chunk_size=1000, chunk_overlap=200):
    """
    利用BGE-large-zh嵌入将Excel文件编码到矢量存储中。

    Args：
        path：指向Excel文件的路径。
        chunk_size：每个文本块的期望大小。
        chunk_overlap：连续区块之间的重叠程度。

    Returns：
        一个包含编码Excel内容的FAISS向量存储器。
    """

    # Load Excel documents
    # 使用pandas读取Excel文件
    df = pd.read_excel(path)

    # 将DataFrame转换为文档
    documents = []
    for index, row in df.iterrows():
        # 构建完整的文本内容
        row_text = f"案由名称：{row['案由名称']}\n"
        row_text += f"违法依据：{row['违法依据']}\n"
        row_text += f"行政处罚依据：{row['行政处罚依据']}\n"
        row_text += f"处罚内容：{row['处罚内容']}"
        
        # 构建丰富的元数据
        metadata = {
            "row_index": index,
            "source": path,
            "案由名称": row['案由名称'],
            "所属专业": row['所属专业'],
            "适用的法律法规": row['适用的法律法规'],
            "类型": "案由"
        }

        doc = Document(
            page_content=row_text.strip(),
            metadata=metadata
        )
        documents.append(doc)

    document_cleaned = replace_t_with_space(documents)

    # Create embeddings and vector store
    # 使用本地BGE-large-zh模型
    embeddings = HuggingFaceEmbeddings(
        model_name="local_models/bge-large-zh",
        model_kwargs={'device': 'cpu'}
    )   
    vectorstore = FAISS.from_documents(document_cleaned, embeddings)

    return vectorstore


def encode_word(path, chunk_size=1000, chunk_overlap=200):
    """
    利用BGE-large-zh嵌入将Word文件编码到矢量存储中。

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

    # 分割文档并添加元数据
    chunks = text_splitter.split_documents(documents)
    print(f"文档分割为 {len(chunks)} 个块")
    
    # 为每个块添加丰富的元数据
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "类型": "法律法规",
            "来源": "行政处罚法",
            "chunk_index": i
        })

    # 重新创建向量存储
    # 使用本地BGE-large-zh模型
    embeddings = HuggingFaceEmbeddings(
        model_name="local_models/bge-large-zh",
        model_kwargs={'device': 'cpu'}
    )   
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
    # 使用本地BGE-large-zh模型
    embeddings = HuggingFaceEmbeddings(
        model_name="local_models/bge-large-zh",
        model_kwargs={'device': 'cpu'}
    )   
    case_causes_vector_store = FAISS.load_local("case_causes_vector_store", embeddings, allow_dangerous_deserialization=True)
    legal_vector_store = FAISS.load_local("legal_vector_store", embeddings, allow_dangerous_deserialization=True)

    # 设置更大的k值以获取更多候选结果，用于后续混合检索
    case_causes_retriever = case_causes_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 20})
    legal_retriever = legal_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 20})

    return case_causes_retriever, legal_retriever

def _match_metadata(metadata, metadata_filters):
    """
    检查文档元数据是否匹配过滤条件
    
    Args：
        metadata：文档的元数据
        metadata_filters：元数据过滤条件，格式为{"key": "value"}
        
    Returns：
        如果匹配返回True，否则返回False
    """
    if not metadata_filters:
        return True
    
    for key, value in metadata_filters.items():
        if key not in metadata or metadata[key] != value:
            return False
    
    return True

def hybrid_retrieval(query, semantic_retriever, bm25_retriever, metadata_filters=None, k=10):
    """
    混合检索：结合语义检索和BM25关键词检索
    
    Args：
        query：查询文本
        semantic_retriever：语义检索器
        bm25_retriever：BM25关键词检索器
        metadata_filters：元数据过滤条件
        k：返回结果数量
        
    Returns：
        混合检索结果
    """
    # 语义检索
    # 使用_langchain_core.Retriever的invoke方法，这是标准接口
    semantic_results = semantic_retriever.invoke(query)
    
    # BM25检索
    bm25_results_with_scores = bm25_retriever.retrieve(query, k=k)
    bm25_results = [doc for doc, score in bm25_results_with_scores]
    
    # 元数据过滤
    if metadata_filters:
        semantic_results = [doc for doc in semantic_results if _match_metadata(doc.metadata, metadata_filters)]
        bm25_results = [doc for doc in bm25_results if _match_metadata(doc.metadata, metadata_filters)]
    
    # 合并结果（去重）
    combined_results = []
    seen_ids = set()
    
    # 先添加语义检索结果
    for doc in semantic_results:
        doc_id = f"{doc.metadata.get('source', '')}_{doc.metadata.get('row_index', '')}_{doc.metadata.get('chunk_index', '')}"
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            combined_results.append((doc, 'semantic'))
    
    # 再添加BM25检索结果
    for doc in bm25_results:
        doc_id = f"{doc.metadata.get('source', '')}_{doc.metadata.get('row_index', '')}_{doc.metadata.get('chunk_index', '')}"
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            combined_results.append((doc, 'bm25'))
    
    return combined_results

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
        case_causes_context = retriever.invoke(question)
        state["retrieved_context"] = case_causes_context
    elif retriever == legal_retriever:
        legal_context = retriever.invoke(question)
        state["retrieved_context"] = legal_context
    return state

def create_hybrid_retriever():
    """
    创建混合检索器，包括语义检索器、BM25检索器和重排序器
    
    Returns：
        语义检索器、BM25检索器和重排序器
    """
    # 加载向量存储
    embeddings = HuggingFaceEmbeddings(
        model_name="local_models/bge-large-zh",
        model_kwargs={'device': 'cpu'}
    )   
    case_causes_vector_store = FAISS.load_local("case_causes_vector_store", embeddings, allow_dangerous_deserialization=True)
    
    # 创建语义检索器
    semantic_retriever = case_causes_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 20})
    
    # 重新加载文档以初始化BM25
    df = pd.read_excel('data/案由清单.xlsx')
    documents = []
    for index, row in df.iterrows():
        # 构建文档
        row_text = f"案由名称：{row['案由名称']}\n"
        row_text += f"违法依据：{row['违法依据']}\n"
        row_text += f"行政处罚依据：{row['行政处罚依据']}\n"
        row_text += f"处罚内容：{row['处罚内容']}"
        
        metadata = {
            "row_index": index,
            "source": 'data/案由清单.xlsx',
            "案由名称": row['案由名称'],
            "所属专业": row['所属专业'],
            "适用的法律法规": row['适用的法律法规'],
            "类型": "案由"
        }
        
        doc = Document(page_content=row_text.strip(), metadata=metadata)
        documents.append(doc)
    
    # 创建BM25检索器
    bm25_retriever = BM25Retriever(documents)
    
    # 创建重排序器
    reranker = Reranker()
    
    return semantic_retriever, bm25_retriever, reranker

def hybrid_search(query, semantic_retriever, bm25_retriever, reranker, metadata_filters=None, k=5):
    """
    完整的混合检索流程：语义检索 + BM25检索 + 结果去重 + 重排序
    
    Args：
        query：查询文本
        semantic_retriever：语义检索器
        bm25_retriever：BM25检索器
        reranker：重排序器
        metadata_filters：元数据过滤条件
        k：返回结果数量
        
    Returns：
        重排序后的检索结果
    """
    # 混合检索
    combined_results = hybrid_retrieval(query, semantic_retriever, bm25_retriever, metadata_filters, k=20)
    
    # 提取文档列表
    documents = [doc for doc, source in combined_results]
    
    # 重排序
    reranked_results = reranker.rerank(query, documents, k=k)
    
    return reranked_results


if __name__=="__main__":
    # 首先创建向量存储
    # Creating_Vector_Store()
    # 创建混合检索器
    semantic_retriever, bm25_retriever, reranker = create_hybrid_retriever()

    # 执行混合检索
    query = "未取得医疗机构执业许可证擅自执业"
    results = hybrid_search(query, semantic_retriever, bm25_retriever, reranker, k=3)

    # 带元数据过滤的混合检索
    metadata_filters = {"所属专业": "医疗机构"}
    results = hybrid_search(query, semantic_retriever, bm25_retriever, reranker, metadata_filters=metadata_filters, k=1)
    print(results)



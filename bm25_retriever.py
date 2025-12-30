from rank_bm25 import BM25Okapi
import jieba
from langchain_core.documents import Document
from reasoning_trace import trace_manager

class BM25Retriever:
    def __init__(self, documents):
        """
        初始化BM25检索器
        
        Args:
            documents: Document对象列表
        """
        self.documents = documents
        self.tokenized_corpus = [self.tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def tokenize(self, text):
        """
        使用jieba进行中文分词
        
        Args:
            text: 待分词的文本
            
        Returns:
            分词后的词语列表
        """
        # 使用jieba进行中文分词，采用精确模式
        return list(jieba.cut_for_search(text))
    
    def retrieve(self, query, k=5, session_id=None):
        """
        使用BM25进行关键词检索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            session_id: 会话ID，用于记录推理轨迹
            
        Returns:
            包含Document对象和BM25分数的元组列表
        """
        # 记录BM25检索开始
        if session_id:
            trace_manager.add_step(
                session_id=session_id,
                step_type="bm25_retrieval",
                description="开始BM25检索",
                data={"query": query, "k": k}
            )
        
        # 对查询进行分词
        tokenized_query = self.tokenize(query)
        
        # 记录查询分词结果
        if session_id:
            trace_manager.add_step(
                session_id=session_id,
                step_type="bm25_retrieval",
                description="查询分词完成",
                data={"tokenized_query": tokenized_query}
            )
        
        # 获取BM25分数
        scores = self.bm25.get_scores(tokenized_query)
        
        # 记录BM25分数计算结果
        if session_id:
            trace_manager.add_step(
                session_id=session_id,
                step_type="bm25_retrieval",
                description="BM25分数计算完成",
                data={
                    "scores": scores.tolist(),
                    "max_score": float(max(scores) if scores.size > 0 else 0),
                    "min_score": float(min(scores) if scores.size > 0 else 0)
                }
            )
        
        # 获取top-k文档
        top_indices = scores.argsort()[-k:][::-1]
        top_documents = [self.documents[i] for i in top_indices]
        top_scores = [scores[i] for i in top_indices]
        
        # 记录BM25检索结果
        if session_id:
            trace_manager.add_step(
                session_id=session_id,
                step_type="bm25_retrieval",
                description="BM25检索完成",
                data={
                    "top_k_results": [
                        {
                            "document_index": int(doc_idx),
                            "score": float(score),
                            "content_snippet": self.documents[doc_idx].page_content[:100],
                            "metadata": self.documents[doc_idx].metadata
                        }
                        for doc_idx, score in zip(top_indices, top_scores)
                    ],
                    "total_documents": len(self.documents)
                }
            )
        
        return list(zip(top_documents, top_scores))
    
    def filter_by_metadata(self, documents, metadata_filters):
        """
        根据元数据过滤文档
        
        Args:
            documents: Document对象列表
            metadata_filters: 元数据过滤条件，格式为{"key": "value"}
            
        Returns:
            过滤后的Document对象列表
        """
        filtered_documents = []
        for doc in documents:
            match = True
            for key, value in metadata_filters.items():
                if key not in doc.metadata or doc.metadata[key] != value:
                    match = False
                    break
            if match:
                filtered_documents.append(doc)
        return filtered_documents

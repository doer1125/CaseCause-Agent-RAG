from rank_bm25 import BM25Okapi
import jieba
from langchain_core.documents import Document

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
    
    def retrieve(self, query, k=5):
        """
        使用BM25进行关键词检索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            包含Document对象和BM25分数的元组列表
        """
        # 对查询进行分词
        tokenized_query = self.tokenize(query)
        
        # 获取BM25分数
        scores = self.bm25.get_scores(tokenized_query)
        
        # 获取top-k文档
        top_indices = scores.argsort()[-k:][::-1]
        top_documents = [self.documents[i] for i in top_indices]
        top_scores = [scores[i] for i in top_indices]
        
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

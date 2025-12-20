import jieba
from rank_bm25 import BM25Okapi

class Reranker:
    def __init__(self):
        """
        初始化重排序器，使用基于BM25和关键词匹配的简单重排序策略
        """
        pass
    
    def tokenize(self, text):
        """
        使用jieba进行中文分词
        
        Args:
            text: 待分词的文本
            
        Returns:
            分词后的词语列表
        """
        return list(jieba.cut_for_search(text))
    
    def rerank(self, query, documents, k=5):
        """
        根据查询和文档的相关性对结果进行重排序
        使用基于BM25的简单重排序策略
        
        Args:
            query: 查询文本
            documents: Document对象列表
            k: 返回结果数量
            
        Returns:
            重排序后的Document对象列表
        """
        if not documents:
            return []
        
        # 对文档内容进行分词
        tokenized_corpus = [self.tokenize(doc.page_content) for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)
        
        # 对查询进行分词
        tokenized_query = self.tokenize(query)
        
        # 获取BM25分数
        scores = bm25.get_scores(tokenized_query)
        
        # 结合文档元数据进行加权
        weighted_scores = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            # 基础BM25分数
            total_score = score
            
            # 元数据匹配加分
            # 如果案由名称包含查询关键词，加分
            if query in doc.metadata.get("案由名称", ""):
                total_score += 2.0
            
            # 如果所属专业包含查询关键词，加分
            if query in doc.metadata.get("所属专业", ""):
                total_score += 1.0
            
            weighted_scores.append((doc, total_score))
        
        # 按分数排序
        weighted_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回top-k结果
        return [doc for doc, score in weighted_scores[:k]]
    
    def rerank_with_metadata_score(self, query, documents, metadata_weights=None, k=5):
        """
        结合元数据权重进行重排序
        
        Args:
            query: 查询文本
            documents: Document对象列表
            metadata_weights: 元数据权重，格式为{"key": weight}
            k: 返回结果数量
            
        Returns:
            重排序后的Document对象列表
        """
        if not documents:
            return []
            
        if metadata_weights is None:
            metadata_weights = {}
        
        # 对文档内容进行分词
        tokenized_corpus = [self.tokenize(doc.page_content) for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)
        
        # 对查询进行分词
        tokenized_query = self.tokenize(query)
        
        # 获取BM25分数
        scores = bm25.get_scores(tokenized_query)
        
        # 结合元数据权重进行加权
        weighted_scores = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            # 基础BM25分数
            total_score = score
            
            # 元数据匹配加分
            for key, weight in metadata_weights.items():
                if key in doc.metadata:
                    # 如果元数据值包含查询关键词，按权重加分
                    if query in str(doc.metadata[key]):
                        total_score += weight
            
            weighted_scores.append((doc, total_score))
        
        # 按分数排序
        weighted_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回top-k结果
        return [doc for doc, score in weighted_scores[:k]]

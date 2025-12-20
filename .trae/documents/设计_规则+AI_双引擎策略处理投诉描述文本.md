# 基于BGE-large-zh的多源向量库构建与混合检索方案

## 一、方案概述

本方案将构建基于BGE-large-zh的多源向量库，并实现"语义+BM25关键词+元数据"的混合检索与结果重排序功能。

## 二、实施步骤

### 1. 改进向量库构建

**文件修改**：`Rag_Retrieved.py`

**改进内容**：

* 优化Excel文档处理，为每个案由添加更丰富的元数据

* 增加元数据字段：案由名称、所属专业、适用法律法规等

* 确保每个文档的page\_content包含完整的案由信息

**核心代码**：

```python
def encode_excel(path, chunk_size=1000, chunk_overlap=200):
    # 读取Excel文件
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
    
    # 使用BGE-large-zh模型进行嵌入
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh", model_kwargs={'device': 'cpu'})
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    return vectorstore
```

### 2. 实现BM25关键词检索

**文件创建**：`bm25_retriever.py`

**功能实现**：

* 使用rank\_bm25库实现BM25检索

* 支持中文分词（使用jieba或hanlp）

* 实现BM25检索器类

**核心代码**：

```python
from rank_bm25 import BM25Okapi
import jieba

class BM25Retriever:
    def __init__(self, documents):
        self.documents = documents
        self.tokenized_corpus = [self.tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def tokenize(self, text):
        # 使用jieba进行中文分词
        return list(jieba.cut_for_search(text))
    
    def retrieve(self, query, k=5):
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # 获取top-k文档
        top_indices = scores.argsort()[-k:][::-1]
        top_documents = [self.documents[i] for i in top_indices]
        top_scores = [scores[i] for i in top_indices]
        
        return list(zip(top_documents, top_scores))
```

### 3. 实现混合检索系统

**文件修改**：`Rag_Retrieved.py`

**功能实现**：

* 集成语义检索和BM25关键词检索

* 实现元数据过滤

* 实现混合检索结果的合并与去重

**核心代码**：

```python
def hybrid_retrieval(query, semantic_retriever, bm25_retriever, metadata_filters=None, k=10):
    # 语义检索
    semantic_results = semantic_retriever.get_relevant_documents(query, k=k)
    
    # BM25检索
    bm25_results_with_scores = bm25_retriever.retrieve(query, k=k)
    bm25_results = [doc for doc, score in bm25_results_with_scores]
    
    # 元数据过滤
    if metadata_filters:
        semantic_results = [doc for doc in semantic_results if self._match_metadata(doc.metadata, metadata_filters)]
        bm25_results = [doc for doc in bm25_results if self._match_metadata(doc.metadata, metadata_filters)]
    
    # 合并结果（去重）
    combined_results = []
    seen_ids = set()
    
    # 先添加语义检索结果
    for doc in semantic_results:
        doc_id = f"{doc.metadata.get('source', '')}_{doc.metadata.get('row_index', '')}"
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            combined_results.append((doc, 'semantic'))
    
    # 再添加BM25检索结果
    for doc in bm25_results:
        doc_id = f"{doc.metadata.get('source', '')}_{doc.metadata.get('row_index', '')}"
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            combined_results.append((doc, 'bm25'))
    
    return combined_results
```

### 4. 实现结果重排序

**文件创建**：`reranker.py`

**功能实现**：

* 实现基于BERT的重排序模型

* 支持语义相关性评分

* 结合元数据进行综合评分

**核心代码**：

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

class Reranker:
    def __init__(self, model_name="bert-base-chinese"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)
        self.model.eval()
    
    def rerank(self, query, documents, k=5):
        scores = []
        
        for doc in documents:
            # 构建输入
            inputs = self.tokenizer(
                query,
                doc.page_content,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # 获取相关性分数
            with torch.no_grad():
                outputs = self.model(**inputs)
                score = outputs.logits.item()
            
            scores.append((doc, score))
        
        # 按分数排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回top-k结果
        return [doc for doc, score in scores[:k]]
```

### 5. 实现完整的混合检索流程

**文件修改**：`Rag_Retrieved.py`

**功能实现**：

* 整合向量库加载、BM25检索器初始化、混合检索和结果重排序

* 提供统一的检索接口

**核心代码**：

```python
def create_hybrid_retriever():
    # 加载向量存储
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh", model_kwargs={'device': 'cpu'})
    case_causes_vector_store = FAISS.load_local("case_causes_vector_store", embeddings, allow_dangerous_deserialization=True)
    
    # 创建语义检索器
    semantic_retriever = case_causes_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    
    # 初始化BM25检索器
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
    # 混合检索
    combined_results = hybrid_retrieval(query, semantic_retriever, bm25_retriever, metadata_filters, k=20)
    
    # 提取文档列表
    documents = [doc for doc, source in combined_results]
    
    # 重排序
    reranked_results = reranker.rerank(query, documents, k=k)
    
    return reranked_results
```

### 6. 测试与优化

**文件创建**：`test_hybrid_search.py`

**功能实现**：

* 测试混合检索的效果

* 比较不同检索方式的性能

* 优化检索参数

**核心代码**：

```python
from Rag_Retrieved import create_hybrid_retriever, hybrid_search

# 创建检索器
semantic_retriever, bm25_retriever, reranker = create_hybrid_retriever()

# 测试查询
queries = [
    "未取得医疗机构执业许可证擅自执业",
    "公共场所未取得卫生许可证擅自营业",
    "安排未获得有效健康合格证明的从业人员"
]

# 执行混合检索
for query in queries:
    print(f"\n查询：{query}")
    results = hybrid_search(query, semantic_retriever, bm25_retriever, reranker, k=3)
    
    for i, result in enumerate(results):
        print(f"\n结果 {i+1}：")
        print(f"案由名称：{result.metadata['案由名称']}")
        print(f"所属专业：{result.metadata['所属专业']}")
        print(f"适用法律法规：{result.metadata['适用的法律法规']}")
        print(f"内容：{result.page_content[:200]}...")
```

## 三、技术栈

* **嵌入模型**：BGE-large-zh

* **向量数据库**：FAISS

* **关键词检索**：rank\_bm25

* **中文分词**：jieba

* **重排序模型**：bert-base-chinese

* **框架**：LangChain

## 四、预期效果

1. 提高检索准确性：结合语义和关键词检索，覆盖更多相关结果
2. 增强结果相关性：通过重排序模型优化结果顺序
3. 支持元数据过滤：可以根据所属专业、适用法律法规等进行筛选
4. 提高检索效率：通过混合检索和重排序，平衡准确性和性能

## 五、后续优化方向

1. 优化BM25参数，提高关键词检索效果
2. 训练专门的重排序模型，适应案由检索场景
3. 实现增量更新机制，支持向量库的动态更新
4. 优化检索性能，支持大规模向量库的高效检索
5. 添加用户反馈机制，持续优化检索效果


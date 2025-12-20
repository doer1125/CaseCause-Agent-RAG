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

# 测试带元数据过滤的检索
print(f"\n\n=== 带元数据过滤的检索 ===")
query = "未取得医疗机构执业许可证擅自执业"
metadata_filters = {"所属专业": "医疗机构"}
results = hybrid_search(query, semantic_retriever, bm25_retriever, reranker, metadata_filters=metadata_filters, k=3)

print(f"\n查询：{query} (过滤条件：{metadata_filters})")
for i, result in enumerate(results):
    print(f"\n结果 {i+1}：")
    print(f"案由名称：{result.metadata['案由名称']}")
    print(f"所属专业：{result.metadata['所属专业']}")
    print(f"适用法律法规：{result.metadata['适用的法律法规']}")
    print(f"内容：{result.page_content[:200]}...")
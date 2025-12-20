from Rag_Retrieved import create_hybrid_retriever, hybrid_search, create_retrievers

def test_hybrid_search():
    """
    测试混合检索效果
    """
    print("正在初始化混合检索器...")
    
    # 创建混合检索器
    semantic_retriever, bm25_retriever, reranker = create_hybrid_retriever()
    
    # 测试查询
    queries = [
        "未取得医疗机构执业许可证擅自执业",
        "公共场所未取得卫生许可证擅自营业",
        "安排未获得有效健康合格证明的从业人员",
        "未按照规定对顾客用品用具进行清洗、消毒、保洁"
    ]
    
    # 执行混合检索
    for query in queries:
        print(f"\n{'='*50}")
        print(f"查询：{query}")
        print(f"{'='*50}")
        
        # 混合检索
        results = hybrid_search(query, semantic_retriever, bm25_retriever, reranker, k=3)
        
        print(f"混合检索结果（前3条）：")
        for i, result in enumerate(results):
            print(f"\n结果 {i+1}：")
            print(f"案由名称：{result.metadata['案由名称']}")
            print(f"所属专业：{result.metadata['所属专业']}")
            print(f"适用法律法规：{result.metadata['适用的法律法规']}")
            print(f"内容：{result.page_content[:200]}...")
        
        print(f"{'='*50}")

def compare_retrieval_methods():
    """
    比较不同检索方式的效果
    """
    print("正在比较不同检索方式...")
    
    # 创建检索器
    semantic_retriever, bm25_retriever, reranker = create_hybrid_retriever()
    case_causes_retriever, _ = create_retrievers()
    
    # 测试查询
    query = "未取得医疗机构执业许可证擅自执业"
    
    print(f"\n{'='*60}")
    print(f"比较不同检索方式：{query}")
    print(f"{'='*60}")
    
    # 1. 仅语义检索
    print(f"\n1. 仅语义检索结果：")
    semantic_results = case_causes_retriever.invoke(query)[:3]
    for i, result in enumerate(semantic_results):
        print(f"\n结果 {i+1}：")
        print(f"案由名称：{result.metadata['案由名称']}")
    
    # 2. 仅BM25检索
    print(f"\n2. 仅BM25检索结果：")
    bm25_results_with_scores = bm25_retriever.retrieve(query, k=3)
    for i, (result, score) in enumerate(bm25_results_with_scores):
        print(f"\n结果 {i+1}（BM25分数：{score:.4f}）：")
        print(f"案由名称：{result.metadata['案由名称']}")
    
    # 3. 混合检索（语义+BM25+重排序）
    print(f"\n3. 混合检索结果：")
    hybrid_results = hybrid_search(query, semantic_retriever, bm25_retriever, reranker, k=3)
    for i, result in enumerate(hybrid_results):
        print(f"\n结果 {i+1}：")
        print(f"案由名称：{result.metadata['案由名称']}")
        print(f"所属专业：{result.metadata['所属专业']}")
        print(f"适用法律法规：{result.metadata['适用的法律法规']}")
    
    print(f"{'='*60}")

def test_with_metadata_filter():
    """
    测试带元数据过滤的混合检索
    """
    print("正在测试带元数据过滤的混合检索...")
    
    # 创建混合检索器
    semantic_retriever, bm25_retriever, reranker = create_hybrid_retriever()
    
    # 测试查询
    query = "擅自执业"
    
    # 元数据过滤条件：所属专业为"医疗机构"
    metadata_filters = {"所属专业": "医疗机构"}
    
    print(f"\n{'='*60}")
    print(f"带元数据过滤的混合检索：{query}")
    print(f"元数据过滤条件：{metadata_filters}")
    print(f"{'='*60}")
    
    # 执行带元数据过滤的混合检索
    results = hybrid_search(query, semantic_retriever, bm25_retriever, reranker, metadata_filters=metadata_filters, k=3)
    
    print(f"过滤后的混合检索结果（前3条）：")
    for i, result in enumerate(results):
        print(f"\n结果 {i+1}：")
        print(f"案由名称：{result.metadata['案由名称']}")
        print(f"所属专业：{result.metadata['所属专业']}")
        print(f"适用法律法规：{result.metadata['适用的法律法规']}")
        print(f"内容：{result.page_content[:200]}...")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    # 测试1：基础混合检索
    test_hybrid_search()
    
    # 测试2：比较不同检索方式
    compare_retrieval_methods()
    
    # 测试3：带元数据过滤的混合检索
    test_with_metadata_filter()
    
    print("\n所有测试完成！")

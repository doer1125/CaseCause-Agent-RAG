import re
import jieba
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from bm25_retriever import BM25Retriever
from reranker import Reranker
from Rag_Retrieved import create_hybrid_retriever, hybrid_search

class ComplaintAnalyzer:
    def __init__(self):
        # 初始化规则引擎的关键词列表
        self.keyword_patterns = {
            "非法行医": ["非法行医", "无证行医", "黑诊所", "未取得医疗机构执业许可证"],
            "服务态度": ["态度不耐烦", "态度差", "服务不好", "不耐烦"],
            "乱收费": ["乱收费", "不合理收费", "未提供服务却收费", "多收费"],
            "卫生问题": ["卫生差", "卫生状况极差", "异味刺鼻", "污水横流"],
            "药品回扣": ["药品回扣", "利益输送", "医药代表", "异常用药"]
        }
        
        # 初始化AI引擎
        self.tokenizer = None
        self.model = None
        self.embeddings = HuggingFaceEmbeddings(
            model_name="local_models/bge-large-zh",
            model_kwargs={'device': 'cpu'}
        )
        
        # 初始化混合检索器
        self.semantic_retriever, self.bm25_retriever, self.reranker = create_hybrid_retriever()
    
    def rule_based_extraction(self, complaint):
        """
        基于规则的投诉要点提取
        
        Args:
            complaint: 投诉文本
            
        Returns:
            提取的投诉要点字典
        """
        # 提取投诉类型
        complaint_type = []
        for type_name, keywords in self.keyword_patterns.items():
            for keyword in keywords:
                if keyword in complaint:
                    complaint_type.append(type_name)
                    break
        
        # 提取涉及的机构/人员
        # 使用正则表达式提取人名（假设为X姓或XX医生格式）
        persons = re.findall(r'[\u4e00-\u9fa5]{2,4}(?:医生|医师|主任|副主任|护士)', complaint)
        # 提取机构名称
        institutions = re.findall(r'(?:XX|YY|ZZ|AA|BB)[\u4e00-\u9fa5]{0,4}(?:医院|诊所|大厦|物业|卫健委)', complaint)
        
        # 提取时间
        times = re.findall(r'(?:上周|昨天|今天|明天|后天|\d{4}-\d{2}-\d{2}|\d{1,2}月\d{1,2}日|\d{1,2}日)', complaint)
        
        # 提取地点
        locations = re.findall(r'(?:XX|YY|ZZ|AA|BB)[\u4e00-\u9fa5]{0,4}(?:路|街|巷|号|室|楼|座|区|县)', complaint)
        
        # 提取具体问题
        issues = []
        if "非法行医" in complaint_type:
            issues.append("未取得医疗机构执业许可证擅自执业")
        if "服务态度" in complaint_type:
            issues.append("医务人员服务态度恶劣")
        if "乱收费" in complaint_type:
            issues.append("医疗机构乱收费")
        if "卫生问题" in complaint_type:
            issues.append("公共场所卫生状况不符合要求")
        if "药品回扣" in complaint_type:
            issues.append("医务人员收受药品回扣")
        
        return {
            "complaint_type": complaint_type,
            "persons": persons,
            "institutions": institutions,
            "times": times,
            "locations": locations,
            "issues": issues,
            "raw_complaint": complaint
        }
    
    def ai_based_extraction(self, complaint):
        """
        基于AI的投诉要点提取
        
        Args:
            complaint: 投诉文本
            
        Returns:
            提取的投诉要点文本
        """
        # 使用向量模型生成投诉的语义表示
        # 这里简化处理，直接使用投诉文本作为检索查询
        # 实际应用中可以使用更复杂的AI模型进行要点提取
        
        # 分词处理，提取关键词
        seg_list = jieba.cut(complaint)
        keywords = [word for word in seg_list if len(word) > 1]
        
        # 去重并排序
        keywords = list(set(keywords))
        keywords.sort()
        
        return keywords
    
    def extract_complaint_points(self, complaint):
        """
        结合规则引擎和AI引擎提取投诉要点
        
        Args:
            complaint: 投诉文本
            
        Returns:
            综合提取的投诉要点
        """
        # 基于规则提取要点
        rule_based_result = self.rule_based_extraction(complaint)
        
        # 基于AI提取关键词
        ai_keywords = self.ai_based_extraction(complaint)
        
        # 综合结果
        combined_issues = rule_based_result["issues"]
        
        # 如果规则提取没有找到问题类型，使用AI关键词进行补充
        if not combined_issues:
            combined_issues = [" ".join(ai_keywords[:5])]
        
        # 将AI关键词添加到综合结果中
        rule_based_result["ai_keywords"] = ai_keywords
        
        return rule_based_result
    
    def retrieve_case_causes(self, complaint_points, k=3):
        """
        根据投诉要点检索相应的案由
        
        Args:
            complaint_points: 投诉要点字典
            k: 返回结果数量
            
        Returns:
            检索到的案由列表
        """
        # 构建更精确的检索查询
        
        # 优先使用具体问题作为主要查询
        main_queries = []
        if complaint_points["issues"]:
            main_queries = complaint_points["issues"]
        
        # 准备检索结果列表
        all_retrieved_causes = []
        
        # 对每个主要问题类型进行单独检索，然后合并结果
        for main_query in main_queries:
            # 构建查询
            query = main_query
            
            # 使用混合检索获取案由，增加k值以获取更多候选结果
            retrieved_causes = hybrid_search(
                query, 
                self.semantic_retriever, 
                self.bm25_retriever, 
                self.reranker, 
                k=k*2
            )
            
            # 添加到结果列表
            all_retrieved_causes.extend(retrieved_causes)
        
        # 如果没有主要问题类型，使用原始方式构建查询
        if not main_queries:
            query_parts = []
            if complaint_points["ai_keywords"]:
                query_parts.extend(complaint_points["ai_keywords"][:10])
            query = " ".join(query_parts)
            
            retrieved_causes = hybrid_search(
                query, 
                self.semantic_retriever, 
                self.bm25_retriever, 
                self.reranker, 
                k=k
            )
            all_retrieved_causes.extend(retrieved_causes)
        
        # 结果去重
        unique_causes = []
        seen_ids = set()
        for cause in all_retrieved_causes:
            cause_id = f"{cause.metadata.get('案由名称', '')}_{cause.metadata.get('适用的法律法规', '')}"
            if cause_id not in seen_ids:
                seen_ids.add(cause_id)
                unique_causes.append(cause)
        
        # 重新排序，优先保留与投诉类型相关的案由
        sorted_causes = []
        remaining_causes = []
        
        # 优先保留与投诉类型相关的案由
        for cause in unique_causes:
            relevance = False
            for complaint_type in complaint_points["complaint_type"]:
                # 检查案由名称或内容是否包含投诉类型关键词
                if complaint_type in cause.metadata.get("案由名称", "") or complaint_type in cause.page_content:
                    sorted_causes.append(cause)
                    relevance = True
                    break
            if not relevance:
                remaining_causes.append(cause)
        
        # 补充剩余结果
        sorted_causes.extend(remaining_causes)
        
        # 返回指定数量的结果
        return sorted_causes[:k]
    
    def analyze_complaint(self, complaint):
        """
        完整的投诉分析流程：提取要点 -> 检索案由
        
        Args:
            complaint: 投诉文本
            
        Returns:
            分析结果，包括投诉要点和检索到的案由
        """
        # 提取投诉要点
        complaint_points = self.extract_complaint_points(complaint)
        
        # 检索案由
        retrieved_causes = self.retrieve_case_causes(complaint_points)
        
        # 构建分析结果
        analysis_result = {
            "complaint_points": complaint_points,
            "retrieved_causes": retrieved_causes
        }
        
        return analysis_result

# 测试代码
if __name__ == "__main__":
    # 读取投诉文件
    with open("complaints.txt", "r", encoding="utf-8") as f:
        complaints = f.readlines()
    
    # 初始化投诉分析器
    analyzer = ComplaintAnalyzer()
    
    # 分析每个投诉
    for i, complaint in enumerate(complaints):
        print(f"\n=== 投诉 {i+1} 分析结果 ===")
        print(f"原始投诉：{complaint.strip()}")
        
        analysis_result = analyzer.analyze_complaint(complaint)
        
        # 输出投诉要点
        print("\n投诉要点：")
        points = analysis_result["complaint_points"]
        print(f"  投诉类型：{points['complaint_type']}")
        print(f"  涉及人员：{points['persons']}")
        print(f"  涉及机构：{points['institutions']}")
        print(f"  涉及时间：{points['times']}")
        print(f"  涉及地点：{points['locations']}")
        print(f"  具体问题：{points['issues']}")
        print(f"  AI关键词：{points['ai_keywords'][:10]}...")
        
        # 输出检索到的案由
        print("\n检索到的案由：")
        causes = analysis_result["retrieved_causes"]
        for j, cause in enumerate(causes):
            print(f"\n  案由 {j+1}：")
            print(f"    案由名称：{cause.metadata['案由名称']}")
            print(f"    所属专业：{cause.metadata['所属专业']}")
            print(f"    适用法律法规：{cause.metadata['适用的法律法规']}")
            print(f"    内容：{cause.page_content[:200]}...")
        
        print("\n" + "="*50)
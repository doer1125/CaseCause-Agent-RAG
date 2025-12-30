import re
from typing import List, Dict, Any

class IntentRecognizer:
    def __init__(self):
        # 意图规则配置
        self.intent_patterns = {
            "complaint_description": [
                r"投诉", r"举报", r"反映问题", r"遇到问题", r"有个情况",
                r"我要投诉", r"我想举报", r"我要反映"
            ],
            "points_confirmation": [
                r"确认", r"是的", r"没错", r"对的", r"正确",
                r"调整", r"修改", r"补充", r"需要修改", r"不太对"
            ],
            "case_retrieval": [
                r"案由", r"什么案由", r"相关案由", r"检索案由"
            ],
            "legal_retrieval": [
                r"法律法规", r"相关法律", r"法律依据", r"检索法律",
                r"适用法律", r"法律条款"
            ],
            "document_generation": [
                r"生成文书", r"文书", r"投诉书", r"举报信",
                r"申请函", r"需要文书", r"制作文书"
            ],
            "clarification": [
                r"什么是", r"为什么", r"怎么", r"如何", r"解释",
                r"什么意思", r"不太明白", r"不清楚"
            ],
            "exit": [
                r"结束", r"退出", r"再见", r"完成", r"结束对话"
            ]
        }
        
        # 关键词权重配置
        self.keyword_weights = {
            "complaint_description": 2.0,
            "points_confirmation": 1.5,
            "case_retrieval": 1.8,
            "legal_retrieval": 1.8,
            "document_generation": 2.0,
            "clarification": 1.0,
            "exit": 1.2
        }
    
    def rule_based_intent_recognition(self, query: str) -> List[Dict[str, Any]]:
        """
        基于规则的意图识别
        
        Args:
            query: 用户查询文本
            
        Returns:
            意图识别结果列表，包含意图类型和置信度
        """
        results = []
        
        # 遍历所有意图模式
        for intent, patterns in self.intent_patterns.items():
            score = 0.0
            
            # 检查每个模式是否匹配
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    score += 1.0
            
            # 计算置信度
            if score > 0:
                confidence = (score / len(patterns)) * self.keyword_weights[intent]
                results.append({
                    "intent": intent,
                    "confidence": min(1.0, confidence),
                    "method": "rule_based"
                })
        
        # 按置信度排序
        results.sort(key=lambda x: x["confidence"], reverse=True)
        
        return results
    
    def ai_based_intent_recognition(self, query: str) -> List[Dict[str, Any]]:
        """
        基于AI的意图识别
        
        Args:
            query: 用户查询文本
            
        Returns:
            意图识别结果列表，包含意图类型和置信度
        """
        try:
            import ollama
            
            # 设计提示模板
            prompt = f"""请识别以下用户查询的意图，从以下列表中选择最匹配的意图：
1. complaint_description: 用户描述投诉内容
2. points_confirmation: 用户确认或修改投诉要点
3. case_retrieval: 用户请求检索相关案由
4. legal_retrieval: 用户请求检索相关法律法规
5. document_generation: 用户请求生成文书
6. clarification: 用户请求解释或澄清
7. exit: 用户希望结束对话

请以JSON格式输出，包含intent（意图类型）和confidence（置信度，0-1之间）。

用户查询：{query}
"""
            
            # 调用本地Ollama模型
            response = ollama.chat(
                model="qwen2.5:7b-instruct",
                messages=[
                    {
                        "role": "system",
                        "content": "你是一名专业的对话意图识别专家，擅长识别用户在投诉处理系统中的意图。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                format="json"
            )
            
            # 解析模型输出
            import json
            ai_result = json.loads(response["message"]["content"])
            
            return [{
                "intent": ai_result["intent"],
                "confidence": ai_result["confidence"],
                "method": "ai_based"
            }]
        except Exception as e:
            print(f"AI意图识别失败：{e}")
            # 失败时回退到规则识别
            return self.rule_based_intent_recognition(query)
    
    def recognize_intent(self, query: str, use_ai: bool = False) -> Dict[str, Any]:
        """
        综合意图识别，结合规则和AI
        
        Args:
            query: 用户查询文本
            use_ai: 是否使用AI识别
            
        Returns:
            最终意图识别结果，包含意图类型和置信度
        """
        # 规则识别结果
        rule_results = self.rule_based_intent_recognition(query)
        
        if use_ai:
            try:
                # AI识别结果
                ai_results = self.ai_based_intent_recognition(query)
                
                # 综合结果
                combined_results = rule_results + ai_results
                
                # 合并相同意图的结果，取最高置信度
                intent_scores = {}
                for result in combined_results:
                    intent = result["intent"]
                    confidence = result["confidence"]
                    if intent not in intent_scores or confidence > intent_scores[intent]:
                        intent_scores[intent] = confidence
                
                # 转换为列表并排序
                final_results = [
                    {"intent": intent, "confidence": score}
                    for intent, score in intent_scores.items()
                ]
                final_results.sort(key=lambda x: x["confidence"], reverse=True)
            except Exception as e:
                # AI识别失败，回退到规则识别
                print(f"AI意图识别失败，回退到规则识别：{e}")
                final_results = rule_results
        else:
            final_results = rule_results
        
        # 返回最高置信度的意图
        if final_results:
            return final_results[0]
        else:
            # 默认意图
            return {"intent": "unknown", "confidence": 0.5}
    
    def get_intent_suggestion(self, intent: str, dialogue_state: Dict[str, Any]) -> str:
        """
        根据意图和对话状态生成系统提示建议
        
        Args:
            intent: 识别到的意图
            dialogue_state: 当前对话状态
            
        Returns:
            系统提示建议
        """
        suggestions = {
            "complaint_description": "请详细描述您的投诉内容，包括时间、地点、涉及人员和具体问题。",
            "points_confirmation": "",
            "case_retrieval": "",
            "legal_retrieval": "",
            "document_generation": "",
            "clarification": "",
            "exit": "感谢您的使用，祝您工作顺利！"
        }
        
        # 根据对话状态和意图生成更具体的建议
        current_task = dialogue_state.get("current_task", "initial")
        
        if intent == "points_confirmation":
            if current_task == "points_extraction":
                suggestions["points_confirmation"] = "您对提取的投诉要点有什么需要调整或补充的吗？"
        elif intent == "case_retrieval":
            if dialogue_state.get("confirmed"):
                suggestions["case_retrieval"] = "正在为您检索相关案由..."
            else:
                suggestions["case_retrieval"] = "请先确认投诉要点，然后再进行案由检索。"
        elif intent == "legal_retrieval":
            if dialogue_state.get("confirmed"):
                suggestions["legal_retrieval"] = "正在为您检索相关法律法规..."
            else:
                suggestions["legal_retrieval"] = "请先确认投诉要点，然后再进行法律法规检索。"
        elif intent == "document_generation":
            if dialogue_state.get("confirmed"):
                suggestions["document_generation"] = "请告诉我您需要生成哪种类型的文书？（如：投诉书、举报信等）"
            else:
                suggestions["document_generation"] = "请先确认投诉要点，然后再进行文书生成。"
        
        return suggestions[intent]

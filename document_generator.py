from typing import Dict, Any, Optional
import os
from generate_pdf_document import HealthDocumentGenerator

class DocumentGenerator:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or "./qwen2.5-7b-health-llamafactory"
        self.generator = None
        self.initialized = False
        
        # 暂时禁用微调模型加载，加快启动速度
        print("已禁用微调模型加载，将使用简化模式生成文书")
    
    def generate_document(self, document_type: str, complaint_points: Dict[str, Any], additional_params: Optional[Dict[str, Any]] = None) -> str:
        """
        根据投诉要点生成相应文书
        
        Args:
            document_type: 文书类型（如：投诉书、举报信等）
            complaint_points: 投诉要点
            additional_params: 额外参数（如：投诉人信息、投诉对象信息等）
            
        Returns:
            生成的文书内容
        """
        # 构建文书生成指令
        instruction = self._build_instruction(document_type, complaint_points, additional_params)
        
        if self.initialized and self.generator:
            # 使用微调模型生成文书
            return self.generator.generate_document(instruction)
        else:
            # 使用简化模式生成文书
            return self._generate_simple_document(document_type, complaint_points, additional_params)
    
    def _build_instruction(self, document_type: str, complaint_points: Dict[str, Any], additional_params: Optional[Dict[str, Any]]) -> str:
        """
        构建文书生成指令
        
        Args:
            document_type: 文书类型
            complaint_points: 投诉要点
            additional_params: 额外参数
            
        Returns:
            构建好的生成指令
        """
        instruction = f"请生成一份{document_type}，基于以下投诉要点：\n\n"
        
        # 添加投诉要点
        if complaint_points.get("complaint_type"):
            instruction += f"投诉类型：{', '.join(complaint_points['complaint_type'])}\n"
        if complaint_points.get("issues"):
            instruction += f"具体问题：{', '.join(complaint_points['issues'])}\n"
        if complaint_points.get("persons"):
            instruction += f"涉及人员：{', '.join(complaint_points['persons'])}\n"
        if complaint_points.get("institutions"):
            instruction += f"涉及机构：{', '.join(complaint_points['institutions'])}\n"
        if complaint_points.get("times"):
            instruction += f"涉及时间：{', '.join(complaint_points['times'])}\n"
        if complaint_points.get("locations"):
            instruction += f"涉及地点：{', '.join(complaint_points['locations'])}\n"
        
        # 添加额外参数
        if additional_params:
            instruction += "\n额外信息：\n"
            for key, value in additional_params.items():
                instruction += f"{key}：{value}\n"
        
        instruction += f"\n请严格按照{document_type}的规范格式撰写，语言正式、严谨。"
        
        return instruction
    
    def _generate_simple_document(self, document_type: str, complaint_points: Dict[str, Any], additional_params: Optional[Dict[str, Any]]) -> str:
        """
        简化模式生成文书
        
        Args:
            document_type: 文书类型
            complaint_points: 投诉要点
            additional_params: 额外参数
            
        Returns:
            生成的文书内容
        """
        # 基础文书模板
        templates = {
            "投诉书": """
{document_type}

投诉人：{complainant_name}
联系电话：{contact_phone}
联系地址：{contact_address}

投诉对象：{institutions}

投诉事项：
{issues}

投诉事实与理由：
{complaint_details}

请求事项：
1. 请依法查处上述违法行为
2. 请将处理结果书面告知投诉人

投诉人（签名）：{complainant_name}
日期：{date}
            """,
            "举报信": """
{document_type}

举报人：{complainant_name}
联系电话：{contact_phone}
联系地址：{contact_address}

举报对象：{institutions}

举报事项：
{issues}

举报事实与理由：
{complaint_details}

请求事项：
1. 请依法查处上述违法行为
2. 请对举报人的身份信息予以保密

举报人（签名）：{complainant_name}
日期：{date}
            """
        }
        
        # 使用默认模板
        template = templates.get(document_type, """{document_type}\n\n{complaint_details}\n""")
        
        # 准备模板参数
        params = {
            "document_type": document_type,
            "complainant_name": additional_params.get("complainant_name", "匿名"),
            "contact_phone": additional_params.get("contact_phone", "") if additional_params else "",
            "contact_address": additional_params.get("contact_address", "") if additional_params else "",
            "institutions": ", ".join(complaint_points.get("institutions", ["相关机构"])),
            "issues": ", ".join(complaint_points.get("issues", [])),
            "date": additional_params.get("date", "") if additional_params else "",
            "complaint_details": "\n".join([
                f"• {item}" for item in complaint_points.get("issues", [])
            ])
        }
        
        # 生成文书
        return template.format(**params).strip()
    
    def save_document(self, content: str, output_path: str, format: str = "txt") -> str:
        """
        保存生成的文书
        
        Args:
            content: 文书内容
            output_path: 输出路径
            format: 文件格式（txt, pdf）
            
        Returns:
            保存后的文件路径
        """
        if format == "pdf" and self.initialized and self.generator:
            # 使用微调模型的PDF生成功能
            return self.generator.generate_pdf(content, output_path)
        else:
            # 保存为文本文件
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            return output_path
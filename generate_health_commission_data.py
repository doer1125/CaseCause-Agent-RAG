import json
import random
from datetime import datetime, timedelta

# 扩展的主题列表
urgent_topics = ["紧急", "重要", "专项", "年度", "季度", "月度"]
medical_institutions = ["综合医院", "专科医院", "社区卫生服务中心", "乡镇卫生院", "民营医院", "妇幼保健院"]
health_topics = ["传染病防控", "医疗废物管理", "爱国卫生运动", "食品安全监管", "职业病防治", 
                 "无偿献血宣传", "疫苗接种管理", "突发事件医疗救援", "医疗卫生行风建设", "医疗质量安全",
                 "中医药发展", "老年健康服务", "儿童健康促进", "精神卫生服务", "基本公共卫生服务",
                 "医疗卫生信息化", "医疗卫生人才培养", "医疗卫生体制改革", "医疗保障制度", "健康促进与教育"]

device_types = ["医疗设备", "检验设备", "影像设备", "急救设备", "康复设备", "消毒设备"]
building_types = ["社区卫生服务中心", "发热门诊", "核酸检测实验室", "疫苗接种点", "急救中心", "老年护理院"]
training_types = ["医疗卫生技术", "传染病防控", "医疗质量管理", "医院感染控制", "急救技能", "中医药适宜技术"]
new_tech_types = ["人工智能辅助诊断", "远程医疗", "精准医疗", "微创手术", "再生医学", "基因检测"]
fund_types = ["传染病防控", "医疗设备购置", "基层医疗建设", "人才培养", "疫情应急处置", "公共卫生服务"]
award_types = ["卫生健康工作先进", "疫情防控先进", "医疗质量安全先进", "无偿献血先进", "医德医风先进",
              "科研创新先进", "基层卫生先进", "中医药工作先进", "健康促进先进", "卫生信息化先进"]

# 公文类型模板
document_templates = {
    "通知": {
        "instruction": "撰写一份关于{urgent}{year}年度{topic}{institution}的通知",
        "structure": [
            "检查范围", "检查内容", "检查时间", "检查方式", "工作要求"
        ]
    },
    "通报": {
        "instruction": "起草一份关于{urgent}{topic}工作情况的通报",
        "structure": [
            "基本情况", "存在问题", "工作要求", "下一步工作"
        ]
    },
    "请示": {
        "instruction": "撰写一份关于{urgent}申请{action}{type}的请示",
        "structure": [
            "申请理由", "基本情况", "资金来源", "保障措施", "预期效果"
        ]
    },
    "批复": {
        "instruction": "起草一份关于同意{action}{type}的批复",
        "structure": [
            "批复事项", "具体要求", "实施步骤", "保障措施"
        ]
    },
    "决定": {
        "instruction": "撰写一份关于{urgent}表彰{year}年度{award}集体和个人的决定",
        "structure": [
            "表彰对象", "表彰理由", "希望和要求"
        ]
    }
}

# 随机生成日期
def random_date(start_year=2023, end_year=2025):
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    return start_date + timedelta(days=random_days)

# 生成随机编号
def random_number(prefix="", length=2):
    return f"{prefix}{random.randint(10**(length-1), 10**length-1)}"

# 生成公文内容
def generate_document_content(doc_type, params):
    date = random_date()
    year = date.year
    month = date.month
    day = date.day
    
    if doc_type == "通知":
        urgent = params.get('urgent', '')
        year = params.get('year', year)
        topic = params.get('topic', '')
        institution = params.get('institution', '')
        return f"关于开展{urgent}{year}年度{topic}{institution}的通知\n\n各市、县（区）卫生健康委员会，各委属医疗机构：\n\n为进一步加强{topic}管理，规范{topic}行为，保障{topic}工作质量和安全，根据《中华人民共和国基本医疗卫生与健康促进法》《医疗机构管理条例》等法律法规要求，我委决定在全市范围内开展{urgent}{year}年度{topic}{institution}。现将有关事项通知如下：\n\n一、检查范围\n全市所有取得《医疗机构执业许可证》的各级各类医疗机构，重点检查二级及以上公立医院、社会办医疗机构、基层医疗机构和专科医疗机构。\n\n二、检查内容\n（一）医疗机构依法执业情况；\n（二）医疗质量和医疗安全管理情况；\n（三）医疗服务收费情况；\n（四）医疗机构感染防控情况；\n（五）医德医风建设情况；\n（六）其他依法需要检查的事项。\n\n三、检查时间\n{year}年5月1日至{year}年10月31日，分三个阶段进行：\n（一）自查自纠阶段（5月1日至6月30日）；\n（二）集中检查阶段（7月1日至9月30日）；\n（三）总结整改阶段（10月1日至10月31日）。\n\n四、检查方式\n采取现场检查、资料审查、数据分析、社会调查等相结合的方式进行。\n\n五、工作要求\n（一）高度重视，加强领导；\n（二）严格标准，依法检查；\n（三）强化整改，注重实效；\n（四）严肃纪律，公正执法。\n\n请各单位按照通知要求，认真组织实施，确保专项检查工作顺利完成。检查结束后，各设区市卫生健康委要将专项检查工作总结于{year}年11月15日前报送我委医政医管处。\n\n联系人：XXX，联系电话：XXXX-XXXXXXXX\n\n附件：1. {urgent}{year}年度{topic}{institution}评分标准\n2. {urgent}{year}年度{topic}{institution}自查表\n\nXX市卫生健康委员会\n{year}年4月15日"
    
    elif doc_type == "通报":
        urgent = params.get('urgent', '')
        topic = params.get('topic', '')
        return f"关于近期我市{urgent}{topic}工作情况的通报\n\n各市、县（区）卫生健康委员会，各委属医疗机构：\n\n近期，我市开展了{urgent}{topic}工作，为进一步加强{topic}管理，保障人民群众身体健康和生命安全，现将有关情况通报如下：\n\n一、基本情况\n截至{year}年{month}月{day}日，我市共完成{topic}工作任务X项，其中完成率达到X%，主要涉及{random.choice(['传染病防控', '医疗废物管理', '爱国卫生运动', '食品安全监管'])}等方面。\n\n二、存在问题\n（一）部分医疗机构工作落实不及时、不到位；\n（二）个别基层医疗机构制度执行不严格；\n（三）部分单位宣传培训工作不足；\n（四）公众参与意识有待提高。\n\n三、工作要求\n（一）加强组织领导；\n（二）规范工作流程；\n（三）强化监督检查；\n（四）加大宣传力度；\n（五）做好总结反馈。\n\n四、下一步工作\n（一）组织开展专项督导检查；\n（二）举办专业知识培训；\n（三）加强部门间协作配合；\n（四）建立长效工作机制。\n\n请各单位高度重视{urgent}{topic}工作，切实加强组织领导，落实各项工作措施，确保我市{topic}工作取得实效。\n\nXX市卫生健康委员会\n{year}年{month}月{day}日"
    
    elif doc_type == "请示":
        urgent = params.get('urgent', '')
        action = params.get('action', '')
        type = params.get('type', '')
        return f"关于{urgent}申请{action}{type}的请示\n\n省卫生健康委员会：\n\n为进一步提高我市医疗服务能力，满足人民群众日益增长的医疗需求，根据《XX省{type}管理办法》的有关规定，结合我市实际情况，现将我市{urgent}申请{action}{type}的有关事项请示如下：\n\n一、申请理由\n（一）提升医疗服务能力的需要；\n（二）满足群众医疗需求的需要；\n（三）促进医疗事业发展的需要。\n\n二、基本情况\n本次申请{action}{type}共X项，总价值约X万元，主要包括：\n1. {type} X项，价值约X万元；\n2. 相关配套设施 X项，价值约X万元；\n3. 其他相关费用 X万元。\n\n三、资金来源\n本次申请{action}{type}的资金来源为：\n1. 中央财政转移支付资金 X万元；\n2. 地方财政配套资金 X万元；\n3. 医疗机构自筹资金 X万元。\n\n四、保障措施\n（一）加强组织领导；\n（二）严格审批程序；\n（三）加强管理和监督；\n（四）加强人员培训。\n\n五、预期效果\n通过{action}{type}，预计将使我市医疗服务能力得到显著提升，能够更好地满足人民群众的医疗需求，促进我市医疗事业的发展。\n\n以上请示妥否，请批示。\n\n附件：1. {action}{type}清单\n2. 资金预算表\n3. 可行性论证报告\n\nXX市卫生健康委员会\n{year}年{month}月{day}日"
    
    elif doc_type == "批复":
        action = params.get('action', '')
        type = params.get('type', '')
        return f"关于同意{action}{type}的批复\n\nXX市卫生健康委员会：\n\n你委《关于申请{action}{type}的请示》（X卫医〔{year}〕{random_number('')}号）收悉。经研究，现批复如下：\n\n一、同意{action}{type}，实施单位为XX单位，负责人为XXX，实施地点位于XX市XX区XX路XX号。\n\n二、项目类别：{random.choice(['基础设施建设', '设备购置', '技术改造', '人员培训'])}；级别：{random.choice(['市级', '省级', '国家级'])}；性质：{random.choice(['非营利性', '营利性'])}；服务对象：社会。\n\n三、项目规模：设置{type}X项，其中：主要设备X项，配套设施X项，其他相关项目X项。\n\n四、项目内容：包括{type}采购、安装、调试、培训等（具体内容以项目实施方案为准）。\n\n五、投资总额：人民币X万元，其中：设备投资X万元，安装调试费用X万元，培训费用X万元，其他费用X万元。\n\n六、请你委按照有关规定，指导实施单位按照国家有关标准进行项目实施，在完成实施并通过验收后，按程序办理相关手续。\n\n七、项目完成后，要严格遵守国家有关法律法规，加强项目管理，提高项目效益，确保项目为人民群众提供优质的服务。\n\nXX省卫生健康委员会\n{year}年{month}月{day}日"
    
    elif doc_type == "决定":
        urgent = params.get('urgent', '')
        year = params.get('year', year)
        award = params.get('award', '')
        return f"关于{urgent}表彰{year}年度{award}集体和个人的决定\n\n各市、县（区）卫生健康委员会，各委属医疗机构：\n\n{year}年，全市卫生健康系统认真贯彻落实党中央、国务院和省、市关于卫生健康工作的决策部署，坚持以人民为中心的发展思想，扎实推进健康XX建设，各项工作取得了显著成效，涌现出一批先进集体和先进个人。\n\n为表彰先进，树立典型，进一步激励全市卫生健康系统广大干部职工的工作积极性和创造性，经研究，决定对XX市第一人民医院等X个先进集体和XXX等X名先进个人予以表彰。\n\n希望受到表彰的先进集体和先进个人珍惜荣誉，再接再厉，继续发挥模范带头作用，在今后的工作中取得更大的成绩。全市卫生健康系统广大干部职工要以先进为榜样，学习他们爱岗敬业、无私奉献的精神，学习他们勇于创新、锐意进取的作风，学习他们全心全意为人民服务的宗旨意识，为推进健康XX建设、保障人民群众身体健康和生命安全做出更大的贡献。\n\n附件：1. {year}年度{award}先进集体名单\n2. {year}年度{award}先进个人名单\n\nXX市卫生健康委员会\n{year}年{month}月{day}日"

# 生成1000条数据
def generate_health_commission_data(count=1000):
    data = []
    generated_instructions = set()  # 用于跟踪已生成的指令，确保唯一性
    
    while len(data) < count:
        # 随机选择公文类型
        doc_type = random.choice(list(document_templates.keys()))
        template = document_templates[doc_type]
        
        # 为每种公文类型生成不同的参数
        params = {}
        
        if doc_type == "通知":
            # 生成通知参数
            params = {
                'urgent': random.choice(urgent_topics),
                'year': random.choice([2023, 2024, 2025]),
                'topic': random.choice(health_topics),
                'institution': random.choice([''] + medical_institutions)
            }
            
        elif doc_type == "通报":
            # 生成通报参数
            params = {
                'urgent': random.choice([''] + urgent_topics),
                'topic': random.choice(health_topics)
            }
            
        elif doc_type == "请示":
            # 生成请示参数
            action_type = random.choice([
                ('购置', device_types),
                ('建设', building_types),
                ('举办', training_types),
                ('开展', new_tech_types),
                ('资金用于', fund_types),
                ('表彰', award_types)
            ])
            action, type_list = action_type
            
            params = {
                'urgent': random.choice([''] + urgent_topics),
                'action': action,
                'type': random.choice(type_list)
            }
            
        elif doc_type == "批复":
            # 生成批复参数
            action_type = random.choice([
                ('设置', ['XX医院', 'XX诊所', 'XX卫生服务中心']),
                ('购置', device_types),
                ('建设', building_types),
                ('举办', training_types),
                ('开展', new_tech_types),
                ('资金用于', fund_types),
                ('表彰', award_types)
            ])
            action, type_list = action_type
            
            params = {
                'action': action,
                'type': random.choice(type_list)
            }
            
        elif doc_type == "决定":
            # 生成决定参数
            params = {
                'urgent': random.choice([''] + urgent_topics),
                'year': random.choice([2023, 2024, 2025]),
                'award': random.choice(award_types)
            }
        
        # 生成指令
        instruction = template["instruction"].format(**params)
        
        # 检查指令是否已存在，避免重复
        if instruction not in generated_instructions:
            # 生成公文内容
            output = generate_document_content(doc_type, params)
            
            # 添加到数据列表
            data.append({
                "instruction": instruction,
                "input": "",
                "output": output
            })
            
            # 将指令添加到已生成集合
            generated_instructions.add(instruction)
    
    return data

# 主函数
if __name__ == "__main__":
    # 生成1000条数据
    data = generate_health_commission_data(1000)
    
    # 保存到文件
    with open("health_commission_data_1000_unique.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Generated 1000 unique health commission documents and saved to health_commission_data_1000_unique.json")
    print(f"Unique instructions: {len(set(doc['instruction'] for doc in data))}")

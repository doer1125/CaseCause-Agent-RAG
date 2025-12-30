import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from conversation_manager import ContextFuser, Message

# 测试对话历史
test_history = [
    Message(role="user", content="你好，我想了解一下行政处罚法的相关内容。"),
    Message(role="assistant", content="您好！行政处罚法是调整行政处罚的设定和实施的法律规范的总称。它规定了行政处罚的种类、设定、实施机关、管辖、适用、决定程序、执行等内容。"),
    Message(role="user", content="行政处罚的种类有哪些？"),
    Message(role="assistant", content="根据行政处罚法，行政处罚的种类包括：（一）警告、通报批评；（二）罚款、没收违法所得、没收非法财物；（三）暂扣许可证件、降低资质等级、吊销许可证件；（四）限制开展生产经营活动、责令停产停业、责令关闭、限制从业；（五）行政拘留；（六）法律、行政法规规定的其他行政处罚。"),
    Message(role="user", content="什么是警告处罚？"),
    Message(role="assistant", content="警告是指行政机关对公民、法人或者其他组织违反行政管理秩序的行为，给予的谴责和告诫，是最轻的一种行政处罚。警告具有教育性和制裁性双重性质，目的是通过对违法行为人的谴责和告诫，使其认识到自己的错误，不再违法。"),
    Message(role="user", content="罚款和没收违法所得有什么区别？"),
    Message(role="assistant", content="罚款是指行政机关强制违法行为人缴纳一定数额的金钱，以剥夺其一定财产权利的行政处罚；没收违法所得是指行政机关依法将违法行为人通过违法途径和方法取得的财产收归国有的行政处罚。两者的主要区别在于：罚款的对象是违法行为人的合法财产，而没收违法所得的对象是违法行为人通过违法手段取得的非法财产。"),
    Message(role="user", content="行政拘留的期限是多久？"),
    Message(role="assistant", content="根据行政处罚法和治安管理处罚法的规定，行政拘留的期限一般为1日以上15日以下。有两种以上违反治安管理行为的，分别决定，合并执行。行政拘留处罚合并执行的，最长不超过20日。"),
    Message(role="user", content="行政处罚的决定程序有哪些？"),
    Message(role="assistant", content="行政处罚的决定程序包括简易程序、一般程序和听证程序。简易程序适用于事实清楚、证据确凿、情节轻微的违法行为；一般程序适用于大多数行政处罚案件；听证程序适用于责令停产停业、吊销许可证件、较大数额罚款等较重的行政处罚案件。"),
    Message(role="user", content="什么是听证程序？"),
    Message(role="assistant", content="听证程序是指行政机关在作出行政处罚决定之前，听取当事人的陈述和申辩，由当事人和调查人员就有关事实和证据进行质证和辩论的程序。听证程序是行政处罚决定程序中的一种特殊程序，旨在保障当事人的合法权益，确保行政处罚决定的合法性和公正性。"),
    Message(role="user", content="行政处罚的执行程序是怎样的？"),
    Message(role="assistant", content="行政处罚的执行程序包括自觉履行和强制执行两种方式。当事人应当在行政处罚决定的期限内，予以履行；当事人逾期不履行行政处罚决定的，作出行政处罚决定的行政机关可以采取下列措施：（一）到期不缴纳罚款的，每日按罚款数额的百分之三加处罚款，加处罚款的数额不得超出罚款的数额；（二）根据法律规定，将查封、扣押的财物拍卖、依法处理或者将冻结的存款、汇款划拨抵缴罚款；（三）根据法律规定，采取其他行政强制执行方式；（四）依照《中华人民共和国行政强制法》的规定申请人民法院强制执行。"),
    Message(role="user", content="什么是一事不再罚原则？"),
    Message(role="assistant", content="一事不再罚原则是指对当事人的同一个违法行为，不得给予两次以上罚款的行政处罚。这一原则的目的是防止行政机关对同一违法行为重复处罚，保护当事人的合法权益。但是，一事不再罚原则并不排除对同一违法行为给予不同种类的行政处罚，例如同时给予罚款和警告处罚。"),
    Message(role="user", content="行政处罚的追诉时效是多久？"),
    Message(role="assistant", content="根据行政处罚法的规定，违法行为在二年内未被发现的，不再给予行政处罚；涉及公民生命健康安全、金融安全且有危害后果的，上述期限延长至五年。法律另有规定的除外。前款规定的期限，从违法行为发生之日起计算；违法行为有连续或者继续状态的，从行为终了之日起计算。"),
    Message(role="user", content="什么是行政处罚的听证主持人？"),
    Message(role="assistant", content="听证主持人是指负责主持行政处罚听证程序的人员。听证主持人由行政机关指定的非本案调查人员担任，与本案无直接利害关系。听证主持人的职责包括：（一）决定举行听证的时间、地点；（二）审查听证参加人的资格；（三）主持听证；（四）决定听证的中止或者终止；（五）制作听证笔录；（六）其他有关职责。"),
    Message(role="user", content="行政处罚法的立法目的是什么？"),
    Message(role="assistant", content="行政处罚法的立法目的是：（一）规范行政处罚的设定和实施；（二）保障和监督行政机关有效实施行政管理；（三）维护公共利益和社会秩序；（四）保护公民、法人或者其他组织的合法权益。"),
    Message(role="user", content="最后一个问题，行政处罚和行政处分有什么区别？"),
    Message(role="assistant", content="行政处罚和行政处分的主要区别在于：（一）适用对象不同：行政处罚适用于公民、法人或者其他组织；行政处分适用于行政机关公务员和法律、法规授权的具有公共事务管理职能的组织中从事公务的人员。（二）适用机关不同：行政处罚由具有行政处罚权的行政机关或者法律、法规授权的组织实施；行政处分由任免机关或者监察机关实施。（三）适用依据不同：行政处罚的依据是法律、行政法规、地方性法规、部门规章和地方政府规章；行政处分的依据是法律、行政法规、部门规章和国务院决定。（四）种类不同：行政处罚的种类包括警告、罚款、没收违法所得、没收非法财物、暂扣许可证件、降低资质等级、吊销许可证件、限制开展生产经营活动、责令停产停业、责令关闭、限制从业、行政拘留等；行政处分的种类包括警告、记过、记大过、降级、撤职、开除。（五）救济途径不同：当事人对行政处罚决定不服的，可以申请行政复议或者提起行政诉讼；当事人对行政处分决定不服的，可以申请复核或者申诉。")
]

# 测试查询
test_query = "行政处罚的听证程序需要哪些条件？"

# 测试不同压缩策略
def test_compression_strategies():
    """测试不同的上下文压缩策略"""
    strategies = ["original", "importance", "summary", "semantic", "hybrid"]
    max_context_length = 1000
    
    print("=== 上下文压缩策略测试 ===")
    print(f"测试查询：{test_query}")
    print(f"最大上下文长度：{max_context_length}")
    print(f"原始对话轮次：{len(test_history) // 2}")
    
    for strategy in strategies:
        print(f"\n--- 策略：{strategy} ---")
        
        # 创建上下文融合器
        context_fuser = ContextFuser(
            max_context_length=max_context_length,
            compression_strategy=strategy
        )
        
        # 融合上下文
        fused_context = context_fuser.fuse_context(test_query, test_history)
        
        # 计算压缩率
        original_length = len(context_fuser._fuse_with_original(test_query, test_history))
        compressed_length = len(fused_context)
        compression_rate = 1 - (compressed_length / original_length)
        
        print(f"原始上下文长度：{original_length}")
        print(f"压缩后长度：{compressed_length}")
        print(f"压缩率：{compression_rate:.2%}")
        print(f"压缩上下文内容：")
        print(fused_context)
        print("-" * 50)

# 测试不同最大上下文长度的影响
def test_max_context_length():
    """测试不同最大上下文长度对压缩效果的影响"""
    max_lengths = [500, 1000, 1500, 2000, 3000]
    strategy = "hybrid"
    
    print("\n\n=== 不同最大上下文长度测试 ===")
    print(f"测试策略：{strategy}")
    print(f"测试查询：{test_query}")
    
    for max_length in max_lengths:
        print(f"\n--- 最大上下文长度：{max_length} ---")
        
        # 创建上下文融合器
        context_fuser = ContextFuser(
            max_context_length=max_length,
            compression_strategy=strategy
        )
        
        # 融合上下文
        fused_context = context_fuser.fuse_context(test_query, test_history)
        
        # 计算压缩率
        original_length = len(context_fuser._fuse_with_original(test_query, test_history))
        compressed_length = len(fused_context)
        compression_rate = 1 - (compressed_length / original_length)
        
        print(f"原始上下文长度：{original_length}")
        print(f"压缩后长度：{compressed_length}")
        print(f"压缩率：{compression_rate:.2%}")
        print("-" * 50)

# 运行测试
if __name__ == "__main__":
    test_compression_strategies()
    test_max_context_length()

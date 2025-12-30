import json

# 读取原始数据
with open('health_commission_data_1000_unique.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 转换为alpaca格式
alpaca_data = []
for item in data:
    alpaca_data.append({
        "instruction": item["instruction"],
        "input": "",
        "output": item["output"]
    })

# 保存转换后的数据
with open('health_commission_data_alpaca.json', 'w', encoding='utf-8') as f:
    json.dump(alpaca_data, f, ensure_ascii=False, indent=2)

# 复制到LLaMA-Factory数据目录
import shutil
shutil.copy('health_commission_data_alpaca.json', '../LLaMA-Factory/data/health_commission_data_alpaca.json')

print("数据转换完成，已保存到health_commission_data_alpaca.json")

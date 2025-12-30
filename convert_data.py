import json

# 读取原始数据
with open('health_commission_data_1000_unique.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 转换为LlamaFactory格式
llama_factory_data = []
for item in data:
    llama_factory_data.append({
        "conversations": [
            {
                "from": "human",
                "value": item["instruction"]
            },
            {
                "from": "assistant",
                "value": item["output"]
            }
        ]
    })

# 保存转换后的数据到LlamaFactory的数据目录
with open('../LLaMA-Factory/data/health_commission_data.json', 'w', encoding='utf-8') as f:
    json.dump(llama_factory_data, f, ensure_ascii=False, indent=2)

print("数据转换完成，已保存到../LLaMA-Factory/data/health_commission_data.json")

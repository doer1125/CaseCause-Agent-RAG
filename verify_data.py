import json

# 验证生成的数据格式
def verify_data_format(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total documents: {len(data)}")
    print(f"Data format: {'Array' if isinstance(data, list) else 'Object'}")
    
    # 验证前5条数据
    for i, doc in enumerate(data[:5]):
        print(f"\nDocument {i+1}:")
        print(f"  Has instruction: {'Yes' if 'instruction' in doc else 'No'}")
        print(f"  Has input: {'Yes' if 'input' in doc else 'No'}")
        print(f"  Has output: {'Yes' if 'output' in doc else 'No'}")
        print(f"  Input length: {len(doc.get('input', ''))}")
        print(f"  Output length: {len(doc.get('output', ''))}")
        print(f"  Instruction: {doc.get('instruction', '')[:50]}...")
    
    # 检查是否所有数据都符合格式
    valid = True
    for i, doc in enumerate(data):
        if not all(key in doc for key in ['instruction', 'input', 'output']):
            print(f"Document {i+1} is invalid: missing required fields")
            valid = False
    
    if valid:
        print("\n✅ All documents are valid!")
    else:
        print("\n❌ Some documents are invalid!")

# 主函数
if __name__ == "__main__":
    verify_data_format("health_commission_data_1000.json")

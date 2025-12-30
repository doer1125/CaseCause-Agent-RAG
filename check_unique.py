import json
import sys

def check_duplicates(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total documents: {len(data)}")
    
    # 使用集合检查重复
    unique_instruction = set()
    duplicate_count = 0
    
    for i, doc in enumerate(data):
        instruction = doc['instruction']
        if instruction in unique_instruction:
            print(f"Duplicate found at index {i}: {instruction}")
            duplicate_count += 1
        else:
            unique_instruction.add(instruction)
    
    print(f"\nUnique instructions: {len(unique_instruction)}")
    print(f"Duplicate count: {duplicate_count}")
    
    if duplicate_count > 0:
        print("❌ Duplicates found!")
    else:
        print("✅ No duplicates found!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "health_commission_data_1000_unique.json"
    
    check_duplicates(file_path)

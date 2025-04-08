import json
import os

def check_json_file(file_path):
    # 读取JSON文件
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 打印基本信息
    print(f"\n文件: {file_path}")
    print(f"样本数量: {len(data)}")
    
    # 检查第一个样本的结构
    if len(data) > 0:
        print("\n第一个样本的结构:")
        first_sample = data[0]
        for key, value in first_sample.items():
            if key == 'graph':
                print(f"{key}: 包含 {len(value)} 条边")
            elif key == 'node_features':
                print(f"{key}: 包含 {len(value)} 个节点")
            else:
                print(f"{key}: {value}")
    
    # 统计标签分布
    labels = [sample['label'] for sample in data]
    print(f"\n标签分布:")
    print(f"漏洞样本 (label=1): {labels.count(1)}")
    print(f"正常样本 (label=0): {labels.count(0)}")

if __name__ == "__main__":
    # 检查训练集和验证集
    base_dir = "./preprocessing/tools/GNNSCVulDetector-master/train_data/reentrancy"
    train_file = os.path.join(base_dir, "train.json")
    valid_file = os.path.join(base_dir, "valid.json")
    
    print("检查训练集:")
    check_json_file(train_file)
    
    print("\n" + "="*50 + "\n")
    
    print("检查验证集:")
    check_json_file(valid_file) 
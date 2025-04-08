import pickle
import json
import os
import random
from collections import defaultdict

def convert_to_gnnsc_format(pickle_file, output_dir):
    # 加载pickle数据
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    # 准备输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 合并所有样本
    all_samples = data['good'] + data['bad']
    
    # 随机打乱数据
    random.shuffle(all_samples)
    
    # 计算训练集和验证集的分割点
    split_point = int(len(all_samples) * 0.8)  # 80% 训练集
    
    # 分割数据
    train_data = all_samples[:split_point]
    valid_data = all_samples[split_point:]
    
    # 转换为GNNSC格式
    def convert_sample(sample):
        # 提取函数名
        func_name = sample['funcname']
        
        # 提取函数体
        func_body = sample['funcbody']
        
        # 构建图结构（这里使用简单的图结构，你可能需要根据实际需求调整）
        graph = []
        node_features = []
        
        # 为每个函数行创建节点
        for i, line in enumerate(func_body):
            # 添加节点特征
            node_features.append([1.0] * 10)  # 示例特征，需要根据实际情况调整
            
            # 添加边（这里使用简单的线性连接）
            if i > 0:
                graph.append([i-1, 0, i])  # [source, edge_type, target]
        
        return {
            "contract_name": sample['filename'].split('/')[-1],
            "func_name": func_name,
            "graph": graph,
            "node_features": node_features,
            "label": 1 if sample in data['bad'] else 0
        }
    
    # 转换训练集和验证集
    train_json = [convert_sample(sample) for sample in train_data]
    valid_json = [convert_sample(sample) for sample in valid_data]
    
    # 保存为JSON文件
    with open(os.path.join(output_dir, 'train.json'), 'w') as f:
        json.dump(train_json, f, indent=2)
    
    with open(os.path.join(output_dir, 'valid.json'), 'w') as f:
        json.dump(valid_json, f, indent=2)
    
    print(f"转换完成！")
    print(f"训练集大小: {len(train_json)}")
    print(f"验证集大小: {len(valid_json)}")
    print(f"文件保存在: {output_dir}")

if __name__ == "__main__":
    # 设置输入输出路径
    # pickle_file = "./preprocessing/train_data/dfcf_data_timestamp.pkl"
    # output_dir = "./preprocessing/tools/GNNSCVulDetector-master/train_data/timestamp"

    pickle_file = "./preprocessing/train_data/dfcf_data_reentrancy.pkl"
    output_dir = "./preprocessing/tools/GNNSCVulDetector-master/train_data/reentrancy"  
    
    # 执行转换
    convert_to_gnnsc_format(pickle_file, output_dir) 
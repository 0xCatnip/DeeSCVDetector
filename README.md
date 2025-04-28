# GNNSCVulDetector ![GitHub stars](https://img.shields.io/github/stars/Messi-Q/GNNSCVulDetector.svg?style=plastic) ![GitHub forks](https://img.shields.io/github/forks/Messi-Q/GNNSCVulDetector.svg?color=blue&style=plastic)

本项目是使用图神经网络（GNN）进行智能合约漏洞检测的 Python 实现。

## 引用

我们参考了[论文](https://www.ijcai.org/Proceedings/2020/0454.pdf)以及部分实现代码：

```
原作者信息：
@inproceedings{zhuang2020smart,
  title={Smart Contract Vulnerability Detection using Graph Neural Network.},
  author={Zhuang, Yuan and Liu, Zhenguang and Qian, Peng and Liu, Qi and Wang, Xiang and He, Qinming},
  booktitle={IJCAI},
  pages={3283--3290},
  year={2020}
}
```

## 环境要求

### 必需的库

*   **Python** 3.6
*   **TensorFlow** 1.14.0 (不支持 tf2.0)
*   **Keras** 2.2.4 (使用 TensorFlow 后端)
*   **scikit-learn** 0.20.2
*   **docopt** (用于命令行界面解析)

运行以下脚本来安装所需的库：

```shell
pip install --upgrade pip
pip install tensorflow==1.14.0
pip install keras==2.2.4
pip install scikit-learn==0.20.2
pip install docopt
```

## 数据集

### 数据集划分与评估

对于每个数据集，我们随机选择 80% 的合约作为训练集，剩余的 20% 作为测试集。
评估指标包括准确率（Accuracy）、召回率（Recall）、精确率（Precision）和 F1 分数（F1 Score）。
我们目前实验复现了重入（Reentrancy）和时间戳依赖（Timestamp Dependence）漏洞检测。

### 数据集获取

我们提供了一个用于从 Etherscan 爬取智能合约源代码的[工具](https://github.com/Messi-Q/Crawler)（开发于 2018 年 8 月）。如果该工具已过时，您可以自行进行相应的改进。

原始数据集请参考数据集[仓库](https://github.com/Messi-Q/Smart-Contract-Dataset)。

### 项目中的数据集结构

智能合约源代码、图数据和训练数据分别存储在以下结构的文件夹中：

```
${GNNSCVulDetector}
├── data
│   ├── timestamp         # 时间戳依赖漏洞数据
│   │   ├── source_code   # 源代码
│   │   └── graph_data    # 图数据 (节点和边)
│   └── reentrancy        # 重入漏洞数据
│       ├── source_code   # 源代码
│       └── graph_data    # 图数据 (节点和边)
├── train_data
│   ├── timestamp         # 时间戳依赖漏洞训练/验证数据
│   │   ├── train.json
│   │   └── valid.json
│   └── reentrancy        # 重入漏洞训练/验证数据
│       ├── train.json
│       └── valid.json
├── features              # (可选) 提取的特征存储位置 (根据README描述推断)
│   └── reentrancy        # 重入漏洞特征
```

*   `data/<vulnerability_type>/source_code`: 存放对应漏洞类型的智能合约源代码。
*   `data/<vulnerability_type>/graph_data`: 存放由 `AutoExtractGraph.py` 提取的智能合约图结构数据（包括节点和边）。
*   `train_data/<vulnerability_type>/train.json`: 对应漏洞类型的训练数据。
*   `train_data/<vulnerability_type>/valid.json`: 对应漏洞类型的验证（测试）数据。
*   `features/reentrancy`: (根据原 README 推断) 存放模型提取的重入漏洞特征。

## 代码文件说明

用于提取图特征（向量）的工具位于 `tools` 目录下：

```
${GNNSCVulDetector}
├── tools
│   ├── remove_comment.py       # (推测) 用于移除代码注释
│   ├── construct_fragment.py   # (推测) 用于构建代码片段
│   ├── reentrancy              # 重入漏洞相关工具
│   │   ├── AutoExtractGraph.py # 提取合约图结构
│   │   └── graph2vec.py        # 将图转换为向量表示
│   └── timestamp               # (推测) 时间戳依赖漏洞相关工具 (结构类似reentrancy)
│       ├── AutoExtractGraph.py
│       └── graph2vec.py
├── BasicModel.py               # 模型基类
├── GNNSCModel.py               # GNN 模型实现与主运行脚本
└── utils.py                    # 辅助函数和类
```

### 主要工具脚本

1.  **`tools/<vulnerability_type>/AutoExtractGraph.py`**
    *   自动分割和存储智能合约代码中的所有函数。
    *   查找函数之间的关系。
    *   将所有智能合约源代码提取为相应的合约图（包含节点和边）。
    *   **运行方式:** (假设针对 reentrancy)
        ```shell
        cd tools/reentrancy
        python AutoExtractGraph.py
        ```
        *注意：原 README 中的 `python3` 可能需要根据你的 Python 环境调整为 `python`。*

2.  **`tools/<vulnerability_type>/graph2vec.py`**
    *   进行特征消融实验。
    *   将合约图转换为向量表示。
    *   **运行方式:** (假设针对 reentrancy)
        ```shell
        cd tools/reentrancy
        python graph2vec.py
        ```
        *注意：原 README 中的 `python3` 可能需要根据你的 Python 环境调整为 `python`。*

## 运行项目

使用以下命令运行主模型训练和评估程序：

```shell
python GNNSCModel.py [options]
```

**可用选项:** (通过运行 `python GNNSCModel.py --help` 查看)

*   `--config-file FILE`: 指定超参数配置文件的路径 (JSON 格式)。
*   `--config CONFIG`: 以 JSON 字符串形式直接提供超参数配置。
*   `--log_dir DIR`: 指定日志文件存储目录。
*   `--data_dir DIR`: 指定数据根目录 (包含 `data` 和 `train_data` 的父目录)。
*   `--random_seed SEED`: 设置随机种子。
*   `--thresholds THRESHOLD`: 设置分类阈值。
*   `--restore FILE`: 从指定的模型文件恢复权重。

**示例:**

```shell
python GNNSCModel.py --random_seed 9930 --thresholds 0.45 --data_dir .
```
*注意：原 README 中的 `python3` 可能需要根据你的 Python 环境调整为 `python`。默认情况下，模型会使用 `train_data/timestamp/` 下的数据，如需更改，请修改 <mcfile name="BasicModel.py" path="e:\Alex\GNNSCVHunter\GNNSCVulDetector-master\BasicModel.py"></mcfile> 中的 `train_file` 和 `valid_file` 参数或使用配置文件指定。*

## 注意事项

*   数据处理相关的代码已包含在项目中。

## 相关文献参考

1.  Li Y, Tarlow D, Brockschmidt M, et al. Gated graph sequence neural networks. ICLR, 2016. ([GGNN 论文](https://arxiv.org/abs/1511.05493))
2.  Qian P, Liu Z, He Q, et al. Towards automated reentrancy detection for smart contracts based on sequential models. 2020. ([ReChecker 项目](https://github.com/Messi-Q/ReChecker))
```

        

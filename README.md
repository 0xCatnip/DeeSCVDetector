# DeeSCVDetector

A smart contract vulnerability detection system based on deep learning techniques.

## Project Overview

DeeSCVDetector is a comprehensive tool for detecting vulnerabilities in Solidity smart contracts using deep learning approaches. The project combines static analysis with machine learning to identify potential security issues in smart contract code.

## Project Structure

```
DeeSCVDetector/
├── preprocessing/           # Data preprocessing and feature extraction
│   ├── data/               # Raw contract data
│   ├── tools/              # Preprocessing utilities
│   ├── train_data/         # Processed training data
│   └── label/              # Vulnerability labels
├── detection/              # Vulnerability detection models
├── python-solidity-parser/ # Solidity parser for code analysis
└── anaconda_projects/      # Development and testing notebooks
```

## Features

- Solidity code parsing and analysis
- Deep learning-based vulnerability detection
- Support for multiple vulnerability types
- Comprehensive data preprocessing pipeline
- Model training and evaluation tools

## Prerequisites

- Python 3.7+
- ANTLR4
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DeeSCVDetector.git
cd DeeSCVDetector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up the Solidity parser:
```bash
cd python-solidity-parser
python setup.py install
```

## Usage

1. Data Preprocessing:
```bash
cd preprocessing
# Run preprocessing scripts
```

2. Model Training:
```bash
cd detection
# Train the detection model
```

3. Vulnerability Detection:
```bash
# Run detection on new contracts
```

## Development

- The project uses Jupyter notebooks for development and testing
- Main development environment is in `anaconda_projects/`
- Preprocessing scripts are in `preprocessing/tools/`

## License

[Add your license information here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

[Add your contact information here]

# DeeSCVDetector - 智能合约漏洞检测系统

[![Solidity](https://img.shields.io/badge/Solidity-%5E0.8.0-blue)](https://soliditylang.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

基于深度学习的智能合约安全漏洞检测工具，支持以太坊虚拟机（EVM）兼容合约的自动化安全审计。

## ✨ 主要功能
- 重入攻击检测（Reentrancy）
- 整数溢出/下溢检测（Integer Overflow/Underflow）
- 访问控制漏洞检测（Access Control）
- 未检查返回值检测（Unchecked Return Values）
- 时间戳依赖检测（Timestamp Dependency）
- 闪电贷攻击模式识别（Flash Loan Attack Patterns）

## 🛠️ 技术栈
- **静态分析引擎**：Slither + Mythril
- **深度学习框架**：PyTorch 2.0
- **合约编译环境**：solc 0.8.x
- **依赖管理**：Poetry
- **可视化分析**：Matplotlib + Plotly

## 📦 安装指南

### 前置要求
- Python 3.10+
- Node.js 16.x (用于Hardhat测试环境)
- solc 0.8.x

```bash
# 克隆仓库
git clone https://github.com/0xCatnip/DeeSCVDetector.git
cd DeeSCVDetector
```

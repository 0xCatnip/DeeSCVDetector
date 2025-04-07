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
git clone https://github.com/yourusername/DeeSCVDetector.git
cd DeeSCVDetector

# 安装Python依赖
pip3 install -r requirements.txt

# 安装智能合约开发环境（可选）
npm install --save-dev hardhat @nomicfoundation/hardhat-toolbox
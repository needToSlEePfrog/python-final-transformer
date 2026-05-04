[README (1).md](https://github.com/user-attachments/files/27350356/README.1.md)
# Transformer 论文复现：《Attention Is All You Need》

安徽大学 · Python 工程应用课程期末大作业

## 项目简介

本项目从零复现 Transformer 架构，分两个阶段：

- **阶段一**：使用纯 NumPy 手动实现 Transformer 各核心模块，逐步验证数学原理
- **阶段二**：迁移至 PyTorch，在 Anki 英中句对数据集上完成端到端翻译训练

## 文件说明

| 文件 | 说明 |
|------|------|
| `transformer.ipynb` | 阶段一：NumPy 手动实现（本地 VS Code 运行） |
| `googlecolab.ipynb` | 阶段二：PyTorch 英中翻译训练（Google Colab GPU 运行） |
| `report.docx` | 课程报告 |

## 复现内容

**阶段一（NumPy）**
- 缩放点积注意力（Scaled Dot-Product Attention）
- 位置编码（Positional Encoding）及可视化
- 多头注意力（Multi-Head Attention）
- 前馈网络（FFN）与层归一化（Layer Norm）
- 完整 Encoder（6 层堆叠）

**阶段二（PyTorch）**
- 完整 Transformer（Encoder + Decoder）
- 数据预处理：jieba 分词、opencc 繁简转换、低频词过滤
- 五轮迭代训练，Loss 从 5.06 降至 0.89
- 推理优化：EOS 截断、temperature 解码

## 训练结果摘要

数据集：Anki 英中句对，约 30,189 条有效句对

| 输入 | 输出 |
|------|------|
| I love you. | 我爱你。 |
| She is a student. | 她是个学生。 |
| I am hungry. | 我饿了。 |
| Thank you. | 感谢你。 |
| Where are you going? | 你要去哪里？ |
| I don't know. | 我不知道我不知道。 |

## 环境依赖

**阶段一（本地）**
```
numpy
matplotlib
```

**阶段二（Google Colab）**
```
torch
jieba
opencc-python-reimplemented
nltk
```

## 主要发现

- 数据简繁混用会显著干扰训练，繁简转换是必要预处理步骤
- 低频词翻译质量天然受限（`Hello` 仅 8 条训练样本，始终无法正确翻译）
- 推理阶段的 EOS 截断和 temperature 参数对输出质量影响显著

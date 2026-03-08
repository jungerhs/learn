# LLM & VLM & Agent 面试笔记

欢迎来到我的个人学习笔记！这里整理了大模型相关的面试常见问题和参考答案。

## 📚 内容目录

- [LLM 八股](/chapters/1-LLM八股.md) - 大语言模型相关面试问题
- [VLM 八股](/chapters/2-VLM八股.md) - 视觉语言模型相关面试问题
- [RLHF 八股](/chapters/3.RLHF八股.md) - RLHF 相关面试问题
- [Agent](/chapters/4-Agent.md) - Agent 相关面试问题
- [RAG](/chapters/5-RAG.md) - RAG 相关面试问题
- [模型评估与 Agent 评估](/chapters/6-模型与Agent评估.md) - 模型与 Agent 评估相关面试问题

## 🎯 主要内容

### LLM 八股
- Transformer 模型原理
- 自注意力机制
- 位置编码（RoPE）
- MHA/MQA/GQA 注意力变体
- LLM 架构对比（Encoder-Only, Decoder-Only, Encoder-Decoder）
- Scaling Laws

### VLM 八股
- 多模态模型核心挑战
- CLIP 原理
- LLaVA/MiniGPT-4 架构
- 视觉指令微调
- 视频多模态处理
- Grounding 定位能力

### RLHF 八股
- RLHF 原理与流程
- PPO 算法
- DPO 方法

### Agent
- Agent 定义与核心组件
- ReAct 框架
- Agent 循环与规划
- 记忆模块
- 工具使用

### RAG
- RAG 工作原理
- RAG 流水线详解
- 检索与重排序
- RAG 与微调对比

### 模型评估与 Agent 评估
- 传统评估指标局限性
- MMLU/Big-Bench/HumanEval 基准测试
- 事实性评估
- 推理能力评估

## 💡 使用说明

这是一个基于 [Docsify](https://docsify.js.org/) 构建的在线文档站点。

### 本地运行

```bash
npm install -g docsify-cli
cd 项目目录
docsify serve .
```
### 访问 GitHub Pages 查看笔记：https://jungerhs.github.io/learn/
然后在浏览器中打开 `http://localhost:3000`

## 📝 说明

所有内容均为个人学习整理，欢迎指正交流！
内容来自于：[datawhale](https://github.com/datawhalechina/hello-agents/tree/main/Extra-Chapter)

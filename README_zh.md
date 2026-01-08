# CoRAG: Chain-of-Retrieval Augmented Generation

[English Version](README.md) | [中文版](README_zh.md)

本仓库提供了 [Chain-of-Retrieval Augmented Generation](https://arxiv.org/abs/2501.14342) (CoRAG) 的实现代码。本项目支持推理 (Agentic RAG)、训练以及数据集构建。

## 项目结构 (Project Structure)

```text
corag/
├── src/
│   ├── agent/           # CoRAG 智能体逻辑 (路径生成, 树搜索等)
│   ├── inference/       # 推理工具及指标计算
│   ├── search/          # 检索客户端 (Graph API, HTTP)
│   ├── train/           # 训练脚本 (数据准备, SFT 循环)
│   ├── config.py        # 全局配置参数
│   └── vllm_client.py   # 与 VLLM 服务器通信的客户端
├── scripts/
│   ├── custom_batch_eval.py    # 批量评测主脚本
│   ├── rejection_sampling_gen.py # 推理轨迹生成脚本
│   └── start_vllm_server.sh    # 启动 VLLM 的辅助脚本
├── interactive_demo.py  # 用于测试单条查询的交互式 CLI
└── requirements.txt     # Python 依赖
```

# 第一部分：推理 (Agentic RAG)

本章节详细介绍如何运行 CoRAG 智能体进行推理，包括环境配置、参数说明、运行交互式 Demo 以及执行批量评测。

## 1. 环境配置 (Environment Setup)

### 软件依赖
请使用提供的 `requirements.txt` 安装必要的 Python 依赖包：

```bash
pip install -r requirements.txt
```

核心依赖包括：
- `vllm`: 用于高效的大模型服务部署。
- `transformers`, `torch`: 核心深度学习库。
- `flash-attn`: 用于加速 Attention 计算。

### 服务端要求
CoRAG 依赖两个外部服务：LLM 服务器和检索 (Retrieval) 服务器（Graph API 或自定义服务）。

#### A. LLM 服务器 (VLLM)
你需要运行一个 VLLM 服务器来处理生成请求。
可以使用提供的脚本启动 VLLM 服务器：
```bash
# 在 8000 端口启动指定模型的 VLLM 服务
bash scripts/start_vllm_server.sh <model_name_or_path>
```
默认 URL: `http://localhost:8000/v1`

#### B. 检索服务器 (Retrieval Server)
你需要一个服务来处理文档检索。这应该是自定义的 Graph API。
默认 URL: `http://localhost:8023/retrieve`

---

## 2. 配置参数 (Configuration Parameters)

系统可以通过命令行参数进行高度配置。`src/config.py` 中的关键参数包括：

### 模型与 API 配置
- `--vllm_api_base`: VLLM API 的 URL (默认: `http://localhost:8000/v1`)。
- `--vllm_api_key`: VLLM 的 API Key (默认: `token-123`)。
- `--vllm_model`: 指定使用的模型名称。
- `--graph_api_url`: 检索服务的 URL (例如: `http://localhost:8023/retrieve`)。
- `--corpus_file`:用于将检索到的 ID 映射到文本的语料库 JSON 文件路径。

### 推理策略
- `--task_desc`: 智能体的任务描述 (默认: "answer multi-hop questions")。
- `--max_path_length`: 允许的最大推理步数 (深度)。
- `--decode_strategy`: 路径生成策略。选项包括：
  - `greedy`: 贪婪策略，总是选择可能性最大的下一步。
  - `tree_search`: 树搜索，探索多个分支。
  - `best_of_n`: 采样 `n` 条路径并选择最好的一条。
- `--sample_temperature`: 树搜索或 best-of-n 策略中的采样温度。
- `--best_n`: `best_of_n` 策略中采样的路径数量。

### 检索设置
- `--num_contexts`: 每步检索的文档数量 (默认: 20)。
- `--num_threads`: 批量评测时的并行检索线程数 (默认: 32)。

---

## 3. 交互式 Demo 脚本 (Interactive Demo Script)

`interactive_demo.py` 脚本允许你与 CoRAG 智能体进行实时对话。它会展示智能体的“思考”过程，包括中间查询、检索到的文档以及思考内容。

### 使用方法
```bash
python interactive_demo.py \
    --vllm_api_base http://localhost:8000/v1 \
    --graph_api_url http://localhost:8023/retrieve \
    --corpus_file data/corpus.json \
    --max_path_length 3 \
    --decode_strategy greedy
```

### 功能特性
- **实时推理**: 展示子问题 (SubFor) 和子答案 (SubAnswer) 生成的每一步。
- **可追踪性**: 显示每一步主要检索到了哪些文档。
- **灵活测试**: 允许快速测试不同的问题，无需运行完整的 benchmark。

---

## 4. 批量评测脚本 (Batch Evaluation Script)

对于大规模评测，请使用 `scripts/custom_batch_eval.py`。该脚本处理问题数据集，计算指标（如 Recall），并保存结果。

### 使用方法
```bash
python scripts/custom_batch_eval.py \
    --eval_file data/my_questions.json \
    --save_file results/my_results.json \
    --corpus_file data/corpus.json \
    --calc_recall \
    --num_threads 16
```

### 输入格式 (`--eval_file`)
输入应为包含字典的 JSON 列表。每个字典必须至少包含 `question` 字段。
```json
[
    {
        "question": "Who is the director of the movie related to...",
        "answer": "Target Answer"  // 可选，用于对比参考
    }
]
```

### 关键参数
- `--eval_file`: 输入 JSON 数据集的路径。
- `--save_file`: 输出 JSON 结果的保存路径。
- `--calc_recall`: 如果设置，将计算检索召回率指标 (输入中需要包含 `answer` 或 golden facts)。
- `--enable_naive_retrieval`: 如果设置，还将运行基线“朴素”检索 (单步) 进行比较。
- `--num_threads`: 用于加速处理的并发线程数。

### 输出结果
输出文件包含：
1. **汇总对象 (Summary Object)**: 位于顶部的聚合统计信息 (平均时间, Micro Recall 等)。
2. **单项结果 (Per-Item Results)**: 每个问题的详细追踪信息，包括：
   - `reasoning_steps`: (子问题, 子答案) 列表。
   - `corag_recall`: CoRAG 的 Hits 和 Total recall。
   - `naive_recall`: Naive baseline 的 Hits 和 Total recall (如果启用)。

# 第二部分：训练 (Training)

本章节介绍如何使用 CoRAG 方法对模型进行全量微调 (SFT)。

## 训练流程 (Training Process)

核心训练逻辑实现于 `src/train/train.py`。我们提供了一个 Shell 脚本 `src/train/run_training.sh` 来简化流程。

### 功能特性
*   **CoRAG Masking 策略**: 模型被训练用于生成 *推理路径* (子问题, 子答案, 最终答案)。我们仅计算这些生成部分的 Loss。*用户查询* 和 *检索到的文档* (外部知识) 会被 Mask 掉 (loss=0)。
*   **DeepSpeed 支持**: 默认配置使用 DeepSpeed ZeRO-3，以便在多 GPU 环境下高效利用显存。

### 运行训练
```bash
bash src/train/run_training.sh
```

### 关键训练参数
脚本使用标准的 Hugging Face `TrainingArguments`。`src/train/run_training.sh` 中的重要参数包括：

*   `--train_file`: 准备好的 JSONL 训练数据路径。
*   `--model_name_or_path`: 用于微调的基座模型 (例如 `Qwen/Qwen2.5-7B-Instruct`)。
*   `--output_dir`: 保存 Checkpoints 和 Logs 的目录。
*   `--learning_rate`: 默认值为 `2e-5`。
*   `--num_train_epochs`: 默认值为 `3`。
*   `--gradient_accumulation_steps`: 调整此参数以达到预期的有效 Batch Size。
*   `--bf16`: 强烈建议在较新的 GPU (A100/H100) 上开启，以节省显存并提高稳定性。

# 第三部分：数据集构造 (Dataset Construction)

本章节包含用于构建训练数据集的工具，可以通过格式化现有 Benchmark 数据或通过拒绝采样生成新的推理路径。

## 拒绝采样策略 (Rejection Sampling Strategy)

为了训练 CoRAG 模型，我们需要包含 (问题, 推理路径, 答案) 的数据。由于标准数据集仅提供 (问题, 答案)，我们使用 CoRAG 智能体探索多条推理路径，并仅保留包含正确答案的路径。

### 脚本: `scripts/rejection_sampling_gen.py`

此脚本在数据集上运行 CoRAG 智能体，对每个问题进行多次路径采样，并将最终答案与标准答案 (Ground Truth) 进行验证。

#### 使用方法

```bash
python scripts/rejection_sampling_gen.py \
    --dataset corag/multihopqa \
    --split train \
    --output_file data/rejection_sampled_data.jsonl \
    --n_samples 5 \
    --temperature 0.7 \
    --max_path_length 3 \
    --vllm_url http://localhost:8000 \
    --graph_api_url http://localhost:8023/retrieve
```

#### 关键参数
- `--n_samples`: 每个问题采样的推理路径数量。
- `--temperature`: 采样温度，用于增加多样性。
- `--output_file`: 有效路径的保存位置。
- `--max_path_length`: 最大推理深度。

#### 输出格式
输出为一个 JSONL 文件，每一行是一个包含有效路径的样本，包含：
- `query`: 原始问题。
- `generated_final_answer`: 智能体生成的正确答案。
- `steps`: 推理步骤列表，每步包含 `subquery` (子问题), `subanswer` (子答案) 和 `thought` (思考)。

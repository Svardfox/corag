## CoRAG：Chain-of-Retrieval Augmented Generation

[English version](README.md) | 中文说明

本仓库实现了论文 *“Chain-of-Retrieval Augmented Generation”*（CoRAG，链式检索增强生成，`https://arxiv.org/abs/2501.14342`）中的方法。  
CoRAG 主要支持三条完整工作流：

- **数据集构造**：生成多步推理轨迹，并为每一步补充检索到的上下文。
- **模型训练**：基于 CoRAG 风格的监督信号微调大模型。
- **QA 评测**：使用 CoRAG agent 进行批量和交互式问答评测。

整个代码假定外部已部署好：

- 一个提供 **OpenAI 兼容 API** 的 **vLLM 服务**。
- 一个可提供文本检索结果的 **检索服务**（Graph / HTTP 检索 API）。

---

## 项目结构

```text
corag/
├── src/
│   ├── agent/                # CoRAG 智能体实现及工具
│   │   ├── corag_agent.py
│   │   └── agent_utils.py
│   ├── train/                # 训练与数据准备脚本
│   │   ├── train.py
│   │   ├── run_training.sh
│   │   ├── prepare_training_data.py
│   │   └── prepare_aligned_data.py
│   ├── config.py             # 全局 CLI 配置（Arguments dataclass）
│   ├── vllm_client.py        # OpenAI 兼容的 vLLM 客户端
│   ├── data_utils.py         # 语料加载与 context 格式化
│   ├── prompts.py            # CoRAG 用到的 Prompt 模板
│   ├── utils.py              # 工具函数（如 AtomicCounter）
│   └── logger_config.py      # 日志配置
├── scripts/
│   ├── rejection_sampling_gen.py   # 使用拒绝采样生成 CoRAG 推理路径
│   └── custom_batch_eval.py        # 使用 CoRAG 的批量 QA 评测脚本
├── interactive_demo.py       # 单样本交互式 QA Demo（命令行）
├── requirements.txt          # Python 依赖
└── images/
    └── corag_framework.png   # CoRAG 框架示意图
```

---

## 环境依赖与安装

- **Python**：建议 3.9+
- **GPU**：需要支持 CUDA 的 GPU 以运行 vLLM 和训练

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

主要依赖：

- **vllm**：用于提供 OpenAI 风格的高性能推理服务。
- **transformers**、**torch**：深度学习基础库。
- **datasets**：数据集加载与处理。
- **deepspeed**：大模型分布式训练（ZeRO-3）。

---

## 外部服务

CoRAG 依赖两个外部服务：

- **vLLM 服务**：负责所有生成调用（OpenAI 风格接口）。
- **检索服务**：根据子问题返回候选文档。

二者都通过 `src/config.py` 中的 `Arguments` dataclass 所定义的命令行参数进行配置。

### vLLM 服务 API

vLLM 服务需要实现 **OpenAI 兼容** 的 REST API：

- 基础地址（示例）：`http://localhost:8000/v1`
- 认证头：`Authorization: Bearer <API_KEY>`

#### Chat Completions

- **Endpoint**：`POST /v1/chat/completions`
- **请求示例**：

```json
{
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is CoRAG?"}
  ],
  "temperature": 0.7,
  "max_tokens": 256
}
```

- **响应结构（简化）**：

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "CoRAG (Chain-of-Retrieval Augmented Generation) is ..."
      }
    }
  ],
  "usage": {
    "prompt_tokens": 50,
    "completion_tokens": 120,
    "total_tokens": 170
  }
}
```

CoRAG 通过 `src/vllm_client.py` 中的 `VllmClient` 使用该 API，主要出现在：

- `scripts/rejection_sampling_gen.py`
- `scripts/custom_batch_eval.py`
- `interactive_demo.py`
- `src/agent/corag_agent.py`

### 检索 API 规范

检索服务用于根据子问题返回支撑文档，通过 `graph_api_url` 访问（例如 `http://localhost:8023/retrieve`），在以下脚本中被调用：

- `scripts/rejection_sampling_gen.py`（由 `CoRagAgent` 间接调用）
- `src/train/prepare_training_data.py`
- `scripts/custom_batch_eval.py`
- `interactive_demo.py`

为保证所有组件都能正常工作，建议检索 API 遵循以下约定。

#### 请求格式

- **方法**：`POST`
- **路径**：`/retrieve`（完整 URL 由 `graph_api_url` 配置）
- **推荐请求体**：

```json
{
  "query": "What is the capital of France?",
  "top_k": 20
}
```

- **字段说明**：
  - **`query`**（string，必选）：自然语言查询或子问题。
  - **`top_k`**（int，可选）：希望返回的文档数量上限。  
    例如 `prepare_training_data.py` 本身也会通过 `--top_k` 控制截断，服务端可以选择参考或忽略该字段。

#### 推荐响应格式

推荐返回以下任意一种响应形态，**检索客户端会统一解析**（训练与 CoRAG agent 都适用）：

```json
{
  "chunks": [
    {
      "id": "wiki_123",
      "contents": "Paris is the capital and most populous city of France."
    },
    {
      "id": "wiki_456",
      "contents": "France is a country located in Western Europe ..."
    }
  ]
}
```

其中 `chunks` 中每个元素：

- **`id`**（string，可选但推荐）：文档唯一标识。
- **`contents`**（string，推荐），或 **`content`** / **`text`**：具体文档内容。

#### 其他兼容响应形态

检索客户端对返回结构有较强的容错能力，同时也接受：

- **字符串列表**：

```json
[
  "Paris is the capital and most populous city of France.",
  "France is a country located in Western Europe ..."
]
```

- **对象列表**（带 `contents` / `content` / `text` 字段）：

```json
[
  {"text": "Paris is the capital and most populous city of France."},
  {"text": "France is a country located in Western Europe ..."}
]
```

- **包含列表字段的对象**：

```json
{
  "results": [
    {"contents": "..."},
    {"contents": "..."}
  ]
}
```

检索客户端会按以下逻辑统一解析：

- 依次检查 `["chunks", "data", "results", "docs", "passages"]` 这些 key 是否存在且对应为列表。
- 若未找到列表，则在必要时将单个对象封装为列表。
- 对每个元素依次从 `contents` → `content` → `text` → `str(doc)` 中提取文本。

只要你的检索服务返回满足上述任意一种形态，CoRAG（训练与推理）都能正常消费结果。

---

## 通过 `src/config.py` 进行统一配置

全局运行配置由 `src/config.py` 中的 `Arguments` dataclass 定义，并继承自 HuggingFace 的 `TrainingArguments`。  
常用且同时被训练与评测脚本使用的重要字段包括：

- **模型与 vLLM**
  - `--vllm_api_base`：vLLM API 基地址（默认 `http://localhost:8000/v1`）。
  - `--vllm_api_key`：vLLM API Key（默认 `token-123`）。
  - `--vllm_model`：vLLM 模型名称 / ID（可选；若为空则自动从 `/models` 中探测）。
  - `--tokenizer_name`：Tokenizer 对应的 Repo 名或路径（默认与 vLLM 模型 ID 一致）。
- **检索相关**
  - `--graph_api_url`：检索服务地址（如 `http://localhost:8023/retrieve`）。
  - `--corpus_file`：用于将文档 ID 映射为文本的语料库 JSON 文件路径（部分模式下使用）。
  - `--num_contexts`：每个样本使用的最大上下文数（默认 20）。
- **Agent 行为**
  - `--max_path_length`：最大推理步数（路径长度）。
  - `--decode_strategy`：`greedy` | `tree_search` | `best_of_n`。
  - `--sample_temperature`：非贪心策略下的采样温度。
  - `--best_n`：`best_of_n` 策略下采样的路径数量。
- **执行相关**
  - `--num_threads`：并行线程数（用于评测与检索）。
  - `--max_len`：最终回答阶段的最大输入长度。

你可以在以下脚本中直接使用这些参数：

- `scripts/custom_batch_eval.py`
- `interactive_demo.py`
- `src/train/train.py`（通常通过 `run_training.sh` 间接传入）

---

## 工作流一：数据集构造

数据集构造流水线分为三个阶段：

1. **准备原始 QA 数据集**：至少包含 `query` 和 `answers` 字段。
2. **运行拒绝采样**：使用 CoRAG agent 生成多条推理路径，只保留答案正确的路径。
3. **补充检索上下文并对监督信号进行拆分对齐**：得到可直接训练的监督样本。

### 1.1 准备原始数据集

要求每个样本至少包含：

- **`query`**：用户问题。
- **`answers`**：可接受答案的字符串列表。

常见做法：

- 直接使用现成的 HF 数据集（如 `corag/multihopqa`，`subset=2wikimultihopqa`）。
- 将自定义数据集转为 JSONL，每行一个 JSON 对象：

```json
{"query": "Who is the CEO of OpenAI?", "answers": ["Sam Altman"]}
```

### 1.2 使用拒绝采样生成推理路径

通过 `scripts/rejection_sampling_gen.py`，使用 CoRAG agent 在数据集上采样多条推理路径，并只保留最终答案正确的路径。

示例（使用 HF 数据集）：

```bash
python scripts/rejection_sampling_gen.py \
  --dataset corag/multihopqa \
  --subset 2wikimultihopqa \
  --split train \
  --output_file data/rejection_sampled_data.jsonl \
  --vllm_api_base http://localhost:8000/v1 \
  --vllm_api_key token-123 \
  --graph_api_url http://localhost:8023/retrieve \
  --n_samples 5 \
  --temperature 0.7 \
  --max_path_length 3
```

输出文件 `data/rejection_sampled_data.jsonl` 是一个 JSONL，每行一个 CoRAG 有效样本，包含（简化）：

- `id`：样本 ID。
- `query`：原始问题。
- `answers`：标准答案列表。
- `generated_final_answer`：被判定为正确的模型最终回答。
- `steps`：推理步骤列表，每个步骤包含：
  - `subquery`
  - `subanswer`
  - 可选 `thought`。

此阶段 **未必包含完整检索上下文文本**，下一步会补齐。

### 1.3 补充检索上下文

`src/train/prepare_training_data.py` 会将拒绝采样得到的样本转为 ChatML 风格对话，并**实际调用检索 API**，为每个 `subquery` 填入对应的检索上下文。

示例：

```bash
python src/train/prepare_training_data.py \
  --dataset data/rejection_sampled_data.jsonl \
  --output_file data/train_with_context.jsonl \
  --graph_api_url http://localhost:8023/retrieve \
  --top_k 3 \
  --num_workers 10 \
  --retrieve
```

输入要求：

- 来自 `rejection_sampling_gen.py` 的 JSONL，包含 `query`、`steps` 和 `generated_final_answer` 等字段。

输出 `data/train_with_context.jsonl`：

- 每行一个对象，主要字段为 `messages`（ChatML 格式），大致形如：
  - `{"role": "user", "content": "<original query>"}`  
  - 对于每个推理步骤：
    - `{"role": "assistant", "content": "SubQuery: <subquery>"}`  
    - `{"role": "observation", "content": "Retrieved Context:\nDoc 1: ...\nDoc 2: ..."}`  
    - `{"role": "assistant", "content": "SubAnswer: <subanswer>"}`  
  - 最终答案：
    - `{"role": "assistant", "content": "Final Answer: <answer>"}`。

该文件即为**已补充检索上下文的对话式数据集**。

### 1.4 构建对齐监督样本

`src/train/prepare_aligned_data.py` 会把每条对话拆分为多个**类型明确的监督样本**：

- `subquery_generation`
- `subanswer_generation`
- `final_answer_generation`

每个样本都是一段 ChatML 对话：

- **输入** 由前面的若干 `system` / `user` / `assistant` / `observation` 消息编码。
- **监督目标** 是单条 `assistant` 消息（如 `SubQuery: ...`）。

示例：

```bash
python src/train/prepare_aligned_data.py \
  --input_file data/train_with_context.jsonl \
  --output_file data/aligned_train.jsonl
```

输出 `data/aligned_train.jsonl`：

- 每行一个 JSON，包含：
  - `type`：`subquery_generation` / `subanswer_generation` / `final_answer_generation`。
  - `messages`：完整的 ChatML 消息列表（包含 Prompt 与金标回答）。

这是后续监督微调使用的**最终训练数据集**。

---

## 工作流二：模型训练

训练逻辑位于 `src/train/train.py`，直接消费上述对齐后的训练数据。  
我们提供了一个默认的启动脚本 `src/train/run_training.sh` 以简化使用。

### 2.1 训练命令示例

默认的 `run_training.sh` 使用 DeepSpeed ZeRO-3 在多卡上训练：

```bash
bash src/train/run_training.sh
```

脚本内部等价于执行：

```bash
torchrun --nproc_per_node 8 src/train/train.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --train_file data/train_with_graph_retrieval.jsonl \
  --output_dir output/corag_qwen_finetuned \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing True \
  --deepspeed src/train/ds_config_zero3.json \
  --learning_rate 2e-5 \
  --max_len 4096 \
  --do_train \
  --save_steps 500 \
  --logging_steps 10 \
  --report_to wandb \
  --bf16 True \
  --overwrite_output_dir
```

你通常需要：

- 将 `--train_file` 指向自己准备好的训练数据（例如 `data/aligned_train.jsonl` 或 `data/train_with_graph_retrieval.jsonl`，取决于你的设置）。
- 根据设备情况调整 `--model_name_or_path`、`--output_dir`、DeepSpeed 配置以及各类训练超参。

### 2.2 训练目标（高层描述）

CoRAG 的训练目标强调**推理路径的生成**而非直接记忆答案：

- 模型需学会生成：
  - `SubQuery: ...`
  - `SubAnswer: ...`
  - `Final Answer: ...`
- 用户问题与检索到的原始文档仅作为**输入上下文**，在 loss 计算中被 mask 掉（loss=0）。

这样可以促使模型学习 **基于检索证据的多步推理能力**，而不是简单的端到端记忆。

---

## 工作流三：QA 评测与交互 Demo

CoRAG 提供两种主要评测与调试方式：

- 使用 `custom_batch_eval.py` 进行**批量评测**。
- 使用 `interactive_demo.py` 进行**交互式调试与展示**。

### 3.1 批量评测：`scripts/custom_batch_eval.py`

`custom_batch_eval.py` 会在自定义评测集上运行 CoRAG agent，并可选地计算**检索召回率**，以及与朴素检索基线进行对比。

示例：

```bash
python scripts/custom_batch_eval.py \
  --eval_file data/my_eval.json \
  --save_file results/my_eval_results.json \
  --vllm_api_base http://localhost:8000/v1 \
  --vllm_api_key token-123 \
  --graph_api_url http://localhost:8023/retrieve \
  --corpus_file data/corpus.json \
  --max_path_length 3 \
  --decode_strategy greedy \
  --num_threads 16 \
  --calc_recall \
  --enable_naive_retrieval
```

#### 输入格式（`--eval_file`）

`eval_file` 应为一个 JSON 文件，内容是**对象列表**。至少需要：

- `question`：输入问题。
- `answer`：参考答案（可选但推荐，用于日志记录）。

若使用 HotpotQA / MuSiQue 等多跳数据集，还可以包含：

- `paragraphs`、`context`、`supporting_facts` 等字段，脚本会据此构建 **golden facts**，用于检索召回率计算。  
当开启 `--calc_recall` 后，脚本将：

- 计算 CoRAG 的检索召回率（对所有 golden chunk 做 **Micro Recall**）。
- 若再设置 `--enable_naive_retrieval`，则额外计算朴素单步检索的召回率。

#### 输出格式（`--save_file`）

`save_file` 是一个 JSON 文件，内部结构为：

- 索引 0 位置是一条 **汇总对象**，包含：
  - 平均时间开销、
  - Micro Recall 等整体统计指标。
- 其余位置为逐样本结果，每项包含：
  - `question`、`answer`、`ground_truth`。
  - `reasoning_steps`（CoRAG 的子问题与子答案列表）。
  - `time`（时间拆分）。
  - `corag_recall` 与 `naive_recall`（若启用相应选项）。

### 3.2 交互式 Demo：`interactive_demo.py`

`interactive_demo.py` 提供了一个命令行交互界面，可逐条输入问题并观察 CoRAG 的内部推理过程。

示例：

```bash
python interactive_demo.py \
  --vllm_api_base http://localhost:8000/v1 \
  --vllm_api_key token-123 \
  --graph_api_url http://localhost:8023/retrieve \
  --corpus_file data/corpus.json \
  --max_path_length 3 \
  --decode_strategy greedy
```

运行后你会看到：

- 提示：`Enter your query:`，可直接输入问题。
- 控制台输出的**中间推理步骤**：
  - 子问题（`SubQuery`）
  - 子答案（`SubAnswer`）
  - 可选的 `Thought` 以及检索到的文档 ID。
- 最后的 CoRAG 回答显示在 `--- Final Answer ---` 部分。

推荐使用该脚本来：

- 验证检索 API 是否返回了合理上下文。
- 观察 CoRAG 拆分路径（子问题序列）是否符合预期。
- 在大规模评测前，对模型行为进行快速调试。

---

## 小贴士与常见问题排查

- **检查 vLLM 服务是否正常**：
  - 通过 `GET <vllm_api_base>/models` 确认至少返回一个可用模型 ID。
  - 若自动探测失败，可显式指定 `--vllm_model`。
- **检查检索 API 是否符合约定**：
  - 手动 `curl` 你的 `/retrieve` 接口，确认返回结构满足上文中列出的几种格式之一。
  - 若训练数据中 `Retrieved Context` 为空，优先检查 `--graph_api_url`、`--top_k` 以及网络连通性。
- **Tokenizer 相关问题**：
  - 若加载 tokenizer 失败（例如 vLLM 只在远端有权重），可以通过 `--tokenizer_name` 显式指定一个可从 HuggingFace 下载的 tokenizer repo。

更多训练细节可参考 `src/train/train.py` 与 `src/train/README.md` 中的注释及说明。

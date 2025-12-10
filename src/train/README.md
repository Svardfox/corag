# Chain of RAG Training Module

此目录包含用于 Chain of RAG 模型全量微调（Full Fine-tuning）的代码。

## 目录结构

*   `prepare_training_data.py`: 数据准备脚本。负责将原始数据集转换为包含“检索-生成”链条的交错训练数据。
*   `train.py`: 核心训练脚本。基于 Hugging Face Trainer，实现了支持 Qwen 等模型的微调逻辑及特定的 Masking 策略。
*   `run_training.sh`: 启动脚本模板。串联数据准备和训练过程。

## 1. 数据准备 (`prepare_training_data.py`)

此脚本会将 `corag/multihopqa` 等数据集转换为 JSONL 格式。关键功能是**实时调用 Graph API**，为每个子查询 (`SubQuery`) 获取真实的检索文档作为上下文 (`Context`)。

### 参数说明

| 参数名 | 默认值 | 含义 |
| :--- | :--- | :--- |
| `--dataset` | `corag/multihopqa` | **数据集名称**。从 Hugging Face Hub 加载的数据集路径。 |
| `--task` | `2wikimultihopqa` | **任务名称**。子任务类别，如 `hotpotqa`, `musique`。 |
| `--split` | `train` | **数据集划分**。使用训练集 (`train`)、验证集 (`validation`) 或测试集。 |
| `--output_file` | `data/train_with_graph_retrieval.jsonl` | **输出文件路径**。生成的包含检索上下文的训练数据保存位置。 |
| `--graph_api_url` | `http://localhost:8023/retrieve` | **检索 API 地址**。您的图检索服务地址，用于获取相关文档。 |
| `--retrieve` | (Flag) | **执行检索**。开启此开关将实际调用 API；若不加则只生成结构但不含检索内容（用于快速测试流程）。 |
| `--top_k` | `3` | **Top-K**。每个子查询保留多少个检索到的相关文档。 |

### 运行示例

```bash
python3 src/train/prepare_training_data.py \
    --output_file data/my_train_data.jsonl \
    --retrieve \
    --top_k 3
```

---

## 2. 模型训练 (`train.py` / `run_training.sh`)

此脚本加载准备好的数据，使用 Qwen 等基础模型进行微调。它实现了 **Chain-of-RAG Masking**，即：
*   **计算 Loss**: `SubQuery`, `SubAnswer`, `Final Answer` (模型生成的思考与回答)
*   **不计算 Loss (Masked)**: `User Query`, `Retrieved Context` (外部输入的知识)

### 参数说明 (在 `run_training.sh` 中设置)

该脚本接受标准的 [Hugging Face TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)。

| 参数名 | 推荐/示例值 | 含义 |
| :--- | :--- | :--- |
| `--model_name_or_path` | `Qwen/Qwen2.5-7B-Instruct` | **模型路径**。预训练模型名称或本地路径。 |
| `--train_file` | `data/train_with_graph_retrieval.jsonl` | **训练数据**。由数据准备脚本生成的 JSONL 文件。 |
| `--output_dir` | `output/corag_qwen_finetuned` | **输出目录**。保存 Checkpoints 和 Logs 的位置。 |
| `--nproc_per_node` | `8` | **GPU 数量**。参与训练的 GPU 卡数。 |
| `--num_train_epochs` | `3` | **训练轮数**。 |
| `--per_device_train_batch_size` | `4` | **单卡 Batch Size**。根据显存大小调整。 |
| `--gradient_accumulation_steps` | `2` | **梯度累积**。累积几步更新一次参数。`Global Batch = Batch * GPU数 * 累积步`。 |
| `--learning_rate` | `2e-5` | **学习率**。微调常用 1e-5 ~ 2e-5。 |
| `--max_len` | `4096` | **最大长度**。序列最大 Token 数，超长截断。 |
| `--bf16` | `True` | **BF16 精度**。推荐在 A100/H100/3090 等新显卡上开启以节省显存。 |
| `--save_steps` | `500` | **保存频率**。每多少步保存一次模型。 |

### 运行示例

请直接执行启动脚本：

```bash
bash src/train/run_training.sh
```

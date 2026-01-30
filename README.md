## CoRAG: Chain-of-Retrieval Augmented Generation

English version | [中文说明](README_zh.md)

CoRAG supports three main workflows:

- **Dataset construction**: generate multi-step reasoning traces and enrich them with retrieved contexts.
- **Model training**: fine-tune an LLM with CoRAG-style supervision.
- **QA evaluation**: run batch and interactive QA with the CoRAG agent.

The code assumes the presence of:

- A **vLLM service** exposing an OpenAI-compatible API.
- A **retrieval service** (Graph / HTTP search API) providing textual contexts.

---

## Project Layout

```text
corag/
├── src/
│   ├── agent/                # CoRAG agent implementation and utilities
│   │   ├── corag_agent.py
│   │   └── agent_utils.py
│   ├── train/                # Training & data-preparation scripts
│   │   ├── train.py
│   │   ├── run_training.sh
│   │   ├── prepare_training_data.py
│   │   └── prepare_aligned_data.py
│   ├── config.py             # Global CLI configuration (Arguments dataclass)
│   ├── vllm_client.py        # OpenAI-compatible vLLM client
│   ├── data_utils.py         # Corpus loading & context formatting
│   ├── prompts.py            # Prompt templates for CoRAG
│   ├── utils.py              # Misc utilities (e.g., AtomicCounter)
│   └── logger_config.py      # Logging configuration
├── scripts/
│   ├── rejection_sampling_gen.py   # Sample CoRAG reasoning paths (rejection sampling)
│   └── custom_batch_eval.py        # Batch QA evaluation with CoRAG
├── interactive_demo.py       # Interactive CLI demo for single-question QA
├── requirements.txt          # Python dependencies
└── images/
    └── corag_framework.png   # CoRAG framework illustration
```

---

## Dependencies and Installation

- **Python**: 3.9+ (recommended)
- **GPU**: CUDA-capable GPU for vLLM and training

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Key libraries:

- **vllm** – for efficient LLM serving (OpenAI-compatible API).
- **transformers**, **torch** – core deep learning stack.
- **datasets** – dataset loading and processing.
- **deepspeed** – large-scale training support (ZeRO-3).

---

## External Services

CoRAG interacts with two external services:

- **vLLM service** for generation (OpenAI-style API).
- **Retrieval service** for fetching supporting documents.

Both are configured via command-line arguments defined in `src/config.py` (`Arguments` dataclass).

### vLLM Service API

The vLLM endpoint must implement the **OpenAI-compatible** REST API:

- Base URL (example): `http://localhost:8000/v1`
- Auth header: `Authorization: Bearer <API_KEY>`

#### Chat Completions

- **Endpoint**: `POST /v1/chat/completions`
- **Request example**:

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

- **Response shape (simplified)**:

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

CoRAG uses this API via `VllmClient` (`src/vllm_client.py`) in:

- `scripts/rejection_sampling_gen.py`
- `scripts/custom_batch_eval.py`
- `interactive_demo.py`
- `src/agent/corag_agent.py`

### Retrieval API Specification

The retrieval service is used to fetch supporting contexts given a sub-query.  
It is accessed via `graph_api_url` (e.g. `http://localhost:8023/retrieve`) and called by:

- `scripts/rejection_sampling_gen.py` (indirectly through `CoRagAgent`)
- `src/train/prepare_training_data.py`
- `scripts/custom_batch_eval.py`
- `interactive_demo.py`

To make all components work smoothly, the retrieval API **should** follow this contract.

#### Request

- **Method**: `POST`
- **Path**: `/retrieve` (configurable via `graph_api_url`)
- **Body (recommended)**:

```json
{
  "query": "What is the capital of France?",
  "top_k": 20
}
```

- **Fields**:
  - **`query`** (string, required): the natural language query or sub-query.
  - **`top_k`** (integer, optional): maximum number of documents to return.  
    Scripts such as `prepare_training_data.py` will also enforce their own `--top_k`, so your service can either respect or ignore this field.

#### Response (recommended format)

The recommended response is a JSON object with a `chunks` field:

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

Each element in `chunks`:

- **`id`** (string, optional but recommended): document identifier.
- **`contents`** (string, recommended) or **`content`** / **`text`**: the actual passage text.

#### Other supported shapes

The preprocessing scripts are robust and will also accept:

- A **list of strings**:

```json
[
  "Paris is the capital and most populous city of France.",
  "France is a country located in Western Europe ..."
]
```

- A **list of objects** with `contents` / `content` / `text` fields:

```json
[
  {"text": "Paris is the capital and most populous city of France."},
  {"text": "France is a country located in Western Europe ..."}
]
```

- An **object** with one of the supported list keys:

```json
{
  "results": [
    {"contents": "..."}, 
    {"contents": "..."}
  ]
}
```

`prepare_training_data.py` normalizes the response by:

- Looking for list-valued keys in `["chunks", "data", "results", "docs", "passages"]`.
- Falling back to wrapping a single object into a list when needed.
- Extracting each document’s text from `contents` → `content` → `text` → `str(doc)`.

As long as your API conforms to one of the above patterns, CoRAG will be able to consume it.

---

## Configuration via `src/config.py`

Global runtime configuration is defined by the `Arguments` dataclass in `src/config.py`, which extends HuggingFace `TrainingArguments`.  
Important fields commonly used by both training and evaluation scripts:

- **Model & vLLM**
  - `--vllm_api_base`: vLLM API base URL (default: `http://localhost:8000/v1`).
  - `--vllm_api_key`: vLLM API key (default: `token-123`).
  - `--vllm_model`: model name / ID (optional; auto-detected if omitted).
  - `--tokenizer_name`: tokenizer repo/path (defaults to the vLLM model ID).
- **Retrieval**
  - `--graph_api_url`: retrieval API URL (e.g. `http://localhost:8023/retrieve`).
  - `--corpus_file`: corpus JSON file used to map document IDs to text (for some modes).
  - `--num_contexts`: maximum number of contexts per example (default: 20).
- **Agent behavior**
  - `--max_path_length`: maximum reasoning steps (depth).
  - `--decode_strategy`: `greedy` | `tree_search` | `best_of_n`.
  - `--sample_temperature`: sampling temperature for non-greedy strategies.
  - `--best_n`: number of sampled paths for `best_of_n`.
- **Execution**
  - `--num_threads`: number of worker threads for evaluation or retrieval.
  - `--max_len`: max input length for final-answer prompts.

You can pass these arguments directly to:

- `scripts/custom_batch_eval.py`
- `interactive_demo.py`
- `src/train/train.py` (via `run_training.sh`)

---

## Workflow 1: Dataset Construction

The dataset construction pipeline has three stages:

1. **Prepare a source QA dataset** with `query` and `answers` fields.
2. **Run rejection sampling** to generate CoRAG reasoning paths.
3. **Retrieve contexts and align supervision types** for training.

### 1.1 Prepare Source Dataset

You need a dataset where each example contains at least:

- **`query`**: the user question.
- **`answers`**: a list of acceptable reference answers (strings).

Typical options:

- Use an existing HF dataset (e.g. `corag/multihopqa` with `subset=2wikimultihopqa`).
- Convert your own dataset into a JSONL file with one JSON object per line:

```json
{"query": "Who is the CEO of OpenAI?", "answers": ["Sam Altman"]}
```

### 1.2 Rejection Sampling of Reasoning Paths

Use `scripts/rejection_sampling_gen.py` to let the CoRAG agent explore multiple reasoning paths and keep only those that lead to correct answers.

Example (HF dataset):

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

Output (`data/rejection_sampled_data.jsonl`) is a JSONL file; each line is a valid CoRAG sample containing (simplified):

- `id`: example ID.
- `query`: original question.
- `answers`: list of ground-truth answers.
- `generated_final_answer`: model’s final answer judged as correct.
- `steps`: list of reasoning steps, each with:
  - `subquery`
  - `subanswer`
  - optional `thought`.

At this stage, **retrieval contexts may not yet be fully materialized**; the next step will enrich them.

### 1.3 Enrich with Retrieved Contexts

`src/train/prepare_training_data.py` converts the rejection-sampled paths into ChatML-style conversations and **calls the retrieval API** to fill in contexts for each `subquery`.

Example:

```bash
python src/train/prepare_training_data.py \
  --dataset data/rejection_sampled_data.jsonl \
  --output_file data/train_with_context.jsonl \
  --graph_api_url http://localhost:8023/retrieve \
  --top_k 3 \
  --num_workers 10 \
  --retrieve
```

Input requirements:

- JSONL file from `rejection_sampling_gen.py`, with `query`, `steps`, and `generated_final_answer`.

Output (`data/train_with_context.jsonl`):

- Each line is an object with a `messages` field (ChatML format), roughly:
  - `{"role": "user", "content": "<original query>"}`
  - For each step:
    - `{"role": "assistant", "content": "SubQuery: <subquery>"}`  
    - `{"role": "observation", "content": "Retrieved Context:\nDoc 1: ...\nDoc 2: ..."}`
    - `{"role": "assistant", "content": "SubAnswer: <subanswer>"}`  
  - Final answer:
    - `{"role": "assistant", "content": "Final Answer: <answer>"}`.

This file is the **context-enriched conversation dataset**.

### 1.4 Build Aligned Supervision Data

`src/train/prepare_aligned_data.py` converts each conversation into multiple **aligned training samples** of different types:

- `subquery_generation`
- `subanswer_generation`
- `final_answer_generation`

Each sample is a ChatML conversation where:

- The **input** is encoded in the earlier messages.
- The **target** is a single assistant message (e.g. `SubQuery: ...`).

Example:

```bash
python src/train/prepare_aligned_data.py \
  --input_file data/train_with_context.jsonl \
  --output_file data/aligned_train.jsonl
```

Output (`data/aligned_train.jsonl`):

- JSONL file where each line has:
  - `type`: one of `subquery_generation`, `subanswer_generation`, `final_answer_generation`.
  - `messages`: ChatML message list (prompt + gold assistant answer).

This is the **final dataset** used for supervised fine-tuning.

---

## Workflow 2: Model Training

Training logic lives in `src/train/train.py`, which consumes the aligned dataset produced above.  
For convenience, we provide a default launcher script `src/train/run_training.sh`.

### 2.1 Example Training Command

The default `run_training.sh` runs multi-GPU training with DeepSpeed ZeRO-3:

```bash
bash src/train/run_training.sh
```

Internally, it calls:

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

You should:

- Point `--train_file` to your processed dataset (e.g. `data/aligned_train.jsonl` or `data/train_with_graph_retrieval.jsonl`, depending on your setup).
- Adjust `--model_name_or_path`, `--output_dir`, DeepSpeed config, and training hyperparameters as needed.

### 2.2 Training Objective (High Level)

The CoRAG training objective focuses on **reasoning path generation**:

- The model is trained to output:
  - `SubQuery: ...`
  - `SubAnswer: ...`
  - `Final Answer: ...`
- User queries and raw retrieved documents are treated as **input only** (masked from loss).

This encourages the model to learn **multi-step reasoning over retrieved evidence** rather than memorizing answers.

---

## Workflow 3: QA Evaluation and Demos

CoRAG provides two main ways to evaluate the agent:

- **Batch evaluation** for large-scale experiments.
- **Interactive demo** for manual inspection and debugging.

### 3.1 Batch Evaluation (`scripts/custom_batch_eval.py`)

`custom_batch_eval.py` runs the CoRAG agent over a custom evaluation set, optionally computing **retrieval recall** metrics and comparing against a naive baseline.

Example:

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

#### Input format (`--eval_file`)

`eval_file` should be a JSON file containing a **list** of objects.  
At minimum:

- `question`: the input question.
- `answer`: (optional but recommended) the reference answer for logging.

For datasets like HotpotQA / MuSiQue, additional fields such as `paragraphs`, `context`, and `supporting_facts` are used internally to compute **golden facts** for recall evaluation.  
If `--calc_recall` is enabled, the script will:

- Compute CoRAG retrieval recall (micro-averaged over all golden chunks).
- Optionally compute naive retrieval recall if `--enable_naive_retrieval` is set.

#### Output (`--save_file`)

`save_file` is a JSON file with:

- A **summary object** at index 0 (average timing, micro recall, etc.).
- Per-example results, each including:
  - `question`, `answer`, `ground_truth`.
  - `reasoning_steps` (subqueries and subanswers).
  - `time` breakdown.
  - `corag_recall` and `naive_recall` (if enabled).

### 3.2 Interactive Demo (`interactive_demo.py`)

`interactive_demo.py` provides a CLI interface to test CoRAG on individual questions and inspect its internal reasoning steps.

Example:

```bash
python interactive_demo.py \
  --vllm_api_base http://localhost:8000/v1 \
  --vllm_api_key token-123 \
  --graph_api_url http://localhost:8023/retrieve \
  --corpus_file data/corpus.json \
  --max_path_length 3 \
  --decode_strategy greedy
```

You will see:

- Prompted input: `Enter your query:`
- Printed **intermediate steps**:
  - Sub-queries (`SubQuery`)
  - Sub-answers (`SubAnswer`)
  - Optional `Thought` and retrieved document IDs.
- The final CoRAG answer under `--- Final Answer ---`.

Use this script to:

- Verify that the retrieval API returns sensible contexts.
- Inspect whether the CoRAG path decomposition aligns with your expectations.
- Debug model behavior before running large-scale evaluations.

---

## Tips and Troubleshooting

- **Check vLLM service**:
  - Ensure `GET <vllm_api_base>/models` returns at least one model ID.
  - Use `--vllm_model` if auto-detection fails.
- **Check retrieval API**:
  - Manually `curl` your `/retrieve` endpoint to confirm it returns one of the supported formats.
  - If contexts are missing in training data, verify `--graph_api_url`, `--top_k`, and network connectivity.
- **Tokenizer issues**:
  - If loading the tokenizer fails (especially when vLLM serves a remote-only model), explicitly set `--tokenizer_name` to a valid HuggingFace repo ID.

For more advanced training details, see the comments in `src/train/train.py` and `src/train/README.md` (if present).

# CoRAG: Chain-of-Retrieval Augmented Generation

[English Version](README.md) | [中文版](README_zh.md)

This repository provides the implementation for [Chain-of-Retrieval Augmented Generation](https://arxiv.org/abs/2501.14342) (CoRAG). The codebase supports inference (Agentic RAG), training, and dataset construction.

## Project Structure

```text
corag/
├── src/
│   ├── agent/           # CoRAG agent logic (Path generation, Tree Search, etc.)
│   ├── inference/       # Inference utilities and metrics
│   ├── search/          # Retrieval clients (Graph API, HTTP)
│   ├── train/           # Training scripts (Data preparation, SFT loop)
│   ├── config.py        # Global configuration arguments
│   └── vllm_client.py   # Client for communicating with VLLM server
├── scripts/
│   ├── custom_batch_eval.py    # Main script for batch evaluation
│   ├── rejection_sampling_gen.py # Script for generating reasoning traces
│   └── start_vllm_server.sh    # Helper to launch VLLM
├── interactive_demo.py  # Interactive CLI for testing single queries
└── requirements.txt     # Python dependencies
```

# Part 1: Inference (Agentic RAG)

This section details how to run the CoRAG agent for inference, including setting up the environment, configuring parameters, running the interactive demo, and performing batch evaluations.

## 1. Environment Setup

### Software Requirements
Install the necessary Python dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- `vllm`: For efficient LLM serving.
- `transformers`, `torch`: Core DL libraries.
- `flash-attn`: For accelerated attention.

### Server Requirements
CoRAG relies on two external services: an LLM server and a Retrieval server (Graph API).

#### A. LLM Server (VLLM)
You need to have a VLLM server running to handle generation requests.
You can use the provided script to start a VLLM server:
```bash
# Starts VLLM on port 8000 with a specified model
bash scripts/start_vllm_server.sh <model_name_or_path>
```
Default URL: `http://localhost:8000/v1`

#### B. Retrieval Server
You need a service to handle document retrieval. This should be a custom Graph API.
Default URL: `http://localhost:8023/retrieve`

> [!NOTE]
> **Custom Search API Response Format**
> The retrieval server must accept a JSON POST with `{"query": "your query"}` and return a JSON list of strings (passages).
> 
> **Request:**
> `POST /retrieve`
> ```json
> {
>   "query": "What is the capital of France?"
> }
> ```
> 
> **Response:**
> ```json
> [
>   "Paris is the capital and most populous city of France.",
>   "France is a country located in Western Europe."
> ]
> ```

---

## 2. Configuration Parameters

The system is highly configurable via command-line arguments. Key parameters in `src/config.py` include:

### Model & API Configuration
- `--vllm_api_base`: URL for the VLLM API (default: `http://localhost:8000/v1`).
- `--vllm_api_key`: API key for VLLM (default: `token-123`).
- `--vllm_model`: Specific model name to use.
- `--graph_api_url`: URL for the retrieval service (e.g., `http://localhost:8023/retrieve`).
- `--corpus_file`: Path to the corpus JSON file to map retrieved IDs to text.

### Inference Strategy
- `--task_desc`: Description of the task for the agent (default: "answer multi-hop questions").
- `--max_path_length`: Maximum number of reasoning steps (depth) allowed.
- `--decode_strategy`: Strategy for path generation. Options:
  - `greedy`: Always picks the most likely next step.
  - `tree_search`: Explores multiple branches.
  - `best_of_n`: Samples `n` paths and selects the best one.
- `--sample_temperature`: Temperature for sampling during tree/best-of-n search.
- `--best_n`: Number of paths to sample for `best_of_n` strategy.

### Retrieval Settings
- `--num_contexts`: Number of documents to retrieve per step (default: 20).
- `--num_threads`: Number of threads for parallel retrieval in batch eval (default: 32).

---

## 3. Interactive Demo Script

The `interactive_demo.py` script allows you to chat with the CoRAG agent in real-time. It visualizes the agent's "thinking" process, including intermediate queries, retrieved documents, and thoughts.

### Usage
```bash
python interactive_demo.py \
    --vllm_api_base http://localhost:8000/v1 \
    --graph_api_url http://localhost:8023/retrieve \
    --corpus_file data/corpus.json \
    --max_path_length 3 \
    --decode_strategy greedy
```

### Features
- **Real-time Reasoning**: Displays each step of the sub-query and sub-answer generation.
- **Traceability**: Shows exactly which documents were retrieved at each step.
- **Flexible Testing**: Allows quick testing of different questions without running a full benchmark.

---

## 4. Batch Evaluation Script

For large-scale evaluation, use `scripts/custom_batch_eval.py`. This script processes a dataset of questions, calculates metrics (like Recall), and saves results.

### Usage
```bash
python scripts/custom_batch_eval.py \
    --eval_file data/my_questions.json \
    --save_file results/my_results.json \
    --corpus_file data/corpus.json \
    --calc_recall \
    --num_threads 16
```

### Input Format (`--eval_file`)
The input should be a JSON list of dictionaries. Each dictionary must contain at least a `question` field.
```json
[
    {
        "question": "Who is the director of the movie related to...",
        "answer": "Target Answer"  // Optional, for reference
    }
]
```

### Key Arguments
- `--eval_file`: Path to the input JSON dataset.
- `--save_file`: Path to save the output JSON results.
- `--calc_recall`: If set, calculates retrieval recall metrics (requires `answer` or golden facts in input).
- `--enable_naive_retrieval`: If set, also runs a baseline "naive" retrieval (single step) for comparison.
- `--num_threads`: Number of concurrent threads for faster processing.

### Output
The output file contains:
1. **Summary Object**: Aggregate stats at the top (Average time, Micro Recall, etc.).
2. **Per-Item Results**: Detailed trace for each question, including:
   - `reasoning_steps`: List of (SubQuery, SubAnswer).
   - `corag_recall`: Hits and Total recall for CoRAG.
   - `naive_recall`: Hits and Total recall for Naive baseline (if enabled).

# Part 2: Training (SFT)

This section describes how to fine-tune a model using the CoRAG methodology.

## Training Process

The core training logic is implemented in `src/train/train.py`. We provide a shell script `src/train/run_training.sh` to simplify the process.

### Features
*   **CoRAG Masking Strategy**: The model is trained to generate the *reasoning path* (Sub-Queries, Sub-Answers, Final Answer). We calculate loss only on these generated parts. The *User Query* and *Retrieved Documents* (external knowledge) are masked (loss=0).
*   **DeepSpeed Support**: The default configuration uses DeepSpeed ZeRO-3 for efficient memory usage on multi-GPU setups.

### Running the Training
```bash
bash src/train/run_training.sh
```

### Key Training Arguments
The script uses standard Hugging Face `TrainingArguments`. Important parameters in `src/train/run_training.sh` include:

*   `--train_file`: Path to the prepared JSONL training data.
*   `--model_name_or_path`: Base model to fine-tune (e.g., `Qwen/Qwen2.5-7B-Instruct`).
*   `--output_dir`: Directory to save checkpoints and logs.
*   `--learning_rate`: Default is `2e-5`.
*   `--num_train_epochs`: Default is `3`.
*   `--gradient_accumulation_steps`: Adjust this to achieve your desired effective batch size.
*   `--bf16`: Strongly recommended for newer GPUs (A100/H100) to save memory and improve stability.

# Part 3: Dataset Construction

This section includes tools for creating training datasets, either by formatting existing benchmarks or by generating new reasoning paths through rejection sampling.

## Rejection Sampling Strategy

To train the CoRAG model, we need data that consists of (Question, Reasoning Path, Answer). Since standard datasets only provide (Question, Answer), we use the CoRAG agent to explore multiple reasoning paths and keep only those that lead to the correct answer.

### Script: `scripts/rejection_sampling_gen.py`

This script runs the CoRAG agent on a dataset, samples multiple paths per question, and validates the final answer against the ground truth.

#### Usage

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

#### Key Parameters
- `--n_samples`: Number of reasoning paths to sample for each question.
- `--temperature`: Sampling temperature for diversity.
- `--output_file`: Where to save the valid paths.
- `--max_path_length`: Maximum depth of reasoning.

#### Output Format
The output is a JSONL file where each line is a valid sample containing:
- `query`: The original question.
- `generated_final_answer`: The agent's correct answer.
- `steps`: A list of reasoning steps, each containing `subquery`, `subanswer`, and `thought`.


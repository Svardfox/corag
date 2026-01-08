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


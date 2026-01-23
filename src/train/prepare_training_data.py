import os
import sys
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
# Add project root to path (for imports like 'from src.xxx import yyy')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from search.search_utils import search_by_graph_api

import concurrent.futures

def process_example(example, args):
    # 1. 优先支持 "steps" 格式 (来自 rejection_sampling_gen.py 的输出)
    if 'steps' in example and 'query' in example:
        conversation = [
            {"role": "user", "content": example['query']}
        ]
        
        for step in example['steps']:
            sq = step.get('subquery', '')
            sa = step.get('subanswer', '')
            if not sq:
                continue
                
            conversation.append({"role": "assistant", "content": f"SubQuery: {sq}"})
            docs_text = retrieve_context(sq, args)
            conversation.append({"role": "observation", "content": f"Retrieved Context:\n{docs_text}"})
            conversation.append({"role": "assistant", "content": f"SubAnswer: {sa}"})

        # 确定 Final Answer
        final_answer = ""
        if 'answers' in example and example['answers']:
            final_answer = example['answers'][0]
        elif 'answer' in example:
            final_answer = example['answer']
        elif 'generated_final_answer' in example:
            final_answer = example['generated_final_answer']
            
        conversation.append({"role": "assistant", "content": f"Final Answer: {final_answer}"})
        return {"messages": conversation}

    # 2. 支持 "messages" 格式 (用于对已有对话进行回填)
    if 'messages' in example:
        conversation = []
        initial_query = ""
        for msg in example['messages']:
            if msg['role'] == 'user':
                initial_query = msg['content']
                break
        
        conversation.append({"role": "user", "content": initial_query})
        
        msgs = example['messages']
        i = 0
        while i < len(msgs):
            msg = msgs[i]
            if msg['role'] == 'assistant' and msg['content'].startswith("SubQuery:"):
                sq = msg['content'].replace("SubQuery:", "").strip()
                conversation.append({"role": "assistant", "content": f"SubQuery: {sq}"})
                
                # 重新检索并回填 observation
                docs_text = retrieve_context(sq, args)
                conversation.append({"role": "observation", "content": f"Retrieved Context:\n{docs_text}"})
                
                # 跳过原有的 observation，寻找下一个 SubAnswer
                i += 1
                while i < len(msgs):
                    if msgs[i]['role'] == 'assistant' and msgs[i]['content'].startswith("SubAnswer:"):
                        sa = msgs[i]['content'].replace("SubAnswer:", "").strip()
                        conversation.append({"role": "assistant", "content": f"SubAnswer: {sa}"})
                        break
                    i += 1
            elif msg['role'] == 'assistant' and msg['content'].startswith("Final Answer:"):
                fa = msg['content'].replace("Final Answer:", "").strip()
                conversation.append({"role": "assistant", "content": f"Final Answer: {fa}"})
            i += 1
        return {"messages": conversation}

    # 3. 原始数据格式 (query, subqueries, subanswers, answers)
    conversation = [
        {"role": "user", "content": example['query']}
    ]
    
    subqueries = example.get('subqueries', [])
    subanswers = example.get('subanswers', [])
    
    for sq, sa in zip(subqueries, subanswers):
        conversation.append({"role": "assistant", "content": f"SubQuery: {sq}"})
        docs_text = retrieve_context(sq, args)
        conversation.append({"role": "observation", "content": f"Retrieved Context:\n{docs_text}"})
        conversation.append({"role": "assistant", "content": f"SubAnswer: {sa}"})

    final_answer = example['answers'][0] if example.get('answers') else example.get('answer', "")
    conversation.append({"role": "assistant", "content": f"Final Answer: {final_answer}"})
    
    return {"messages": conversation}

def retrieve_context(sq, args):
    if args.retrieve:
        try:
            results = search_by_graph_api(sq, args.graph_api_url)
            if isinstance(results, dict):
                for key in ['data', 'results', 'docs', 'passages', 'chunks']:
                    if key in results and isinstance(results[key], list):
                        results = results[key]
                        break
                else:
                    results = [results] if results else []
            elif not isinstance(results, list):
                results = []
                
            docs_text = ""
            for i, res in enumerate(results[:args.top_k]):
                if isinstance(res, str):
                    content = res
                else:
                    content = res.get('contents') or res.get('content') or res.get('text') or str(res)
                docs_text += f"Doc {i+1}: {content}\n"
            
            if not docs_text:
                docs_text = "No relevant information found."
        except Exception as e:
            print(f"Retrieval failed for {sq}: {e}")
            docs_text = "Retrieval failed."
    else:
        docs_text = "[Retrieval Skipped in Dry Run]"
    return docs_text

def prepare_data(args):
    # Path correction: if file doesn't exist and doesn't start with /, try adding /
    dataset_path = args.dataset
    if not os.path.exists(dataset_path) and not dataset_path.startswith('/'):
        alt_path = '/' + dataset_path
        if os.path.exists(alt_path):
            print(f"File found at {alt_path}, using absolute path.")
            dataset_path = alt_path

    # Load dataset
    if dataset_path.endswith('.jsonl') or os.path.exists(dataset_path):
        if not os.path.exists(dataset_path):
            print(f"Error: Local file not found at {dataset_path}")
            sys.exit(1)
            
        print(f"Loading local dataset from {dataset_path}...")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            ds = [json.loads(line) for line in f]
    else:
        print(f"Loading dataset {dataset_path}, split {args.split}...")
        try:
            ds = load_dataset('json', data_files=dataset_path, split=args.split)
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            sys.exit(1)
    
    output_data = []
    print(f"Starting processing with {args.num_workers} workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_example, ex, args): i for i, ex in enumerate(ds)}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(ds), desc="Processing examples"):
            try:
                result = future.result()
                output_data.append(result)
            except Exception as e:
                print(f"Error processing example: {e}")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(output_data)} examples to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="corag/multihopqa")
    parser.add_argument("--task", type=str, default="2wikimultihopqa")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_file", type=str, default="data/train_with_graph_retrieval.jsonl")
    parser.add_argument("--graph_api_url", type=str, default="http://localhost:8025/retrieve")
    parser.add_argument("--retrieve", action="store_true", help="Whether to perform actual retrieval")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=16, help="Number of concurrent threads for retrieval")
    
    args = parser.parse_args()
    prepare_data(args)

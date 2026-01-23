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
    # Support "steps" format (from rejection_sampling_gen.py output)
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

        # Final Answer
        final_answer = ""
        if 'answers' in example and example['answers']:
            final_answer = example['answers'][0]
        elif 'answer' in example:
            final_answer = example['answer']
        elif 'generated_final_answer' in example:
            final_answer = example['generated_final_answer']
            
        conversation.append({"role": "assistant", "content": f"Final Answer: {final_answer}"})
        return {"messages": conversation}

    # Check if example is already in "messages" format
    if 'messages' in example:
        # ... (rest of messages logic) ...

def retrieve_context(sq, args):
    if args.retrieve:
        # Call Graph API
        try:
            results = search_by_graph_api(sq, args.graph_api_url)
            # Handle case where API returns a dict (e.g. {"data": [...]}) instead of list
            if isinstance(results, dict):
                # Try to find a list in common keys
                for key in ['data', 'results', 'docs', 'passages']:
                    if key in results and isinstance(results[key], list):
                        results = results[key]
                        break
                else:
                    results = [results] if results else []
            elif not isinstance(results, list):
                results = []
                
            # Format docs (simple concatenation or structured)
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
    # Load dataset
    if os.path.exists(args.dataset):
        print(f"Loading local dataset from {args.dataset}...")
        with open(args.dataset, 'r', encoding='utf-8') as f:
            ds = [json.loads(line) for line in f]
    else:
        print(f"Loading dataset {args.dataset}, split {args.split}...")
        ds = load_dataset('json', data_files=args.dataset, split=args.split)
    
    output_data = []
    
    print(f"Starting processing with {args.num_workers} workers...")
    
    # Use ThreadPoolExecutor for parallelism
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # Map process_example to each item in the dataset
        # Use simple lambda or partial to pass args
        futures = {executor.submit(process_example, ex, args): i for i, ex in enumerate(ds)}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(ds), desc="Processing examples"):
            try:
                result = future.result()
                output_data.append(result)
            except Exception as e:
                print(f"Error processing example: {e}")

    # Save to file
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

import os
import sys
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from search.search_utils import search_by_graph_api

def prepare_data(args):
    # Load dataset
    print(f"Loading dataset {args.dataset}, split {args.split}...")
    ds = load_dataset(args.dataset, args.task, split=args.split)
    
    output_data = []
    
    for example in tqdm(ds, desc="Processing examples"):
        # Initial User Query
        conversation = [
            {"role": "user", "content": example['query']}
        ]
        
        subqueries = example.get('subqueries', [])
        subanswers = example.get('subanswers', [])
        
        # Interleave sub-steps
        for sq, sa in zip(subqueries, subanswers):
            # 1. Model generates SubQuery -> Add to conversation (Assistant)
            conversation.append({"role": "assistant", "content": f"SubQuery: {sq}"})
            
            # 2. Retrieve Documents (System/Observation)
            if args.retrieve:
                # Call Graph API
                # print(f"Retrieving for: {sq}") # verbose
                try:
                    results = search_by_graph_api(sq, args.graph_api_url)
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

            conversation.append({"role": "observation", "content": f"Retrieved Context:\n{docs_text}"})
            
            # 3. Model generates SubAnswer -> Add to conversation (Assistant)
            conversation.append({"role": "assistant", "content": f"SubAnswer: {sa}"})

        # Final Answer
        final_answer = example['answers'][0] if example['answers'] else ""
        conversation.append({"role": "assistant", "content": f"Final Answer: {final_answer}"})
        
        output_data.append({"messages": conversation})

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
    parser.add_argument("--graph_api_url", type=str, default="http://localhost:8023/retrieve")
    parser.add_argument("--retrieve", action="store_true", help="Whether to perform actual retrieval")
    parser.add_argument("--top_k", type=int, default=3)
    
    args = parser.parse_args()
    prepare_data(args)

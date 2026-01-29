import os
import sys
import json
import argparse
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

try:
    from search.search_utils import search_by_graph_api
except ImportError:
    print("Error: Could not import search.search_utils. Please ensure you are running from the project root.")
    sys.exit(1)

import concurrent.futures

def retrieve_context(sq, args):
    """Actually call the search API to fetch context."""
    if args.retrieve:
        try:
            results = search_by_graph_api(sq, args.graph_api_url)
            
            # Handle various API response formats
            if isinstance(results, dict):
                for key in ['chunks', 'data', 'results', 'docs', 'passages']:
                    if key in results and isinstance(results[key], list):
                        results = results[key]
                        break
                else:
                    # If no list found, wrap the single result if it exists
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
                return "No relevant information found."
            return docs_text
        except Exception as e:
            return f"Retrieval failed: {e}"
    else:
        return "[Retrieval Skipped in Dry Run]"

def process_example(example, args):
    """Process a single example line into conversation format."""
    conversation = []
    
    # 1. Get initial query
    query = example.get('query', '')
    if not query:
        return None
        
    conversation.append({"role": "user", "content": query})
    
    # 2. Process intermediate steps from "steps" list
    steps = example.get('steps', [])
    for step in steps:
        sq = step.get('subquery', '')
        sa = step.get('subanswer', '')
        
        if sq:
            # Add SubQuery
            conversation.append({"role": "assistant", "content": f"SubQuery: {sq}"})
            
            # Add Retrieved Context (This is where the API is called)
            context = retrieve_context(sq, args)
            conversation.append({"role": "observation", "content": f"Retrieved Context:\n{context}"})
            
            # Add SubAnswer
            conversation.append({"role": "assistant", "content": f"SubAnswer: {sa}"})

    # 3. Add Final Answer
    final_ans = example.get('generated_final_answer') or \
                (example.get('answers', [""])[0] if example.get('answers') else "")
    
    conversation.append({"role": "assistant", "content": f"Final Answer: {final_ans}"})
    
    return {"messages": conversation}

def prepare_data(args):
    # Path correction for absolute paths
    dataset_path = args.dataset
    if not os.path.exists(dataset_path) and not dataset_path.startswith('/'):
        if os.path.exists('/' + dataset_path):
            dataset_path = '/' + dataset_path

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        sys.exit(1)

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        ds = [json.loads(line) for line in f if line.strip()]
    
    output_data = []
    print(f"Processing {len(ds)} examples with {args.num_workers} workers...")
    
    # Use ThreadPoolExecutor for parallel API calls
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_example, ex, args): i for i, ex in enumerate(ds)}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(ds), desc="Progress"):
            try:
                result = future.result()
                if result:
                    output_data.append(result)
            except Exception as e:
                print(f"Error processing example: {e}")

    # Save to file
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Successfully saved {len(output_data)} examples to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to input jsonl file")
    parser.add_argument("--output_file", type=str, default="data/train_with_context.jsonl", help="Output path")
    parser.add_argument("--graph_api_url", type=str, default="http://localhost:8025/retrieve")
    # 默认开启检索，如需关闭可在代码中修改或后续扩展为 --no_retrieve 参数
    parser.add_argument("--retrieve", action="store_true", default=True, help="Actually perform retrieval")
    parser.add_argument("--top_k", type=int, default=3, help="Max documents per step")
    parser.add_argument("--num_workers", type=int, default=10, help="Parallel threads")
    
    args = parser.parse_args()
    prepare_data(args)

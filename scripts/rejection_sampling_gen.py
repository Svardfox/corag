
import os
import sys
import json
import argparse
import random
import concurrent.futures
from typing import List, Dict, Any
from tqdm import tqdm
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from datasets import load_dataset
from vllm_client import VllmClient
from agent.corag_agent import CoRagAgent
from inference.qa_utils import normalize_squad

def check_answer(prediction: str, ground_truths: List[str]) -> bool:
    """
    Check if the prediction matches any of the ground truths.
    Uses normalized exact match and containment.
    """
    if not prediction:
        return False
    
    norm_pred = normalize_squad(prediction)
    
    for gt in ground_truths:
        norm_gt = normalize_squad(gt)
        if not norm_gt:
            continue
            
        # Exact match
        if norm_pred == norm_gt:
            return True
            
        # Containment (if prediction is short enough, avoiding FP like "the" in "the apple")
        # Ensure sufficient overlap? For now, we use simple inclusion if lengths are reasonable
        # Or just standard SQuAD EM approach + maybe "gt in pred"
        if norm_gt in norm_pred:
            return True
            
    return False

def process_example(example: Dict, agent: CoRagAgent, args: argparse.Namespace) -> List[Dict]:
    """
    Generate multiple paths for a single example and return valid ones.
    """
    query = example['query']
    ground_truths = example['answers'] if 'answers' in example else [example['answer']]
    
    valid_paths = []
    
    # We can try to parallelize the N samples if VLLM handles it, or just sequential
    # CoRagAgent.sample_path is sequential steps.
    # We can run N samples.
    
    # Since agent.sample_path is blocking and makes multiple LLM calls, 
    # we might want to just run loop here.
    
    # Note: CoRagAgent isn't fully thread safe if sharing same vllm client with shared tokenizer?
    # CoRagAgent has self.lock for tokenizer.
    
    for _ in range(args.n_samples):
        try:
            # Sample a path
            # Task desc is usually handled prompt-side or passed here
            task_desc = "Answer the following question based on the retrieved context."
            
            path = agent.sample_path(
                query=query,
                task_desc=task_desc,
                max_path_length=args.max_path_length,
                temperature=args.temperature,
                # Additional args?
            )
            
            # Generate final answer based on the path
            # We construct a mock "path" context for final answer generation
            # CoRagAgent.generate_final_answer takes a RagPath
            
            final_ans = agent.generate_final_answer(
                corag_sample=path,
                task_desc=task_desc,
                temperature=0.0 # Deterministic final answer
            )
            
            # Check correctness
            if check_answer(final_ans, ground_truths):
                # Construct the output object
                # We want to save the steps
                valid_path = {
                    "id": example.get('id', str(random.randint(0, 1000000))),
                    "query": query,
                    "answers": ground_truths,
                    "generated_final_answer": final_ans,
                    "steps": []
                }
                
                # Format steps from path
                # RagPath has lists: past_subqueries, past_subanswers, etc.
                for i in range(len(path.past_subqueries)):
                    step = {
                        "subquery": path.past_subqueries[i],
                        "subanswer": path.past_subanswers[i],
                        # "documents": path.past_documents[i] # This might be large
                        # We might want just the doc_ids if available, or just the reasoning
                    }
                    if path.past_thoughts and i < len(path.past_thoughts):
                        step["thought"] = path.past_thoughts[i]
                        
                    valid_path["steps"].append(step)
                
                valid_paths.append(valid_path)
                
        except Exception as e:
            print(f"Error sampling path for {query}: {e}")
            continue

    return valid_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="corag/multihopqa")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_file", type=str, default="data/rejection_sampled_data.jsonl")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct") 
    parser.add_argument("--graph_api_url", type=str, default="http://localhost:8023/retrieve")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of paths to sample per example")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_path_length", type=int, default=3)
    parser.add_argument("--max_examples", type=int, default=-1, help="Limit number of examples to process (for debugging)")
    
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset {args.dataset}...")
    ds = load_dataset(args.dataset, split=args.split)
    
    if args.max_examples > 0:
        ds = ds.select(range(args.max_examples))

    # Init VLLM Client
    # Assuming VllmClient interface: init(url, model, ...)
    # Adjust based on src/vllm_client.py
    vllm = VllmClient(url=args.vllm_url, model=args.model)
    
    # Init Agent
    # We pass empty list as corpus since we use graph_api
    dummy_corpus = [] 
    agent = CoRagAgent(vllm_client=vllm, corpus=dummy_corpus, graph_api_url=args.graph_api_url)

    output_data = []
    
    # Simple sequential loop for examples, but inside process_example we loop N times.
    # We could parallelize examples.
    
    print("Starting rejection sampling generation...")
    for example in tqdm(ds):
        valid = process_example(example, agent, args)
        if valid:
            output_data.extend(valid)
            # Flush periodically?
            
    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Saved {len(output_data)} valid paths to {args.output_file}")

if __name__ == "__main__":
    main()


import os
import sys
import json
import argparse
import random
import threading
import concurrent.futures
from typing import List, Dict, Any
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from datasets import load_dataset
from vllm_client import VllmClient, get_vllm_model_id
from agent.corag_agent import CoRagAgent

# normalize_squad is used for rejection sampling answer checking.
# Some environments may not include the optional `inference` package/module.
# We provide a lightweight fallback implementation compatible with SQuAD-style normalization.
try:
    from inference.qa_utils import normalize_squad  # type: ignore
except Exception:
    import re
    import string

    def normalize_squad(text: str) -> str:
        """Lower text and remove punctuation, articles and extra whitespace (SQuAD-style)."""
        if text is None:
            return ""

        def lower(s: str) -> str:
            return s.lower()

        def remove_punc(s: str) -> str:
            return "".join(ch for ch in s if ch not in set(string.punctuation))

        def remove_articles(s: str) -> str:
            return re.sub(r"\b(a|an|the)\b", " ", s)

        def white_space_fix(s: str) -> str:
            return " ".join(s.split())

        return white_space_fix(remove_articles(remove_punc(lower(text))))

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


def check_judge_model_available(judge_api_base: str, judge_api_key: str) -> bool:
    """Test if judge model API is available."""
    try:
        import requests
        headers = {"Authorization": f"Bearer {judge_api_key}"} if judge_api_key else {}
        response = requests.get(f"{judge_api_base.rstrip('/')}/models", headers=headers, timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def check_answer_with_llm_judge(
    prediction: str, 
    ground_truths: List[str], 
    query: str,
    judge_client: VllmClient,
    print_lock: threading.Lock = None
) -> bool:
    """
    Use LLM as judge to check if prediction is correct.
    """
    if not prediction:
        return False
    
    # Format ground truths
    gt_text = " or ".join([f'"{gt}"' for gt in ground_truths if gt])
    if not gt_text:
        return False
    
    prompt = f"""You are an expert evaluator. Determine if the predicted answer correctly answers the question.

Question: {query}

Ground truth answer(s): {gt_text}

Predicted answer: {prediction}

Evaluate whether the predicted answer is correct. Consider:
1. Semantic equivalence (same meaning even if wording differs)
2. Factual correctness
3. Completeness (if multiple answers are expected)

Respond with only "YES" if the answer is correct, or "NO" if it is incorrect. Do not provide any explanation."""

    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = judge_client.call_chat(messages=messages, temperature=0.0, max_tokens=10)
        response_upper = response.strip().upper()
        # Check if response starts with "YES" or contains "YES" before "NO"
        if response_upper.startswith("YES"):
            return True
        # Check if "YES" appears before "NO" in the response
        yes_pos = response_upper.find("YES")
        no_pos = response_upper.find("NO")
        if yes_pos != -1 and (no_pos == -1 or yes_pos < no_pos):
            return True
        return False
    except Exception as e:
        if print_lock:
            with print_lock:
                print(f"  [WARNING] LLM judge error: {e}, falling back to string matching")
        else:
            print(f"  [WARNING] LLM judge error: {e}, falling back to string matching")
        return check_answer(prediction, ground_truths)

class VerboseCoRagAgent:
    """Wrapper around CoRagAgent to add verbose logging."""
    def __init__(self, agent: CoRagAgent, verbose: bool = False):
        self.agent = agent
        self.verbose = verbose
    
    def __getattr__(self, name):
        return getattr(self.agent, name)
    
    def sample_path(self, *args, **kwargs):
        if not self.verbose:
            return self.agent.sample_path(*args, **kwargs)
        
        # Import here to avoid circular imports
        from prompts import get_generate_subquery_prompt, get_generate_intermediate_answer_prompt
        from search.search_utils import search_by_graph_api
        
        query = kwargs.get('query') or args[0]
        task_desc = kwargs.get('task_desc') or args[1]
        past_subqueries = kwargs.get('past_subqueries', [])
        past_subanswers = kwargs.get('past_subanswers', [])
        
        # Monkey patch to intercept calls
        original_get_subanswer = self.agent._get_subanswer_and_doc_ids
        original_call_chat = self.agent.vllm_client.call_chat
        
        def verbose_get_subanswer(subquery, max_message_length=4096):
            print(f"\n    [RETRIEVAL] Query: {subquery}")
            if self.agent.graph_api_url:
                print(f"    [RETRIEVAL] API URL: {self.agent.graph_api_url}")
                raw_results = search_by_graph_api(subquery, self.agent.graph_api_url)
                print(f"    [RETRIEVAL] Raw response type: {type(raw_results)}")
                if isinstance(raw_results, dict):
                    print(f"    [RETRIEVAL] Response keys: {list(raw_results.keys())}")
                    if 'chunks' in raw_results:
                        print(f"    [RETRIEVAL] Raw chunks count: {len(raw_results['chunks'])}")
            subanswer, doc_ids, documents = original_get_subanswer(subquery, max_message_length)
            print(f"    [RETRIEVAL] Retrieved {len(documents)} documents")
            for i, doc in enumerate(documents):
                doc_preview = doc[:300] + "..." if len(doc) > 300 else doc
                print(f"      Doc {i+1} ({len(doc)} chars): {doc_preview}")
            
            # Print intermediate answer prompt
            prompt_messages = get_generate_intermediate_answer_prompt(subquery, documents)
            print(f"    [PROMPT] Intermediate Answer Prompt:")
            for msg in prompt_messages:
                content = msg['content']
                if len(content) > 1000:
                    print(f"      {msg['role']}: {content[:500]}...\n      ... (truncated, total {len(content)} chars)")
                else:
                    print(f"      {msg['role']}: {content}")
            
            return subanswer, doc_ids, documents
        
        def verbose_call_chat(messages, **chat_kwargs):
            # Check if this is a subquery generation call
            if len(messages) == 1 and messages[0].get('role') == 'user':
                content = messages[0].get('content', '')
                if 'generate a new simple follow-up question' in content or 'Intermediate query' in content:
                    print(f"\n    [PROMPT] SubQuery Generation Prompt:")
                    if len(content) > 1000:
                        print(f"      {content[:500]}...\n      ... (truncated, total {len(content)} chars)")
                    else:
                        print(f"      {content}")
            
            result = original_call_chat(messages, **chat_kwargs)
            return result
        
        # Temporarily replace methods
        self.agent._get_subanswer_and_doc_ids = verbose_get_subanswer
        self.agent.vllm_client.call_chat = verbose_call_chat
        
        try:
            result = self.agent.sample_path(*args, **kwargs)
        finally:
            # Restore original methods
            self.agent._get_subanswer_and_doc_ids = original_get_subanswer
            self.agent.vllm_client.call_chat = original_call_chat
        
        return result
    
    def generate_final_answer(self, *args, **kwargs):
        if not self.verbose:
            return self.agent.generate_final_answer(*args, **kwargs)
        
        from prompts import get_generate_final_answer_prompt
        
        corag_sample = kwargs.get('corag_sample') or args[0]
        task_desc = kwargs.get('task_desc') or args[1]
        documents = kwargs.get('documents')
        
        # Print final answer prompt
        prompt_messages = get_generate_final_answer_prompt(
            query=corag_sample.query,
            past_subqueries=corag_sample.past_subqueries or [],
            past_subanswers=corag_sample.past_subanswers or [],
            task_desc=task_desc,
            documents=documents
        )
        print(f"\n    [PROMPT] Final Answer Prompt:")
        for msg in prompt_messages:
            content = msg['content']
            if len(content) > 1000:
                print(f"      {msg['role']}: {content[:500]}...\n      ... (truncated, total {len(content)} chars)")
            else:
                print(f"      {msg['role']}: {content}")
        
        return self.agent.generate_final_answer(*args, **kwargs)


def process_example(example: Dict, agent: CoRagAgent, args: argparse.Namespace, print_lock: threading.Lock = None) -> List[Dict]:
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
    
    if print_lock:
        with print_lock:
            print(f"\n[Query] {query}")
            print(f"[Ground Truth] {ground_truths}")
    else:
        print(f"\n[Query] {query}")
        print(f"[Ground Truth] {ground_truths}")
    
    for sample_idx in range(args.n_samples):
        if print_lock:
            with print_lock:
                print(f"\n--- Sampling path {sample_idx + 1}/{args.n_samples} ---")
        else:
            print(f"\n--- Sampling path {sample_idx + 1}/{args.n_samples} ---")
        try:
            # Sample a path
            # Task desc is usually handled prompt-side or passed here
            task_desc = "Answer the following question based on the retrieved context."
            
            path = agent.sample_path(
                query=query,
                task_desc=task_desc,
                max_path_length=args.max_path_length,
                max_message_length=args.max_message_length if args.max_message_length > 0 else None,
                temperature=args.temperature,
            )
            
            # Print sampling path
            if print_lock:
                with print_lock:
                    print(f"  Path steps ({len(path.past_subqueries)} steps):")
                    for i, (sq, sa) in enumerate(zip(path.past_subqueries, path.past_subanswers)):
                        print(f"    Step {i+1}:")
                        print(f"      SubQuery: {sq}")
                        print(f"      SubAnswer: {sa}")
            else:
                print(f"  Path steps ({len(path.past_subqueries)} steps):")
                for i, (sq, sa) in enumerate(zip(path.past_subqueries, path.past_subanswers)):
                    print(f"    Step {i+1}:")
                    print(f"      SubQuery: {sq}")
                    print(f"      SubAnswer: {sa}")
            
            # Generate final answer based on the path
            # We construct a mock "path" context for final answer generation
            # CoRagAgent.generate_final_answer takes a RagPath
            
            final_ans = agent.generate_final_answer(
                corag_sample=path,
                task_desc=task_desc,
                max_message_length=args.max_message_length if args.max_message_length > 0 else None,
                temperature=0.0 # Deterministic final answer
            )
            
            # Check correctness
            if judge_client:
                is_correct = check_answer_with_llm_judge(final_ans, ground_truths, query, judge_client, print_lock)
            else:
                is_correct = check_answer(final_ans, ground_truths)
            
            if print_lock:
                with print_lock:
                    print(f"  Final Answer: {final_ans}")
                    print(f"  Correct: {'✓ YES' if is_correct else '✗ NO'} {'(LLM Judge)' if judge_client else '(String Match)'}")
            else:
                print(f"  Final Answer: {final_ans}")
                print(f"  Correct: {'✓ YES' if is_correct else '✗ NO'} {'(LLM Judge)' if judge_client else '(String Match)'}")
            
            if is_correct:
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
            if print_lock:
                with print_lock:
                    print(f"  ✗ Error: {e}")
            else:
                print(f"  ✗ Error: {e}")
            continue
    
    if print_lock:
        with print_lock:
            print(f"\n[Summary] Query '{query[:50]}...' generated {len(valid_paths)}/{args.n_samples} valid paths")
    else:
        print(f"\n[Summary] Query '{query[:50]}...' generated {len(valid_paths)}/{args.n_samples} valid paths")
    return valid_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="corag/multihopqa")
    parser.add_argument("--subset", type=str, default="2wikimultihopqa", help="HF dataset config name (e.g., 2wikimultihopqa/hotpotqa/musique/bamboogle)")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_file", type=str, default="data/rejection_sampled_data.jsonl")
    # vLLM OpenAI-compatible base URL. Prefer --vllm_api_base (e.g. http://host:8000/v1).
    # --vllm_url is kept for backward compatibility (e.g. http://host:8000).
    parser.add_argument("--vllm_api_base", type=str, default=None, help="vLLM OpenAI API base URL, e.g. http://localhost:8000/v1")
    parser.add_argument("--vllm_api_key", type=str, default="token-123", help="vLLM OpenAI API key (if required)")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000", help="(Deprecated) vLLM host URL without /v1, e.g. http://localhost:8000")
    # If not provided (or invalid), we will auto-detect the first available model from vLLM /v1/models.
    parser.add_argument("--model", type=str, default="", help="vLLM model id/name. If empty or invalid, auto-detect from /v1/models.") 
    parser.add_argument("--graph_api_url", type=str, default="http://localhost:8023/retrieve")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of paths to sample per example")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_path_length", type=int, default=3)
    parser.add_argument("--max_message_length", type=int, default=0, help="Max message length for truncation (0 or negative to disable truncation)")
    parser.add_argument("--max_examples", type=int, default=-1, help="Limit number of examples to process (for debugging)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information including prompts, retrieval queries, and contexts")
    parser.add_argument("--num_threads", type=int, default=1, help="Number of threads for parallel processing")
    parser.add_argument("--judge_model_url", type=str, default=None, help="Judge model API base URL (e.g., http://localhost:8001/v1). If provided and valid, uses LLM as judge instead of string matching.")
    parser.add_argument("--judge_model_api_key", type=str, default="token-123", help="Judge model API key")
    parser.add_argument("--judge_model_name", type=str, default=None, help="Judge model name. If not provided, auto-detects from API.")
    
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset {args.dataset} ({args.subset}), split={args.split} ...")
    ds = load_dataset(args.dataset, args.subset, split=args.split)
    
    if args.max_examples > 0:
        ds = ds.select(range(args.max_examples))

    # Init VLLM Client
    api_base = args.vllm_api_base
    if not api_base:
        # Backward-compatible: accept http://host:port and append /v1
        api_base = args.vllm_url.rstrip("/")
        if not api_base.endswith("/v1"):
            api_base = f"{api_base}/v1"

    # Resolve model id: if user didn't pass --model, or passed an invalid model name, fallback to autodetect.
    try:
        import requests

        headers = {"Authorization": f"Bearer {args.vllm_api_key}"} if args.vllm_api_key else {}
        resp = requests.get(f"{api_base.rstrip('/')}/models", headers=headers, timeout=10)
        resp.raise_for_status()
        models_payload = resp.json()
        available_models = [m.get("id") for m in (models_payload.get("data") or []) if isinstance(m, dict) and m.get("id")]
    except Exception:
        available_models = []

    model_id = (args.model or "").strip()
    if (not model_id) or (available_models and model_id not in available_models):
        try:
            detected = get_vllm_model_id(api_base=api_base, api_key=args.vllm_api_key)
            if available_models and model_id and model_id not in available_models:
                print(f"[warn] --model '{model_id}' not found in vLLM /models; falling back to autodetected model '{detected}'.")
            elif not model_id:
                print(f"[info] --model not provided; using autodetected model '{detected}'.")
            model_id = detected
        except Exception as e:
            if not model_id:
                raise RuntimeError(f"Failed to auto-detect vLLM model id from {api_base}: {e}")
            # If user provided a model but autodetect failed, keep the user-provided value.
            print(f"[warn] Failed to auto-detect vLLM model id from {api_base}: {e}. Proceeding with --model '{model_id}'.")

    vllm = VllmClient(model=model_id, api_base=api_base, api_key=args.vllm_api_key)
    
    # Init Agent
    # We pass empty list as corpus since we use graph_api
    # Tokenizer is optional: if not provided, uses simple char-based truncation
    dummy_corpus = [] 
    base_agent = CoRagAgent(vllm_client=vllm, corpus=dummy_corpus, graph_api_url=args.graph_api_url)
    # Wrap with verbose agent if verbose mode is enabled
    agent = VerboseCoRagAgent(base_agent, verbose=args.verbose) if args.verbose else base_agent
    
    # Init Judge Model (if provided)
    judge_client = None
    if args.judge_model_url:
        judge_api_base = args.judge_model_url.rstrip('/')
        if not judge_api_base.endswith('/v1'):
            judge_api_base = f"{judge_api_base}/v1"
        
        if check_judge_model_available(judge_api_base, args.judge_model_api_key):
            try:
                if args.judge_model_name:
                    judge_model_id = args.judge_model_name
                else:
                    judge_model_id = get_vllm_model_id(api_base=judge_api_base, api_key=args.judge_model_api_key)
                judge_client = VllmClient(model=judge_model_id, api_base=judge_api_base, api_key=args.judge_model_api_key)
                print(f"[INFO] Using LLM judge model: {judge_model_id} at {judge_api_base}")
            except Exception as e:
                print(f"[WARN] Failed to initialize judge model: {e}, falling back to string matching")
                judge_client = None
        else:
            print(f"[WARN] Judge model API at {judge_api_base} is not available, falling back to string matching")
            judge_client = None

    # Create print lock for thread-safe printing
    print_lock = threading.Lock() if args.num_threads > 1 else None
    
    # Convert dataset to list for indexing
    data_items = list(ds)
    total_items = len(data_items)
    
    print(f"Starting rejection sampling generation with {args.num_threads} thread(s)...")
    
    if args.num_threads == 1:
        # Sequential processing (original behavior)
        output_data = []
        for example in tqdm(data_items, desc="Processing"):
            valid = process_example(example, agent, args, print_lock, judge_client)
            if valid:
                output_data.extend(valid)
    else:
        # Multi-threaded processing
        results_map = {}
        
        def process_with_index(index: int, example: Dict) -> tuple:
            """Process example and return (index, result)."""
            try:
                valid = process_example(example, agent, args, print_lock, judge_client)
                return (index, valid)
            except Exception as e:
                if print_lock:
                    with print_lock:
                        print(f"Error processing example at index {index}: {e}")
                else:
                    print(f"Error processing example at index {index}: {e}")
                return (index, [])
        
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(process_with_index, i, example): i 
                for i, example in enumerate(data_items)
            }
            
            # Process completed tasks with progress bar
            for future in tqdm(concurrent.futures.as_completed(future_to_index), 
                             total=total_items, desc="Processing"):
                index, valid = future.result()
                if valid:
                    results_map[index] = valid
        
        # Reconstruct output_data in original order
        output_data = []
        for i in range(total_items):
            if i in results_map:
                output_data.extend(results_map[i])
    
    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Saved {len(output_data)} valid paths to {args.output_file}")

if __name__ == "__main__":
    main()

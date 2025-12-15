import sys
import os
# Disable tokenizer parallelism to avoid "Already borrowed" errors in multithreaded environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import time
import argparse
import copy
import logging
import threading
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add src to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(os.path.dirname(current_dir), 'src')
sys.path.insert(0, src_path)

from transformers import HfArgumentParser, AutoTokenizer, PreTrainedTokenizerFast
from datasets import Dataset

from config import Arguments
from logger_config import logger
from data_utils import load_corpus, format_documents_for_final_answer
from vllm_client import VllmClient, get_vllm_model_id
from agent import CoRagAgent, RagPath
from utils import AtomicCounter

# Suppress httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

def run_custom_eval(args: Arguments):
    # Initialize components
    logger.info("Initializing VLLM Client...")
    if args.vllm_model:
        model_id = args.vllm_model
    else:
        model_id = get_vllm_model_id(api_base=args.vllm_api_base, api_key=args.vllm_api_key)
    
    vllm_client: VllmClient = VllmClient(model=model_id, api_base=args.vllm_api_base, api_key=args.vllm_api_key)
    
    final_vllm_client: VllmClient = None
    if args.final_answer_model or args.final_answer_api_base:
        final_api_base = args.final_answer_api_base if args.final_answer_api_base else args.vllm_api_base
        final_api_key = args.final_answer_api_key if args.final_answer_api_key else args.vllm_api_key
        
        if args.final_answer_model:
            final_model_id = args.final_answer_model
        else:
             logger.info(f"Auto-detecting Final Answer Model from {final_api_base}...")
             final_model_id = get_vllm_model_id(api_base=final_api_base, api_key=final_api_key)

        logger.info(f"Initializing Final Answer VLLM Client ({final_model_id})...")
        final_vllm_client = VllmClient(model=final_model_id, api_base=final_api_base, api_key=final_api_key)
    
    logger.info("Loading Corpus...")
    corpus = None
    if args.corpus_file:
        corpus: Dataset = load_corpus(args.corpus_file)
    else:
        logger.info("No corpus file provided. Skipping corpus loading.")
    
    logger.info("Initializing Agent...")
    tokenizer_name = args.tokenizer_name if args.tokenizer_name else model_id
    try:
        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        logger.error(f"Failed to load tokenizer from '{tokenizer_name}'.")
        raise e
        
    corag_agent: CoRagAgent = CoRagAgent(
        vllm_client=vllm_client, 
        corpus=corpus, 
        graph_api_url=args.graph_api_url, 
        tokenizer=tokenizer,
        final_vllm_client=final_vllm_client
    )
    # Use the same lock as the agent to ensure thread safety for tokenizer access
    tokenizer_lock: threading.Lock = corag_agent.lock

    if args.max_path_length < 1:
        args.decode_strategy = 'greedy'

    # Load custom dataset
    logger.info(f"Loading custom dataset from {args.eval_file}...")
    with open(args.eval_file, 'r', encoding='utf-8') as f:
        data_items = json.load(f)

    if args.dry_run:
        logger.info("Dry run enabled: processing only first 2 items.")
        data_items = data_items[:2]

    processed_cnt = AtomicCounter()
    total_cnt = len(data_items)
    
    # Metrics for timing
    total_infer_time = 0.0
    
    results = []
    
    def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
        question = item.get('question', '')
        ground_truth = item.get('answer', '')
        # Fallback if keys are different
        if not question:
            # Try finding a key that looks like a question
            for k in item.keys():
                if 'question' in k.lower():
                    question = item[k]
                    break
        
        if not question:
            logger.warning(f"Skipping item without question: {item}")
            return None

        task_desc = "answer multi-hop questions" # Default
        
        start_time = time.time()
        
        # 1. Path Generation
        path: RagPath = None
        if args.decode_strategy == 'greedy' or args.max_path_length < 1:
            path = corag_agent.sample_path(
                query=question, task_desc=task_desc,
                max_path_length=args.max_path_length,
                temperature=0., max_tokens=64
            )
        elif args.decode_strategy == 'tree_search':
            path = corag_agent.tree_search(
                query=question, task_desc=task_desc,
                max_path_length=args.max_path_length,
                temperature=args.sample_temperature, max_tokens=64
            )
        elif args.decode_strategy == 'best_of_n':
            path = corag_agent.best_of_n(
                query=question, task_desc=task_desc,
                max_path_length=args.max_path_length,
                temperature=args.sample_temperature,
                n = args.best_n,
                max_tokens=64
            )
        
        path_gen_time = time.time() - start_time
        
        # 2. Document Formatting
        all_documents = []
        if path.past_documents:
            for docs in path.past_documents:
                all_documents.extend(docs)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_documents = [x for x in all_documents if not (x in seen or seen.add(x))]

        # Use lock when modifying Documents via tokenizer (it modifies internal state)
        with tokenizer_lock:
            documents = format_documents_for_final_answer(
                args=args,
                tokenizer=tokenizer,
                corpus=corpus,
                documents=unique_documents,
                lock=None # The lock is already held outside this call
            )

        # 3. Final Answer Generation
        prediction: str = corag_agent.generate_final_answer(
            corag_sample=path,
            task_desc=task_desc,
            documents=documents,
            max_message_length=args.max_len,
            temperature=0., max_tokens=128
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        final_gen_time = total_time - path_gen_time

        # Logging
        processed_cnt.increment()
        if processed_cnt.value % 10 == 0:
            logger.info(f"Processed {processed_cnt.value} / {total_cnt}")

        # Construct result dict
        # Custom metrics if available (Q2T, PPR not captured here)
        time_breakdown = [0.0, 0.0, path_gen_time, final_gen_time]

        result_item = {
            "question": question,
            "answer": prediction,
            "ground_truth": ground_truth,
            "reasoning_steps": [],
            "time": time_breakdown
        }

        # Add reasoning steps
        if path.past_subqueries:
            for sq, sa in zip(path.past_subqueries, path.past_subanswers):
                result_item["reasoning_steps"].append({
                    "subquery": sq,
                    "subanswer": sa
                })
        
        return result_item

    # Use ThreadPoolExecutor for parallel processing
    # Note: Adjust num_threads in args based on system capabilities
    logger.info(f"Processing {total_cnt} items with {args.num_threads} threads...")
    
    processed_results = []
    
    # Use a dictionary to store results by index to preserve order
    results_map = {}
    
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        # Submit all tasks and map future to index
        future_to_index = {executor.submit(process_item, item): i for i, item in enumerate(data_items)}
        
        # Use as_completed so tqdm updates as soon as ANY thread finishes
        for future in tqdm(as_completed(future_to_index), total=total_cnt, desc="Processing"):
            index = future_to_index[future]
            try:
                res = future.result()
                if res:
                    results_map[index] = res
            except Exception as e:
                logger.error(f"Error processing item at index {index}: {e}")
                import traceback
                traceback.print_exc()

    # Reconstruct results in original order
    processed_results = []
    for i in range(len(data_items)):
        if i in results_map:
            processed_results.append(results_map[i])

    # Calculate average stats
    total_q2t = 0
    total_ppr = 0
    total_reranker = 0
    total_llm_call = 0
    
    for res in processed_results:
        t = res["time"]
        total_q2t += t[0]
        total_ppr += t[1]
        total_reranker += t[2]
        total_llm_call += t[3]
    
    num_samples = len(processed_results)
    if num_samples > 0:
        avg_summary = {
            "type": "AVERAGE_TIME_SUMMARY",
            "avg_q2t_time": total_q2t / num_samples,
            "avg_ppr_time": total_ppr / num_samples,
            "avg_reranker_time": total_reranker / num_samples,
            "avg_llm_call_time": total_llm_call / num_samples,
            "total_samples": num_samples
        }
    else:
         avg_summary = {
            "type": "AVERAGE_TIME_SUMMARY",
            "avg_q2t_time": 0,
            "avg_ppr_time": 0,
            "avg_reranker_time": 0,
            "avg_llm_call_time": 0,
            "total_samples": 0
        }
    
    processed_results.insert(0, avg_summary)
    
    # Save results
    logger.info(f"Saving results to {args.save_file}...")
    # Ensure directory exists
    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
    
    with open(args.save_file, 'w', encoding='utf-8') as f:
        json.dump(processed_results, f, ensure_ascii=False, indent=4)
        
    logger.info("Done!")

if __name__ == "__main__":
    parser = HfArgumentParser((Arguments,))
    # Parse known args, allowing for extra args that might be handled differently
    args, unknown = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    if unknown:
        logger.warning(f"Unknown arguments: {unknown}")
    
    # Define script-specific arguments separate from the main configuration
    from dataclasses import dataclass, field
    
    @dataclass
    class ScriptArguments:
        eval_file: str = field(default=None, metadata={"help": "Path to input JSON file"})
        save_file: str = field(default=None, metadata={"help": "Path to output JSON file"})
        
    parser = HfArgumentParser((Arguments, ScriptArguments))
    args, script_args = parser.parse_args_into_dataclasses()
    
    # Merge script arguments into the main arguments object
    args.eval_file = script_args.eval_file
    args.save_file = script_args.save_file
    
    if not args.eval_file or not args.save_file:
        logger.error("Please provide --eval_file and --save_file")
        sys.exit(1)

    run_custom_eval(args)

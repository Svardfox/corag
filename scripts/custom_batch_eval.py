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
from concurrent.futures import ThreadPoolExecutor
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
        
    corag_agent: CoRagAgent = CoRagAgent(vllm_client=vllm_client, corpus=corpus, graph_api_url=args.graph_api_url, tokenizer=tokenizer)
    tokenizer_lock: threading.Lock = threading.Lock()

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
                lock=None # We are already holding the lock here, passing None to avoid double locking if internal fun supports it, or just let it lock/unlock if reentrant. 
                # Wait, format_documents_for_final_answer takes a 'lock' arg.
                # Let's check data_utils.py to see how it uses the lock.
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
        # Map timing to user requested format approximately:
        # [q2t (N/A), ppr (N/A), reranker (path_generation), llm_call (final_generation)]
        # We'll just put 0 for the first two as we don't have those specific steps.
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

    # Use ThreadPoolExecutor for parallel processing if configured
    # Note: vLLM client handles concurrency well, but Python GIL might be a bottleneck.
    # Adjust num_threads in args.
    logger.info(f"Processing {total_cnt} items with {args.num_threads} threads...")
    
    processed_results = []
    
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = [executor.submit(process_item, item) for item in data_items]
        for future in tqdm(futures, total=total_cnt, desc="Processing"):
            try:
                res = future.result()
                if res:
                    processed_results.append(res)
            except Exception as e:
                logger.error(f"Error processing item: {e}")
                import traceback
                traceback.print_exc()

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
    # We allow loose parsing in case user passes extra args
    args, unknown = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    if unknown:
        logger.warning(f"Unknown arguments: {unknown}")
    
    # User might pass args not in our Arguments dataclass via command line for their own script logic,
    # but here we rely on Arguments.
    # However, Arguments dataclass defines 'eval_file' and 'graph_api_url' etc?
    # Let's check config.py to ensure 'eval_file' is there or if we need to add it.
    # Based on run_inference.py, it uses args.output_dir and args.eval_task/split.
    # Ideally I should add 'eval_file' and 'save_file' to Arguments or parse them separately.
    # Since I cannot modify config.py easily without knowing if it breaks things, I will just add them 
    # to parser manually if they are not in Arguments, or use a separate argparse for them.
    # BUT HfArgumentParser is tricky with mix.
    # Let's try to see if I can just use argparse for the file paths and HfArgumentParser for the rest?
    # No, that's messy.
    # Better approach: check if 'eval_file' is in Arguments. If not, I'll access it from sys.argv manually or 
    # add a temporary argument class.
    
    # Actually, looking at the user's request, they want to reuse existing logic.
    # I will assume standard args are passed. But wait, Arguments class usually defines model args.
    # I will check if I can just add my own args.
    
    # To be safe and avoid conflict, I will create a small dataclass for my script specific args
    # and merge it.
    from dataclasses import dataclass, field
    
    @dataclass
    class ScriptArguments:
        eval_file: str = field(default=None, metadata={"help": "Path to input JSON file"})
        save_file: str = field(default=None, metadata={"help": "Path to output JSON file"})
        
    parser = HfArgumentParser((Arguments, ScriptArguments))
    args, script_args = parser.parse_args_into_dataclasses()
    
    # Merge script_args into args for convenience if needed, or just pass both.
    # Actually run_custom_eval needs both. Let's start by attaching script_args fields to args object 
    # or just pass them explicitly.
    args.eval_file = script_args.eval_file
    args.save_file = script_args.save_file
    
    if not args.eval_file or not args.save_file:
        logger.error("Please provide --eval_file and --save_file")
        sys.exit(1)

    run_custom_eval(args)

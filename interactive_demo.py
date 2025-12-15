import sys
import os
import torch
import threading
import logging
from typing import List, Dict

# Add src to sys.path
sys.path.insert(0, 'src')

from transformers import HfArgumentParser, AutoTokenizer, PreTrainedTokenizerFast
from datasets import Dataset

from config import Arguments
from logger_config import logger
from data_utils import load_corpus, format_documents_for_final_answer
from vllm_client import VllmClient, get_vllm_model_id
from agent import CoRagAgent, RagPath


logging.getLogger("httpx").setLevel(logging.WARNING)

def main():
    parser = HfArgumentParser((Arguments,))
    args: Arguments = parser.parse_args_into_dataclasses()[0]
    
    # Initialize components
    logger.info("Initializing VLLM Client...")
    if args.vllm_model:
        model_id = args.vllm_model
    else:
        model_id = get_vllm_model_id(api_base=args.vllm_api_base, api_key=args.vllm_api_key)
    
    vllm_client: VllmClient = VllmClient(model=model_id, api_base=args.vllm_api_base, api_key=args.vllm_api_key)
    
    final_vllm_client: VllmClient = None
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
    if args.corpus_file:
        corpus: Dataset = load_corpus(args.corpus_file)
    else:
        logger.info("No corpus file provided. Skipping corpus loading.")
        corpus = None
    
    logger.info("Initializing Agent...")
    tokenizer_name = args.tokenizer_name if args.tokenizer_name else model_id
    try:
        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        if args.tokenizer_name is None:
            logger.error(f"Failed to load tokenizer from '{tokenizer_name}'. If this is a remote path from vLLM, please specify a local tokenizer using --tokenizer_name <huggingface_repo_id>.")
            sys.exit(1)
        else:
            raise e
            raise e
    corag_agent: CoRagAgent = CoRagAgent(
        vllm_client=vllm_client, 
        corpus=corpus, 
        graph_api_url=args.graph_api_url, 
        tokenizer=tokenizer,
        final_vllm_client=final_vllm_client
    )
    
    tokenizer_lock: threading.Lock = threading.Lock()

    if args.max_path_length < 1:
        args.decode_strategy = 'greedy'

    print("\n" + "="*50)
    print("Interactive CoRAG Demo")
    print("Enter 'q' or 'quit' to exit.")
    print("="*50 + "\n")

    while True:
        try:
            query = input("\nEnter your query: ").strip()
            if query.lower() in ['q', 'quit', 'exit']:
                break
            if not query:
                continue

            task_desc = "answer multi-hop questions" # Default task description

            print(f"\nProcessing query: {query}...")
            
            # 1. Path Generation
            path: RagPath = None
            if args.decode_strategy == 'greedy' or args.max_path_length < 1:
                path = corag_agent.sample_path(
                    query=query, task_desc=task_desc,
                    max_path_length=args.max_path_length,
                    temperature=0., max_tokens=64
                )
            elif args.decode_strategy == 'tree_search':
                path = corag_agent.tree_search(
                    query=query, task_desc=task_desc,
                    max_path_length=args.max_path_length,
                    temperature=args.sample_temperature, max_tokens=64
                )
            elif args.decode_strategy == 'best_of_n':
                path = corag_agent.best_of_n(
                    query=query, task_desc=task_desc,
                    max_path_length=args.max_path_length,
                    temperature=args.sample_temperature,
                    n = args.best_n,
                    max_tokens=64
                )

            # Print intermediate steps
            if path.past_subqueries:
                print("\n--- Intermediate Steps ---")
                for i, (sq, sa) in enumerate(zip(path.past_subqueries, path.past_subanswers)):
                    print(f"Step {i+1}:")
                    if path.past_thoughts and i < len(path.past_thoughts) and path.past_thoughts[i]:
                        print(f"  Thought: {path.past_thoughts[i]}")
                    print(f"  Query: {sq}")
                    print(f"  Answer: {sa}")
                    if path.past_doc_ids and i < len(path.past_doc_ids):
                         print(f"  Docs: {path.past_doc_ids[i]}")

            # 2. Document Formatting
            documents: List[str] = format_documents_for_final_answer(
                args=args,
                context_doc_ids=path.past_doc_ids[-1] if path.past_doc_ids else [], 
                # In the CoRAG loop, we collect all unique document IDs encountered in the path
                # to serve as context for the final answer.
                tokenizer=tokenizer, corpus=corpus,
                lock=tokenizer_lock
            )
            
            # Flatten all doc IDs from the path for the final answer context.
            
            all_documents = []
            if path.past_documents:
                for docs in path.past_documents:
                    all_documents.extend(docs)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_documents = [x for x in all_documents if not (x in seen or seen.add(x))]

            documents = format_documents_for_final_answer(
                args=args,
                tokenizer=tokenizer,
                corpus=corpus,
                documents=unique_documents,
                lock=tokenizer_lock
            )

            # 3. Final Answer Generation
            prediction: str = corag_agent.generate_final_answer(
                corag_sample=path,
                task_desc=task_desc,
                documents=documents,
                max_message_length=args.max_len,
                temperature=0., max_tokens=128
            )

            print("\n--- Final Answer ---")
            print(prediction)
            print("-" * 20)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

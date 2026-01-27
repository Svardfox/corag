import sys
import os
import json
import re
import argparse
from typing import List, Dict, Any
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
try:
    from src.prompts import (
        get_generate_subquery_prompt,
        get_generate_intermediate_answer_prompt,
        get_generate_final_answer_prompt
    )
except ImportError:
    print("Error: Could not import src.prompts. Please ensure you are running from the project root.")
    sys.exit(1)

def extract_docs_from_observation(content: str) -> List[str]:
    """Extract raw docs list from 'Retrieved Context: ...' string."""
    if content.startswith("Retrieved Context:\n"):
        content = content[len("Retrieved Context:\n"):]
    
    # Check if docs are numbered (Doc 1: ...)
    doc_matches = list(re.finditer(r'(Doc \d+:)(.*?)(?=(Doc \d+:|$))', content, re.DOTALL))
    
    if doc_matches:
        return [m.group(2).strip() for m in doc_matches]
    
    # Fallback
    if content.strip():
        return [content.strip()]
    return []

def parse_chatml_to_steps(messages: List[Dict]) -> Dict[str, Any]:
    """Parse a flat ChatML message list into a structured item with steps."""
    if not messages:
        return None
        
    query = ""
    steps = []
    final_answer = ""
    
    current_subquery = None
    current_docs = []
    
    # 1. Get Main Query
    if messages[0]['role'] == 'user':
        query = messages[0]['content']
    
    i = 1
    while i < len(messages):
        msg = messages[i]
        role = msg['role']
        content = msg['content']
        
        if role == 'assistant':
            if content.startswith("SubQuery:"):
                current_subquery = content[len("SubQuery:"):].strip()
                current_docs = [] # Reset docs
            elif content.startswith("SubAnswer:"):
                subanswer = content[len("SubAnswer:"):].strip()
                if current_subquery:
                    steps.append({
                        "subquery": current_subquery,
                        "documents": current_docs,
                        "subanswer": subanswer
                    })
                    current_subquery = None
                    current_docs = []
            elif content.startswith("Final Answer:"):
                final_answer = content[len("Final Answer:"):].strip()
                
        elif role == 'observation' or (role == 'system' and "Retrieved Context" in content):
            docs = extract_docs_from_observation(content)
            if docs:
                current_docs = docs
        
        i += 1
        
    return {
        "query": query,
        "steps": steps,
        "answer": final_answer
    }

def process_item(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a ChatML item into multiple aligned training samples."""
    # 1. Parse ChatML to structured steps
    parsed = parse_chatml_to_steps(item.get('messages', []))
    if not parsed or not parsed['query']:
        return []

    aligned_samples = []
    query = parsed['query']
    history_sq = []
    history_sa = []
    
    task_desc = "answer multi-hop questions" # Default for old code logic

    # 2. Iterate steps
    for step in parsed['steps']:
        sq = step['subquery']
        sa = step['subanswer']
        docs = step['documents']
        
        if not sq or not sa: continue

        # === Sample A: SubQuery ===
        # Input: Main Query + History
        prompt_sq = get_generate_subquery_prompt(query, history_sq, history_sa, task_desc)
        aligned_samples.append({
            "type": "subquery_generation",
            "messages": prompt_sq + [{"role": "assistant", "content": f"SubQuery: {sq}"}]
        })
        
        # === Sample B: SubAnswer ===
        # Input: Current SubQuery + Docs (Independent!)
        prompt_sa = get_generate_intermediate_answer_prompt(sq, docs)
        aligned_samples.append({
            "type": "subanswer_generation",
            "messages": prompt_sa + [{"role": "assistant", "content": f"SubAnswer: {sa}"}]
        })
        
        # Update History
        history_sq.append(sq)
        history_sa.append(sa)

    # 3. Final Answer
    if parsed['answer']:
        prompt_final = get_generate_final_answer_prompt(query, history_sq, history_sa, task_desc)
        aligned_samples.append({
            "type": "final_answer_generation",
            "messages": prompt_final + [{"role": "assistant", "content": f"Final Answer: {parsed['answer']}"}]
        })
        
    return aligned_samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    print(f"Reading {args.input_file}...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    print(f"Processing {len(data)} examples...")
    total_aligned = 0
    counts = {"subquery": 0, "subanswer": 0, "final": 0}
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in tqdm(data):
            samples = process_item(item)
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
                if s['type'] == 'subquery_generation': counts['subquery'] += 1
                elif s['type'] == 'subanswer_generation': counts['subanswer'] += 1
                elif s['type'] == 'final_answer_generation': counts['final'] += 1
            total_aligned += len(samples)
            
    print(f"Done! Generated {total_aligned} aligned samples.")
    print(f"Stats: {counts}")

if __name__ == "__main__":
    main()

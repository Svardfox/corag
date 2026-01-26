from typing import List, Dict, Optional

def get_generate_subquery_prompt(query: str, past_subqueries: List[str], past_subanswers: List[str], task_desc: str) -> List[Dict]:
    assert len(past_subqueries) == len(past_subanswers)
    
    # 统一使用 SubQuery / SubAnswer 术语
    past = ''
    for idx in range(len(past_subqueries)):
        past += f"SubQuery: {past_subqueries[idx]}\nSubAnswer: {past_subanswers[idx]}\n"
    past = past.strip()

    prompt = f"""You are using a search engine to answer a complex question. Decompose the question into simple sub-queries.
    
## Previous Steps
{past or 'None'}

## Main Question
{query}

Respond with the next sub-query starting with 'SubQuery: '. Do not explain yourself."""

    return [
        {'role': 'user', 'content': prompt}
    ]

def get_generate_intermediate_answer_prompt(subquery: str, documents: List[str]) -> List[Dict]:
    """
    构造生成 SubAnswer 的 prompt，与训练数据格式完全一致（Old Code 逻辑）。
    
    单轮 User 消息格式，包含 "Given the following documents..." 指令。
    """
    context = ''
    for idx, doc in enumerate(documents):
        context += f"Doc {idx+1}: {doc}\n\n"

    prompt = f"""Given the following documents retrieved for the sub-query, provide a concise answer.

SubQuery: {subquery}

Retrieved Context:
{context.strip()}

Answer the sub-query concisely and start with 'SubAnswer: '."""

    return [
        {'role': 'user', 'content': prompt}
    ]

def get_generate_final_answer_prompt(
        query: str, past_subqueries: List[str], past_subanswers: List[str], task_desc: str,
        documents: Optional[List[str]] = None
) -> List[Dict]:

    assert len(past_subqueries) == len(past_subanswers)
    past = ''
    for idx in range(len(past_subqueries)):
        past += f"SubQuery: {past_subqueries[idx]}\nSubAnswer: {past_subanswers[idx]}\n"
    
    context = ''
    if documents:
        for idx, doc in enumerate(documents):
            context += f"Doc {idx+1}: {doc}\n\n"

    prompt = f"""Given the following intermediate queries and their answers, provide the final answer to the main question.

## Search History
{past.strip()}

## Final Context
{context.strip() or 'No additional context.'}

## Main Question
{query}

Respond starting with 'Final Answer: '."""

    return [
        {'role': 'user', 'content': prompt}
    ]

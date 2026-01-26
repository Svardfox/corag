from typing import List, Dict, Optional

# 统一的系统提示词，与 train.py 保持高度一致
SYSTEM_PROMPT = (
    "You are a helpful assistant that can use search tools to solve complex multi-step questions. "
    "When you receive a question, you should decompose it into several simple sub-queries. "
    "After receiving the retrieved context for each sub-query, provide a sub-answer. "
    "Finally, give the final answer based on all information. "
    "Follow the format strictly: SubQuery, SubAnswer, and Final Answer. /no_think"
)

def get_generate_subquery_prompt(query: str, past_subqueries: List[str], past_subanswers: List[str], task_desc: str) -> List[Dict]:
    assert len(past_subqueries) == len(past_subanswers)
    
    # 统一使用 SubQuery / SubAnswer 术语
    past = ''
    for idx in range(len(past_subqueries)):
        past += f"SubQuery: {past_subqueries[idx]}\nSubAnswer: {past_subanswers[idx]}\n"
    past = past.strip()

    prompt = f"""You are decomposing a complex question into simple sub-queries.
    
## Previous Steps
{past or 'None'}

## Main Question
{query}

Respond with the next sub-query starting with 'SubQuery: '. Do not explain yourself."""

    return [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': prompt}
    ]

def get_generate_intermediate_answer_prompt(subquery: str, documents: List[str]) -> List[Dict]:
    context = ''
    for idx, doc in enumerate(documents):
        context += f"Doc {idx+1}: {doc}\n\n"

    # 模拟训练时的 Retrieved Context 格式
    return [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': f"Retrieved Context:\n{context.strip()}"},
        {'role': 'user', 'content': f"Based on the context above, provide a concise sub-answer for: {subquery}\nRespond starting with 'SubAnswer: '."}
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

    prompt = f"""Combine all information to give the final answer.

## Search History
{past.strip()}

## Final Context
{context.strip() or 'No additional context.'}

## Main Question
{query}

Respond starting with 'Final Answer: '."""

    return [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': prompt}
    ]

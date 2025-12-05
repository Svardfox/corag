from typing import List, Dict, Union
from concurrent.futures import ThreadPoolExecutor
import requests

from utils import AtomicCounter


def get_vllm_model_id(host: str = "localhost", port: int = 8000, api_key: str = "token-123", api_base: str = None) -> str:
    if api_base:
        base_url = api_base
    else:
        base_url = f"http://{host}:{port}/v1"

    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    try:
        response = requests.get(f"{base_url}/models", headers=headers)
        response.raise_for_status()
        models = response.json()
        return models['data'][0]['id']
    except Exception as e:
        # Fallback or re-raise with clear message
        raise RuntimeError(f"Failed to fetch models from vLLM: {e}")


class VllmClient:

    def __init__(self, model: str, host: str = 'localhost', port: int = 8000, api_key: str = 'token-123', api_base: str = None):
        super().__init__()
        self.model = model
        if api_base:
            self.base_url = api_base
        else:
            self.base_url = f"http://{host}:{port}/v1"
            
        self.api_key = api_key
        self.token_consumed: AtomicCounter = AtomicCounter()

    def call_chat(self, messages: List[Dict], return_str: bool = True, **kwargs) -> Union[str, Dict]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": messages,
            **kwargs
        }
        
        response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        completion = response.json()
        
        if 'usage' in completion:
            self.token_consumed.increment(num=completion['usage']['total_tokens'])

        return completion['choices'][0]['message']['content'] if return_str else completion

    def batch_call_chat(self, messages: List[List[Dict]], return_str: bool = True, num_workers: int = 4, **kwargs) -> List[Union[str, Dict]]:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            return list(executor.map(
                lambda m: self.call_chat(m, return_str=return_str, **kwargs),
                messages
            ))

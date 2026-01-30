import requests

from typing import List, Dict

from logger_config import logger


def _normalize_graph_api_results(results):
    """Normalize Graph API responses into a list for downstream consumers."""
    if isinstance(results, dict):
        for key in ['chunks', 'data', 'results', 'docs', 'passages']:
            value = results.get(key)
            if isinstance(value, list):
                return value
        return [results] if results else []
    if isinstance(results, list):
        return results
    return []


def search_by_http(query: str, host: str = 'localhost', port: int = 8090) -> List[Dict]:
    url = f"http://{host}:{port}"
    response = requests.post(url, json={'query': query})

    if response.status_code == 200:
        return response.json()
    else:
        logger.error(f"Failed to get a response. Status code: {response.status_code}")
        return []


def search_by_graph_api(query: str, url: str) -> List[Dict]:
    try:
        response = requests.post(url, json={'query': query}, headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            return _normalize_graph_api_results(response.json())
        else:
            logger.error(f"Failed to get a response from graph API. Status code: {response.status_code}")
            return []
    except requests.RequestException as e:
        logger.error(f"Error calling graph API: {e}")
        return []

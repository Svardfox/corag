# currently only preloading index is supported, it is expected to allow load and manage multiple gc sources

import gc
import argparse
import uvicorn
import json
import logging
import sys
import os

from fastapi import FastAPI
from pydantic import BaseModel

# Add the project root to sys.path to allow imports from src and other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from utils import str2bool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Assuming these modules are available in the environment or will be added
try:
    from kgc.rag.graphrag_simireranker import GraphRAGModel
    from kgc.scheme_embedding import EMBWorker
    from kgc.vdb.faiss_vdb import FaissVector
    from kgc.gdb.networkx_gdb import NetworkxGraph
except ImportError:
    # Fallback for development/checking if modules are missing
    logging.warning("KGC modules not found. Ensure they are in the python path.")
    pass


class RetrieveInput(BaseModel):
    query: str
    # depth: int
    # max_passages: int


class RAG_Service():
    def __init__(self):
        self.initialized = False
        self.RAG = None
        self.emb = None
    
    def load_index(self, args):
        if not self.initialized:
            self.args = args
            # devices=["npu:1"] is hardcoded as per original snippet
            self.emb = EMBWorker(args, devices=["npu:1"])

            g0 = NetworkxGraph(path_or_name=args.gdb_path)
            v0 = FaissVector(args.embedding_dim, args.vdb_path, type="IndexFlatIP")
            v1 = FaissVector(args.embedding_dim, args.vdb_triples_path, type="IndexFlatIP")
            v2 = FaissVector(args.embedding_dim, args.vdb_passages_path, type="IndexFlatIP")
            self.RAG = GraphRAGModel(args, None, self.emb, g0, v0, v1, v2)
            self.initialized = True

    def clear_index(self):
        del self.RAG
        self.initialized = False
        gc.collect()

    def retrieve(self, query, max_passages=10, depth=2):
        if not self.initialized:
            logger.error("RAG not initialized")
            return []

        self.RAG.subgraph = NetworkxGraph(is_digraph=True)
        top_k_triples, top_k_triple_elements, distance = self.RAG.query_to_triples(query, top_k=10)
        
        logging.info("Query: %s", query)
        # logging.info("Shot Triples: %s", top_k_triples)
        
        retrieved_passages, _ = self.RAG.get_passages_by_ppr(
            query, top_k_triples, top_k_triple_elements, distance, depth
        )
        self.RAG.reset_subgraph()

        return retrieved_passages[:max_passages]


# Global RAG service instance
app = FastAPI()
rag = RAG_Service()
args = None


def getargs():
    global args
    parser = argparse.ArgumentParser()
    
    # Model Args
    parser.add_argument("-m", "--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model_path", type=str, default="/data0/xyao/models/Qwen2.5-7B-Instruct")
    # ... (Keep existing complex URL defaults as they might be specific to user environment)
    parser.add_argument("--model_url", type=str, default='http://10.170.30.9/v1/infers/f13f111a-7b8f-4f35-89ae-0fd3f49b1304/v1/chat/completions?endpoint=infer-app-modelarts-cn-southwest-2')
    parser.add_argument("--model_name", type=str, default="qwen2.5-72b")
    parser.add_argument("--server_type", type=str, default="vllm-ascend")
    parser.add_argument('--headers', type=str, 
                        default='{"Content-Type": "application/json", "X-Apig-AppCode":"7edfd51d270b49aea844be9d4ff5d2fa266ebf3fc33f41b1ad2398279d2e8595"}', 
                        help='Headers in JSON format')

    # Online flags
    parser.add_argument("--model_online", type=str2bool, default=True)
    parser.add_argument("--llmworker_online", type=str2bool, default=True)
    parser.add_argument("--preload_index", type=str2bool, default=True)
    parser.add_argument("--emb_model_online", type=str2bool, default=False)

    # Embedding / Path Args
    parser.add_argument("--emb_model_path", type=str, default="/data/xyao/models/bge-large-en-v1.5/")
    
    # Dataset Paths
    parser.add_argument("--gdb_path", type=str, default="dataset/musique.pkl")
    parser.add_argument("--vdb_type", type=str, default="faiss")
    parser.add_argument("--vdb_path", type=str, default="data_ksc/merged_graph/vdb_musique.faiss")
    parser.add_argument("--vdb_triples_path", type=str, default="data_ksc/merged_graph/vdb_musique_triples.faiss")
    parser.add_argument("--vdb_passages_path", type=str, default="data_ksc/merged_graph/vdb_musique_passages.faiss")
    parser.add_argument("--image_path", type=str, default="./dataset/vdb/image_result")

    # RAG Parameters
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--metric", type=str, choices=["choice", "generation"], default="generation")
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--chunk_overlap", type=int, default=100)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--subgraph_depth", type=int, default=2)
    parser.add_argument("--text", type=bool, default=True)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--embedding_batch_size", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=1024)
    
    args = parser.parse_args()


@app.on_event("startup")
async def startup():
    getargs()
    if args.preload_index:
        logger.info("Preloading index...")
        rag.load_index(args)


@app.get("/")
async def root():
    return {"status": f"{'ok' if rag.initialized else 'index uninitalized'}"}


@app.post("/retrieve")
async def retrieve(input: RetrieveInput):
    logger.info(f"Retrieve Query: {input.query}")
    try:
        response = rag.retrieve(input.query)
        # response is expected to be a list of strings (passages)
        # logger.info(f"Context: {json.dumps(response, ensure_ascii=False)}")
        return response
    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        return []


if __name__ == '__main__':
    # Parse uvicorn specific args first or separately to avoid conflict
    # But since the script uses argparse for global config, we might want to separate them.
    # For now, we reuse the argparse structure but typical usage is `python gc_service.py`
    
    # We'll use a separate parser for the main block to avoid required args issues from getargs()
    # OR we can just hardcode the run call if this is always dev env.
    
    # Simplified main launch
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8025)
    
    # We use parse_known_args to allow other args to flow through if needed, 
    # though getargs() is called inside startup() which parses sys.argv again.
    uv_args, _ = parser.parse_known_args()
    
    uvicorn.run("src.search.gc_service:app", host=uv_args.host, port=uv_args.port, reload=False)
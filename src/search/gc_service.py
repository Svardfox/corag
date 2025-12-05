#currently only preloading index is supported, it is expected to allow load and manage multiple gc sources

import gc
import argparse
import uvicorn
import json 
import logging

import sys
import os

# Add the project root to sys.path to allow imports from src and other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from fastapi import FastAPI
from pydantic import BaseModel

# Assuming these modules are available in the environment or will be added
try:
    from kgc.rag.graphrag_simireranker import GraphRAGModel
    from kgc.scheme_embedding import EMBWorker
    from kgc.vdb.faiss_vdb import FaissVector
    from kgc.gdb.networkx_gdb import NetworkxGraph
    from utils import str2bool
except ImportError:
    # Fallback for development/checking if modules are missing
    logging.warning("KGC modules not found. Ensure they are in the python path.")
    # Mocking for syntax check if needed, but better to let it fail if run
    pass

logger = logging.getLogger(__name__)


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
            self.emb = EMBWorker(args, devices=["npu:1"])

            g0 = NetworkxGraph(path_or_name=args.gdb_path)
            v0 = FaissVector(args.embedding_dim, args.vdb_path, type="IndexFlatIP")
            v1 = FaissVector(args.embedding_dim, args.vdb_triples_path, type="IndexFlatIP") # will be created if path not existed
            v2 = FaissVector(args.embedding_dim, args.vdb_passages_path, type="IndexFlatIP")
            self.RAG = GraphRAGModel(args, None, self.emb, g0, v0, v1, v2)
            self.initialized = True


    def clear_index(self):
        del self.RAG
        self.initialized = False
        gc.collect()

    def retrieve(self, query, max_passages = 10, depth = 2):
        self.RAG.subgraph = NetworkxGraph(is_digraph=True)
        top_k_triples,top_k_triple_elements, distance = self.RAG.query_to_triples(query, top_k=10)
        logging.info("Query: %s", query)
        logging.info("Shot Triples: %s", top_k_triples)
        retrieved_passages, _ = self.RAG.get_passages_by_ppr(query, top_k_triples, top_k_triple_elements, distance, depth)
        self.RAG.reset_subgraph()


        return retrieved_passages[:max_passages]

def getargs():
    global args
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model_path", type=str, default="/data0/xyao/models/Qwen2.5-7B-Instruct")

    parser.add_argument("--model_url", type=str, default='http://10.170.30.9/v1/infers/f13f111a-7b8f-4f35-89ae-0fd3f49b1304/v1/chat/completions?endpoint=infer-app-modelarts-cn-southwest-2')
    parser.add_argument("--model_name", type=str, default="qwen2.5-72b")
    parser.add_argument("--server_type", type=str, default="vllm-ascend")  # mindie or vllm-ascend [vllm format, openai format]
    parser.add_argument('--headers', type=str, 
                        default='{"Content-Type": "application/json", "X-Apig-AppCode":"7edfd51d270b49aea844be9d4ff5d2fa266ebf3fc33f41b1ad2398279d2e8595"}', 
                        help='Headers in JSON format')
    try:
        from utils import str2bool
        parser.add_argument("--model_online", type=str2bool, default=True)
        parser.add_argument("--llmworker_online", type=str2bool, default=True)
        parser.add_argument("--preload_index", type=str2bool, default=True)
    except ImportError:
         parser.add_argument("--model_online", type=bool, default=True)
         parser.add_argument("--llmworker_online", type=bool, default=True)
         parser.add_argument("--preload_index", type=bool, default=True)


    parser.add_argument("--emb_model_path", type=str, default="/data/xyao/models/bge-large-en-v1.5/")

    parser.add_argument("--gdb_path", type=str, default=f"../GC-main-branch/dataset/2wiki.pkl")
    parser.add_argument("--vdb_path", type=str, default=f"../GC-main-branch/data_ksc/merged_graph/vdb_2wiki_bge.faiss")
    parser.add_argument("--vdb_triples_path", type=str, default=f"../GC-main-branch/data_ksc/merged_graph/vdb_2wiki_triples_bge.faiss")
    parser.add_argument("--vdb_passages_path", type=str, default=f"../GC-main-branch/data_ksc/merged_graph/vdb_2wiki_passages_bge.faiss")
    parser.add_argument("--image_path", type=str, default=f"./dataset/vdb/image_result")


    # parser.add_argument("--gdb_path", type=str, default=f"/data0/j84/dbs/serhii/data_ksc/merged_graph/tmp_20250724_10.pkl")
    # parser.add_argument("--vdb_path", type=str, default=f"/data0/j84/dbs/serhii/data_ksc/merged_graph/vdb_tmp_20250724_10.faiss")
    # parser.add_argument("--vdb_triples_path", type=str, default=f"/data0/j84/dbs/serhii/data_ksc/merged_graph/vdb_tmp_20250724_10_triples.faiss")
    # parser.add_argument("--vdb_passages_path", type=str, default=f"/data0/j84/dbs/serhii/data_ksc/merged_graph/vdb_tmp_20250724_10_passages.faiss")
    # parser.add_argument("--image_path", type=str, default=f"/data0/j84/dbs/serhii/data_ksc/image_result")


    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--metric", type=str, choices=["choice", "generation"], default="generation")

    # rag parameters
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
    
app = FastAPI()
rag = RAG_Service()
args = None

@app.on_event("startup")
async def startup():
    getargs()
    if args.preload_index:
        rag.load_index(args)

@app.get("/")
async def root():
    return {"status": f"{'ok' if rag.initialized else 'index uninitalized'}"}

@app.post("/retrieve")
async def retrieve(input: RetrieveInput):
    logger.info(f"Retrieve: {input.query}")
    response = rag.retrieve(input.query)
    # Modified to return the list directly, matching CoRagAgent expectations
    return response


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default = 8023)
    uv_args = parser.parse_args()
    uvicorn.run("src.search.gc_service:app", host=uv_args.host, port=uv_args.port, reload=False)

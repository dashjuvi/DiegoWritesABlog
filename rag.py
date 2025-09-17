from __future__ import annotations
import os
import logging
from typing import List, Tuple
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import Settings
import chromadb

from config import CONFIG
from classifier import classify_title

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TOP_K = CONFIG["DEFAULT_TOP_K"]

def _get_embed_model(category: str) -> HuggingFaceEmbedding:
    model_name = CONFIG["EMBED_TECH_MODEL"] if category == "tech" else CONFIG["EMBED_GEO_MODEL"]
    logger.info(f"Using embedding model {model_name} for category {category}")
    return HuggingFaceEmbedding(model_name=model_name, cache_folder="/workspace/data/.cache/huggingface")

def _get_vector_store(category: str) -> ChromaVectorStore:
    client = chromadb.PersistentClient(path=CONFIG["CHROMA_PATH"])
    collection = client.get_or_create_collection(name=f"articles_{category}")
    return ChromaVectorStore(chroma_collection=collection)

def ensure_index(category: str) -> VectorStoreIndex:
    vs = _get_vector_store(category)
    return VectorStoreIndex.from_vector_store(vector_store=vs)

def retrieve(query: str, hint_title: str = "", top_k: int = DEFAULT_TOP_K) -> Tuple[List[Tuple[str, dict]], str]:
    try:
        category = classify_title(hint_title or query)
        logger.info(f"Classified query as {category}")
        
        embed_model = _get_embed_model(category)
        Settings.embed_model = embed_model
        
        index = ensure_index(category)
        retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k * 2)  # Get extra for reranking
        nodes = retriever.retrieve(query)
        
        # Apply reranking
        reranker = SentenceTransformerRerank(model=CONFIG["RERANK_MODEL"], top_n=top_k)
        nodes = reranker.postprocess_nodes(nodes, query_str=query)
        
        contexts: List[Tuple[str, dict]] = []
        for n in nodes:
            meta = n.metadata or {}
            contexts.append((n.get_content(metadata_mode="none"), meta))
        
        logger.info(f"Retrieved {len(contexts)} contexts for query")
        return contexts, category
    except Exception as e:
        logger.error(f"Error in retrieval: {e}")
        return [], category

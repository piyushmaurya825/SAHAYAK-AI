"""
ChromaDB wrapper with Ollama nomic-embed-text embeddings.
Provides a unified interface for all domain-specific retrievals.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from langchain_ollama import OllamaEmbeddings

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CHROMA_DIR, CHROMA_TOP_K, EMBEDDING_MODEL, OLLAMA_BASE_URL,
    CHROMA_COLLECTION_SPIRITUAL, CHROMA_COLLECTION_CBT,
    CHROMA_COLLECTION_CULTURAL, CHROMA_COLLECTION_PSYCHOEDU,
    CHROMA_COLLECTION_GUIDED
)


def _get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)


def _get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False)
    )


class DomainRetriever:
    """Thin wrapper around a ChromaDB collection for one knowledge domain."""

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self._client = _get_chroma_client()
        self._embeddings = _get_embeddings()
        try:
            self._collection = self._client.get_collection(collection_name)
        except Exception:
            self._collection = None

    @property
    def is_ready(self) -> bool:
        return self._collection is not None and self._collection.count() > 0

    def query(
        self,
        query_text: str,
        n_results: int = CHROMA_TOP_K,
        where: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Return top-n semantically similar documents with metadata."""
        if not self.is_ready:
            return []
        try:
            query_embedding = self._embeddings.embed_query(query_text)
            kwargs: Dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": min(n_results, self._collection.count()),
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                kwargs["where"] = where
            results = self._collection.query(**kwargs)
            chunks = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                relevance = max(0.0, 1.0 - dist)  # cosine distance -> similarity
                chunks.append({"text": doc, "metadata": meta, "relevance": relevance})
            return sorted(chunks, key=lambda x: x["relevance"], reverse=True)
        except Exception as e:
            print(f"[VectorStore] Query error in {self.collection_name}: {e}")
            return []


class VectorStore:
    """Unified access point for all ChromaDB collections."""

    def __init__(self):
        self.spiritual = DomainRetriever(CHROMA_COLLECTION_SPIRITUAL)
        self.cbt = DomainRetriever(CHROMA_COLLECTION_CBT)
        self.cultural = DomainRetriever(CHROMA_COLLECTION_CULTURAL)
        self.psychoedu = DomainRetriever(CHROMA_COLLECTION_PSYCHOEDU)
        self.guided = DomainRetriever(CHROMA_COLLECTION_GUIDED)

    def is_fully_ready(self) -> bool:
        return all([
            self.spiritual.is_ready,
            self.cbt.is_ready,
            self.cultural.is_ready,
        ])

    def query_spiritual(self, query: str, n: int = CHROMA_TOP_K,
                        culture_filter: Optional[str] = None) -> List[Dict]:
        where = None
        if culture_filter and culture_filter.lower() not in ["universal", "unknown"]:
            # ChromaDB $or filter for culture tag
            where = {"$or": [
                {"culture_tag": culture_filter.lower()},
                {"culture_tag": "universal"}
            ]}
        return self.spiritual.query(query, n, where)

    def query_cbt(self, query: str, n: int = CHROMA_TOP_K) -> List[Dict]:
        return self.cbt.query(query, n)

    def query_cultural(self, query: str, n: int = CHROMA_TOP_K,
                       culture_filter: Optional[str] = None) -> List[Dict]:
        where = None
        if culture_filter and culture_filter.lower() not in ["universal", "unknown"]:
            where = {"$or": [
                {"culture": culture_filter},
                {"culture": "Universal"}
            ]}
        return self.cultural.query(query, n, where)

    def query_psychoedu(self, query: str, n: int = CHROMA_TOP_K) -> List[Dict]:
        return self.psychoedu.query(query, n)

    def query_guided(self, query: str, n: int = 2) -> List[Dict]:
        return self.guided.query(query, n)


_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store

"""
Product RAG (Retrieval-Augmented Generation) Service

TF-IDF based semantic search over product documentation.
"""

import re
import asyncio
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import RagConfig


@dataclass
class ProductChunk:
    """A chunk of product documentation."""
    product_name: str
    content: str


@dataclass
class SearchResult:
    """A search result with relevance score."""
    product_name: str
    content: str
    score: float


class ProductRagService:
    """
    RAG service for product documentation using TF-IDF.
    
    Lightweight alternative to neural embeddings - no PyTorch needed.
    Index is built at startup and kept in memory.
    """
    
    def __init__(self, config: RagConfig, base_path: Path = None):
        """
        Initialize the RAG service.
        
        Args:
            config: RAG configuration
            base_path: Base path for resolving document paths
        """
        self._config = config
        self._base_path = base_path or Path(__file__).parent.parent
        self._docs_path = self._base_path / config.documents_path
        
        self._chunks: List[ProductChunk] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
        
        # Product summary for system prompt enrichment
        self._product_summary = ""
    
    @property
    def product_summary(self) -> str:
        """Get product summary for system prompt."""
        return self._product_summary
    
    @property
    def chunk_count(self) -> int:
        """Get number of indexed chunks."""
        return len(self._chunks)
    
    async def initialize(self) -> None:
        """
        Initialize the RAG system (lazy, thread-safe).
        
        Builds the TF-IDF index from product documents.
        """
        if self._initialized:
            return
        
        async with self._init_lock:
            if self._initialized:
                return
            
            # Run CPU-intensive work in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._build_index)
            self._initialized = True
    
    def _build_index(self) -> None:
        """Build TF-IDF index from documents (runs in thread pool)."""
        print(f"[RAG] Building TF-IDF index from {self._docs_path}...")
        
        self._load_and_chunk()
        
        if not self._chunks:
            print("[RAG] No chunks found!")
            return
        
        texts = [chunk.content for chunk in self._chunks]
        
        self._vectorizer = TfidfVectorizer(
            ngram_range=self._config.ngram_range,
            stop_words="english",
            max_features=self._config.max_features,
            sublinear_tf=self._config.sublinear_tf,
        )
        
        self._tfidf_matrix = self._vectorizer.fit_transform(texts)
        
        print(f"[RAG] Index built: {len(self._chunks)} chunks, {self._tfidf_matrix.shape[1]} features")
    
    def _load_and_chunk(self) -> None:
        """Load markdown and split into product-based chunks."""
        if not self._docs_path.exists():
            print(f"[RAG] Document not found: {self._docs_path}")
            return
        
        content = self._docs_path.read_text(encoding="utf-8")
        
        # Split by product sections (### headers)
        sections = re.split(r"\n(?=###\s+(?:\d+\.\d+\s+)?Swiss Life)", content)
        
        product_summaries = []
        
        for section in sections:
            section = section.strip()
            if not section or len(section) < 50:
                continue
            
            # Extract product name from header
            header_match = re.match(r"###\s+(?:\d+\.\d+\s+)?(Swiss Life[^\n]+)", section)
            if header_match:
                product_name = header_match.group(1).strip()
            else:
                name_match = re.search(r"(Swiss Life [A-Za-z\s]+(?:Uno|Duo|Plan|Expert|Start)?)", section)
                product_name = name_match.group(1) if name_match else "Swiss Life Product"
            
            self._chunks.append(ProductChunk(
                product_name=product_name,
                content=section,
            ))
            
            # Extract short description for summary
            desc_match = re.search(r"\*\*Key Features:\*\*\s*\n-\s*([^\n]+)", section)
            if desc_match:
                short_desc = desc_match.group(1).strip()[:60]
            else:
                lines = [l.strip() for l in section.split("\n") if l.strip() and not l.startswith("#")]
                short_desc = lines[0][:60] if lines else "Insurance product"
            
            product_summaries.append(f"- {product_name}: {short_desc}")
        
        self._product_summary = "\n".join(product_summaries)
        print(f"[RAG] Loaded {len(self._chunks)} product chunks")
    
    async def search(self, query: str, top_k: int = None) -> List[SearchResult]:
        """
        Search for relevant product information.
        
        Args:
            query: Search query
            top_k: Number of results to return (default from config)
            
        Returns:
            List of search results sorted by relevance
        """
        await self.initialize()
        
        if self._vectorizer is None or self._tfidf_matrix is None or not self._chunks:
            return []
        
        top_k = top_k or self._config.top_k
        
        # Run search in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._search_sync(query, top_k),
        )
    
    def _search_sync(self, query: str, top_k: int) -> List[SearchResult]:
        """Synchronous search implementation (runs in thread pool)."""
        query_vector = self._vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self._tfidf_matrix).flatten()
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score > self._config.min_score_threshold:
                chunk = self._chunks[idx]
                results.append(SearchResult(
                    product_name=chunk.product_name,
                    content=chunk.content,
                    score=float(score),
                ))
        
        return results


# Global instance (lazy initialized)
_rag_service: Optional[ProductRagService] = None


def get_rag_service(config: RagConfig = None) -> ProductRagService:
    """
    Get the global RAG service instance.
    
    Args:
        config: RAG configuration (only used on first call)
        
    Returns:
        ProductRagService instance
    """
    global _rag_service
    if _rag_service is None:
        from config.settings import get_settings
        config = config or get_settings().rag
        _rag_service = ProductRagService(config)
    return _rag_service


async def initialize_rag_service(config: RagConfig = None) -> ProductRagService:
    """
    Initialize the RAG service at startup.
    
    Args:
        config: RAG configuration
        
    Returns:
        Initialized ProductRagService instance
    """
    service = get_rag_service(config)
    await service.initialize()
    return service

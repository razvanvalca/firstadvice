"""
Product RAG System with TF-IDF

Provides semantic-ish search over product documentation using TF-IDF.
Much lighter weight than neural embeddings, no PyTorch needed.
The index is built at startup and kept in memory.
"""

import os
import re
import asyncio
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ProductChunk:
    """A chunk of product documentation"""
    product_name: str
    content: str


@dataclass  
class SearchResult:
    """A search result with score"""
    product_name: str
    content: str
    score: float


class ProductRAG:
    """
    In-memory RAG system for product documentation using TF-IDF.
    
    - Loads and chunks markdown files at startup
    - Creates TF-IDF index for fast similarity search
    - Provides async search method
    """
    
    def __init__(self, docs_path: str = "sl_products.md"):
        self.docs_path = Path(docs_path)
        self.chunks: List[ProductChunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
        
        # Product summary for system prompt (names + short descriptions)
        self.product_summary = ""
    
    async def initialize(self):
        """Initialize the RAG system (lazy, thread-safe)"""
        if self._initialized:
            return
        
        async with self._init_lock:
            if self._initialized:
                return
            
            # Run CPU-intensive work in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._build_index)
            self._initialized = True
    
    def _build_index(self):
        """Build TF-IDF index from documents (runs in thread pool)"""
        print(f"[RAG] Building TF-IDF index from {self.docs_path}...")
        
        # Load and chunk the document
        self._load_and_chunk()
        
        if not self.chunks:
            print("[RAG] No chunks found!")
            return
        
        # Build TF-IDF index
        texts = [chunk.content for chunk in self.chunks]
        
        # Use TF-IDF with n-grams for better matching
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),  # unigrams and bigrams
            stop_words='english',
            max_features=5000,
            sublinear_tf=True  # Use log(1 + tf) for better results
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        print(f"[RAG] TF-IDF index built with {len(self.chunks)} chunks, {self.tfidf_matrix.shape[1]} features")
    
    def _load_and_chunk(self):
        """Load markdown and split into product-based chunks"""
        if not self.docs_path.exists():
            print(f"[RAG] Document not found: {self.docs_path}")
            return
        
        content = self.docs_path.read_text(encoding='utf-8')
        
        # Split by product sections (### headers)
        # Pattern matches "### 3.X Product Name" or "### Product Name"
        sections = re.split(r'\n(?=###\s+(?:\d+\.\d+\s+)?Swiss Life)', content)
        
        product_summaries = []
        
        for section in sections:
            section = section.strip()
            if not section or len(section) < 50:
                continue
            
            # Extract product name from header
            header_match = re.match(r'###\s+(?:\d+\.\d+\s+)?(Swiss Life[^\n]+)', section)
            if header_match:
                product_name = header_match.group(1).strip()
            else:
                # Try to find product name in content
                name_match = re.search(r'(Swiss Life [A-Za-z\s]+(?:Uno|Duo|Plan|Expert|Start)?)', section)
                product_name = name_match.group(1) if name_match else "Swiss Life Product"
            
            # Create chunk
            self.chunks.append(ProductChunk(
                product_name=product_name,
                content=section
            ))
            
            # Extract short description for summary
            # Look for first sentence or key features
            desc_match = re.search(r'\*\*Key Features:\*\*\s*\n-\s*([^\n]+)', section)
            if desc_match:
                short_desc = desc_match.group(1).strip()[:60]
            else:
                # Use first meaningful line
                lines = [l.strip() for l in section.split('\n') if l.strip() and not l.startswith('#')]
                short_desc = lines[0][:60] if lines else "Insurance product"
            
            product_summaries.append(f"- {product_name}: {short_desc}")
        
        # Build product summary for system prompt
        self.product_summary = "\n".join(product_summaries)
        print(f"[RAG] Loaded {len(self.chunks)} product chunks")
    
    async def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """
        Search for relevant product information.
        
        Args:
            query: Search query from LLM
            top_k: Number of results to return
            
        Returns:
            List of SearchResult with product info and relevance scores
        """
        await self.initialize()
        
        if self.vectorizer is None or self.tfidf_matrix is None or not self.chunks:
            return []
        
        # Run search in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self._search_sync(query, top_k)
        )
    
    def _search_sync(self, query: str, top_k: int) -> List[SearchResult]:
        """Synchronous search (runs in thread pool)"""
        # Transform query using the same vectorizer
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score > 0.01:  # Filter out very low scores
                chunk = self.chunks[idx]
                results.append(SearchResult(
                    product_name=chunk.product_name,
                    content=chunk.content,
                    score=float(score)
                ))
        
        return results
    
    def get_product_summary(self) -> str:
        """Get product summary for system prompt"""
        if not self._initialized:
            # Load synchronously if needed
            self._load_and_chunk()
        return self.product_summary


# Global instance
_product_rag: Optional[ProductRAG] = None


def get_product_rag() -> ProductRAG:
    """Get or create the global ProductRAG instance"""
    global _product_rag
    if _product_rag is None:
        _product_rag = ProductRAG()
    return _product_rag


async def initialize_rag():
    """Initialize RAG at startup (call this early)"""
    rag = get_product_rag()
    await rag.initialize()
    return rag

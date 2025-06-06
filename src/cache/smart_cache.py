"""
Smart RAG Caching System for Multi-Step API Only
Simple API is designed for direct processing without cache
"""

import os
import time
import json
import hashlib
import re
import ast
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from fastapi.encoders import jsonable_encoder

# Caching
import redis
from sklearn.metrics.pairwise import cosine_similarity

# Langchain imports for embeddings
from langchain_openai import OpenAIEmbeddings

@dataclass
class CacheEntry:
    """Cache entry structure with metadata"""
    query: str
    answer: str
    timestamp: float
    ttl_seconds: int
    metadata: Dict[str, Any]
    query_hash: str
    embedding: Optional[List[float]] = None
    access_count: int = 0
    last_accessed: float = 0.0
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return time.time() > (self.timestamp + self.ttl_seconds)
    
    def is_similar(self, query_embedding: List[float], threshold: float = 0.80) -> bool:
        """Check if query is similar to cached entry using cosine similarity"""
        if not self.embedding or not query_embedding:
            return False
        
        # Calculate cosine similarity
        dot_product = np.dot(self.embedding, query_embedding)
        norm_a = np.linalg.norm(self.embedding)
        norm_b = np.linalg.norm(query_embedding)
        
        if norm_a == 0 or norm_b == 0:
            return False
        
        similarity = dot_product / (norm_a * norm_b)
        return similarity >= threshold
    
    def update_access(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_accessed = time.time()

# Utility function for safe parsing
def safe_parse(payload: str) -> Dict[str, Any]:
    """
    Parse string payload dengan fallback ke ast.literal_eval() jika json.loads() gagal.
    """
    try:
        return json.loads(payload)  # ① coba JSON murni
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(payload)  # ② coba repr(dict)
        except Exception as e:
            print(f"[WARNING] Parse failed: {e}")
            return {}  # ③ terakhir, kosong

class SmartRAGCache:
    """Multi-level caching system untuk RAG responses"""

    def __init__(self):
        # Redis connection dengan fallback
        redis_url = os.getenv("REDIS_URL")  # Prioritaskan os.getenv
        if not redis_url:
            # Fallback ke os.environ jika getenv gagal
            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")

        # Debug logging untuk troubleshooting
        print(f"[CACHE] 🔍 Attempting Redis connection...")
        print(f"[CACHE] 🔍 Redis URL available: {'Yes' if redis_url else 'No'}")

        try:
            if not redis_url:
                raise Exception("REDIS_URL environment variable not found")

            self.redis_client = redis.from_url(
                redis_url, socket_connect_timeout=10, socket_timeout=10
            )
            # Test connection
            self.redis_client.ping()
            self.cache_enabled = True
            # Mask password in log for security
            display_url = redis_url
            if "@" in display_url:
                # Replace password with *** for security
                parts = display_url.split("@")
                if ":" in parts[0]:
                    user_pass = parts[0].split(":")
                    if len(user_pass) >= 3:  # redis://default:password format
                        display_url = f"{user_pass[0]}:{user_pass[1]}:***@{parts[1]}"
            print(f"[CACHE] ✅ Connected to Redis Cloud: {display_url}")
        except Exception as e:
            print(f"[CACHE] ❌ Redis connection failed: {e}")
            print(f"[CACHE] 🔍 REDIS_URL present: {'Yes' if redis_url else 'No'}")
            print("[CACHE] 🔄 Running without cache")
            self.cache_enabled = False
            self.redis_client = None

        # Cache settings
        self.similarity_threshold = 0.85
        self.exact_ttl = 3600  # 1 hour
        self.semantic_ttl = 3600  # 1 hour
        self.document_ttl = 7200  # 2 hours

        # In-memory fallback untuk embeddings
        self.embedding_cache = {}

    async def get_cached_response(
        self, query: str, embedding_model: str
    ) -> Optional[Dict]:
        """Multi-level cache lookup"""
        if not self.cache_enabled:
            return None

        try:
            # 🎯 LEVEL 1: Exact Query Cache (fastest)
            exact_result = await self._get_exact_cache(query, embedding_model)
            if exact_result:
                print(f"[CACHE] ✅ Level 1 HIT - Exact match")
                return exact_result

            # 🧠 LEVEL 2: Semantic Similarity Cache
            semantic_result = await self._get_semantic_cache(query, embedding_model)
            if semantic_result:
                print(f"[CACHE] ✅ Level 2 HIT - Semantic match")
                return semantic_result
            # 📚 LEVEL 3: Document-based Cache - REMOVED for simplicity

            print(f"[CACHE] ❌ MISS - Full pipeline needed")
            return None

        except Exception as e:
            print(f"[CACHE] ⚠️ Cache lookup error: {e}")
            return None

    async def _get_exact_cache(self, query: str, model: str) -> Optional[Dict]:
        """Level 1: Exact query matching"""
        cache_key = f"exact:{self._get_query_hash(query, model)}"
        cached = self.redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
        return None

    async def _get_semantic_cache(self, query: str, model: str) -> Optional[Dict]:
        """Level 2: Semantic similarity matching"""
        try:
            # Generate embedding untuk query
            query_embedding = await self._get_query_embedding(query, model)
            if query_embedding is None:
                return None

            # Cari cached queries dengan embedding serupa
            cached_embeddings = self._get_cached_query_embeddings()

            for cached_id, cached_embedding in cached_embeddings.items():
                try:
                    # Reshape untuk cosine similarity
                    query_emb = np.array(query_embedding).reshape(1, -1)
                    cached_emb = np.array(cached_embedding).reshape(1, -1)

                    similarity = cosine_similarity(query_emb, cached_emb)[0][0]

                    if similarity >= self.similarity_threshold:
                        print(
                            f"[CACHE] Found similar query (similarity: {similarity:.3f})"
                        )
                        cache_key = f"semantic:{cached_id}"
                        cached = self.redis_client.get(cache_key)
                        if cached:
                            return json.loads(cached)

                except Exception as e:
                    print(f"[CACHE] Error comparing embeddings: {e}")
                    continue

        except Exception as e:
            print(f"[CACHE] Semantic cache error: {e}")

        return None

    async def cache_response(self, query: str, model: str, response: Dict):
        """Cache response di multiple levels"""
        if not self.cache_enabled:
            return

        try:
            # Convert response to JSON-serializable format safely
            def make_serializable(obj):
                if hasattr(obj, 'dict'):  # Pydantic models
                    return obj.dict()
                elif isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_serializable(item) for item in obj]
                elif isinstance(obj, (str, int, float, bool, type(None))):
                    return obj
                else:
                    return str(obj)
            
            safe = make_serializable(response)
            
            # Level 1: Exact cache
            exact_key = f"exact:{self._get_query_hash(query, model)}"
            self.redis_client.setex(exact_key, self.exact_ttl, json.dumps(safe))

            # Level 2: Semantic cache
            query_embedding = await self._get_query_embedding(query, model)
            if query_embedding is not None:
                semantic_id = hashlib.md5(str(query_embedding).encode()).hexdigest()[
                    :16
                ]
                embedding_key = f"embedding:{semantic_id}"
                self.redis_client.setex(
                    embedding_key,
                    self.semantic_ttl,
                    json.dumps(
                        query_embedding.tolist()
                        if hasattr(query_embedding, "tolist")
                        else query_embedding
                    ),
                )
                semantic_key = f"semantic:{semantic_id}"
                self.redis_client.setex(
                    semantic_key, self.semantic_ttl, json.dumps(safe)
                )

            print(f"[CACHE] ✅ Response cached successfully")

        except Exception as e:
            print(f"[CACHE] ⚠️ Cache storage error: {e}")

    def _get_query_hash(self, query: str, model: str) -> str:
        """Generate hash untuk exact query matching"""
        return hashlib.md5(f"{query.lower().strip()}_{model}".encode()).hexdigest()

    async def _get_query_embedding(
        self, query: str, model: str
    ) -> Optional[List[float]]:
        """Generate embedding untuk query dengan caching"""
        cache_key = f"emb:{self._get_query_hash(query, model)}"

        # Check in-memory cache dulu
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        try:
            # Generate embedding menggunakan OpenAI embeddings
            embeddings = self._get_embeddings(model)
            query_embedding = embeddings.embed_query(query)

            # Cache in memory untuk request berikutnya
            self.embedding_cache[cache_key] = query_embedding

            # Limit in-memory cache size
            if len(self.embedding_cache) > 100:
                # Remove oldest entries
                oldest_keys = list(self.embedding_cache.keys())[:20]
                for key in oldest_keys:
                    del self.embedding_cache[key]

            return query_embedding

        except Exception as e:
            print(f"[CACHE] Embedding generation error: {e}")
            return None

    def _get_embeddings(self, embedding_model="large"):
        """Dapatkan objek embedding berdasarkan model yang dipilih"""
        EMBEDDING_CONFIG = {
            "small": {"model": "text-embedding-3-small", "table": "documents_small"},
            "large": {"model": "text-embedding-3-large", "table": "documents"},
        }
        
        if embedding_model not in EMBEDDING_CONFIG:
            raise ValueError(f"Model embedding tidak valid: {embedding_model}")

        model_name = EMBEDDING_CONFIG[embedding_model]["model"]
        return OpenAIEmbeddings(model=model_name)

    def _get_cached_query_embeddings(self) -> Dict[str, List[float]]:
        """Ambil semua cached embeddings untuk similarity search"""
        embeddings = {}
        try:
            # Get all embedding keys
            embedding_keys = self.redis_client.keys("embedding:*")
            for key in embedding_keys[:50]:  # Limit untuk performa
                try:
                    cached_data = self.redis_client.get(key)
                    if cached_data:
                        embedding = json.loads(cached_data)
                        embedding_id = key.decode().split(":")[-1]
                        embeddings[embedding_id] = embedding
                except Exception as e:
                    print(f"[CACHE] Error loading embedding {key}: {e}")
                    continue
        except Exception as e:
            print(f"[CACHE] Error getting cached embeddings: {e}")

        return embeddings

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.cache_enabled:
            return {"enabled": False}

        try:
            info = self.redis_client.info()
            return {
                "enabled": True,
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "total_keys": len(self.redis_client.keys("*")),
                "exact_cache_keys": len(self.redis_client.keys("exact:*")),
                "semantic_cache_keys": len(self.redis_client.keys("semantic:*")),
                "document_cache_keys": 0,  # Removed Level 3 cache
                "embedding_cache_keys": len(self.redis_client.keys("embedding:*")),
            }
        except Exception as e:
            return {"enabled": True, "error": str(e)}

    def clear_cache(self):
        """Clear all cache entries"""
        try:
            if not self.cache_enabled:
                print("[CACHE] ⚠️ Cache not enabled, nothing to clear")
                return

            # Clear Redis cache
            self.redis_client.flushall()
            
            # Clear in-memory cache
            self.embedding_cache.clear()
            
            print("🧹 Multi-Step RAG cache cleared successfully")
            
        except Exception as e:
            print(f"⚠️ Failed to clear cache: {e}")

# Initialize global cache system - ONLY for Multi-Step RAG
cache_system = SmartRAGCache()

# Convenience functions - Only support Multi API
def get_cached_response(
    query: str,
    api_type: str = "multi",
    query_embedding: Optional[List[float]] = None,
    metadata: Dict[str, Any] = None
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Get cached response - ONLY for Multi API"""
    
    if api_type != "multi":
        print(f"⚠️ Cache not available for {api_type} API - only Multi-Step API uses cache")
        return None
    
    # Use the legacy cache system for now
    embedding_model = metadata.get("embedding_model", "large") if metadata else "large"
    
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in an async context, we need to handle this differently
            # For now, return None to skip cache in async context
            return None
        else:
            cached = loop.run_until_complete(
                cache_system.get_cached_response(query, embedding_model)
            )
            if cached:
                return cached.get("answer", ""), {"cache_hit": True, "cache_type": "legacy_redis"}
    except Exception as e:
        print(f"[CACHE] Error in async cache lookup: {e}")
    
    return None

def cache_response(
    query: str,
    answer: str,
    api_type: str = "multi",
    query_embedding: Optional[List[float]] = None,
    metadata: Dict[str, Any] = None,
    ttl: Optional[int] = None
):
    """Cache response - ONLY for Multi API"""
    
    if api_type != "multi":
        print(f"⚠️ Cache not available for {api_type} API - only Multi-Step API uses cache")
        return
    
    # Use the legacy cache system
    embedding_model = metadata.get("embedding_model", "large") if metadata else "large"
    
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in an async context, we need to handle this differently
            # For now, skip caching in async context
            return
        else:
            response_data = {
                "answer": answer,
                "referenced_documents": metadata.get("referenced_documents", []) if metadata else [],
                "model_info": {"cached": False, "model": embedding_model}
            }
            loop.run_until_complete(
                cache_system.cache_response(query, embedding_model, response_data)
            )
    except Exception as e:
        print(f"[CACHE] Error in async cache storage: {e}")

def get_cache_stats(api_type: str = "multi") -> Dict[str, Any]:
    """Get cache statistics - ONLY for Multi API"""
    
    if api_type != "multi":
        return {
            "enabled": False,
            "reason": f"{api_type} API doesn't use cache",
            "cache_policy": "Simple API is designed for direct processing without cache"
        }
    
    return cache_system.get_cache_stats()

def clear_cache(api_type: str = "multi"):
    """Clear cache - ONLY for Multi API"""
    
    if api_type != "multi":
        print(f"⚠️ Cache not available for {api_type} API - only Multi-Step API uses cache")
        return
    
    cache_system.clear_cache()
    print(f"🧹 Multi-Step RAG cache cleared")
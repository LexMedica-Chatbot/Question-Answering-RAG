"""
LangFuse Configuration for RAG System Observability
Simple setup focused on cost, tokens, and RAG performance
"""

import os
from typing import Dict, Any, Optional
import time

# Simple tracking without requiring environment variables
class SimpleLangFuseTracker:
    """Simple LangFuse tracker that works with or without API keys"""
    
    def __init__(self):
        self.enabled = False
        self.langfuse_client = None
        self.init_langfuse()
    
    def init_langfuse(self):
        """Initialize LangFuse with environment variables"""
        try:
            # Check for LangFuse credentials
            public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
            secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
            host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
            
            if not public_key or not secret_key:
                print("⚠️  LangFuse keys not found. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
                print("   Get your keys from: https://cloud.langfuse.com")
                print("   Running without LangFuse tracking...")
                self.enabled = False
                return
            
            # Try to import and initialize LangFuse
            from langfuse import Langfuse
            
            self.langfuse_client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host
            )
            
            # Test connection
            self.langfuse_client.auth_check()
            self.enabled = True
            print("✅ LangFuse initialized successfully!")
            
        except ImportError:
            print("⚠️  LangFuse library not installed. Install with: pip install langfuse")
            self.enabled = False
        except Exception as e:
            print(f"❌ LangFuse initialization failed: {e}")
            self.enabled = False
    
    def create_trace(self, name: str, input_data: Dict[str, Any], metadata: Dict[str, Any] = None):
        """Create a new trace"""
        if not self.enabled:
            return None
        
        try:
            trace = self.langfuse_client.trace(
                name=name,
                input=input_data,
                metadata=metadata or {}
            )
            return trace
        except Exception as e:
            print(f"❌ Failed to create trace: {e}")
            return None
    
    def add_span(self, trace, name: str, input_data: Any = None, output_data: Any = None, metadata: Dict[str, Any] = None):
        """Add a span to a trace"""
        if not self.enabled or not trace:
            return None
        
        try:
            span = trace.span(
                name=name,
                input=input_data,
                output=output_data,
                metadata=metadata or {}
            )
            return span
        except Exception as e:
            print(f"❌ Failed to create span: {e}")
            return None
    
    def add_generation(self, trace, name: str, model: str, input_data: Any, output_data: Any, usage: Dict[str, int] = None, metadata: Dict[str, Any] = None):
        """Add a generation (LLM call) to a trace"""
        if not self.enabled or not trace:
            return None
        
        try:
            generation = trace.generation(
                name=name,
                model=model,
                input=input_data,
                output=output_data,
                usage=usage,
                metadata=metadata or {}
            )
            return generation
        except Exception as e:
            print(f"❌ Failed to create generation: {e}")
            return None
    
    def finalize_trace(self, trace, output_data: Any, metadata: Dict[str, Any] = None):
        """Finalize a trace with output"""
        if not self.enabled or not trace:
            return
        
        try:
            trace.update(
                output=output_data,
                metadata=metadata or {}
            )
            # Flush data to ensure it's sent to LangFuse
            if self.langfuse_client:
                self.langfuse_client.flush()
                print("📊 LangFuse data flushed to dashboard")
        except Exception as e:
            print(f"❌ Failed to finalize trace: {e}")

# Global tracker instance
langfuse_tracker = SimpleLangFuseTracker()

# Helper functions for easy tracking
def track_rag_session(user_query: str, embedding_model: str) -> Optional[Any]:
    """Start tracking a RAG session"""
    if not langfuse_tracker.enabled:
        return None
    
    trace = langfuse_tracker.create_trace(
        name="rag_session",
        input_data={"user_query": user_query},
        metadata={
            "api_type": "simple",
            "embedding_model": embedding_model,
            "timestamp": time.time()
        }
    )
    return trace

def track_document_retrieval(trace, query: str, model: str, num_docs: int, docs: list = None):
    """Track document retrieval operation"""
    if not langfuse_tracker.enabled or not trace:
        return None
    
    span = langfuse_tracker.add_span(
        trace=trace,
        name="document_retrieval",
        input_data={"query": query, "embedding_model": model},
        output_data={"num_documents": num_docs, "documents": docs[:2] if docs else []},
        metadata={
            "operation": "document_retrieval",
            "embedding_model": model,
            "num_documents": num_docs,
        }
    )
    return span

def track_llm_call(trace, model: str, messages: list, response: str, usage: Dict[str, int] = None):
    """Track LLM API call with cost and token usage"""
    if not langfuse_tracker.enabled or not trace:
        return None
    
    # Calculate cost (approximate)
    cost = calculate_cost(model, usage) if usage else 0
    
    generation = langfuse_tracker.add_generation(
        trace=trace,
        name="llm_call",
        model=model,
        input_data=messages,
        output_data=response,
        usage={
            "input": usage.get('prompt_tokens', 0) if usage else 0,
            "output": usage.get('completion_tokens', 0) if usage else 0,
            "total": usage.get('total_tokens', 0) if usage else 0,
            "unit": "TOKENS"
        },
        metadata={
            "model": model,
            "cost_usd": round(cost, 6),
            "input_tokens": usage.get('prompt_tokens', 0) if usage else 0,
            "output_tokens": usage.get('completion_tokens', 0) if usage else 0,
        }
    )
    return generation

def finalize_rag_session(trace, final_answer: str, processing_time_ms: int, estimated_cost: float):
    """Finalize RAG session tracking"""
    if not langfuse_tracker.enabled or not trace:
        return
    
    langfuse_tracker.finalize_trace(
        trace=trace,
        output_data={"answer": final_answer},
        metadata={
            "processing_time_ms": processing_time_ms,
            "estimated_cost_usd": round(estimated_cost, 6),
            "status": "success"
        }
    )

# Cost tracking utilities
PRICING = {
    # OpenAI Pricing (per 1K tokens)
    "gpt-4o": {"input": 0.00250, "output": 0.01000},
    "gpt-4.1-mini": {"input": 0.000150, "output": 0.000600},
    "gpt-4.1-mini": {"input": 0.000150, "output": 0.000600},  # Same as gpt-4.1-mini
    "gpt-4-turbo": {"input": 0.01000, "output": 0.03000},
    "gpt-3.5-turbo": {"input": 0.000500, "output": 0.001500},
    
    # Embeddings
    "text-embedding-3-large": {"input": 0.000130, "output": 0},
    "text-embedding-3-small": {"input": 0.000020, "output": 0},
    "text-embedding-ada-002": {"input": 0.000100, "output": 0},
}

def calculate_cost(model: str, usage: Dict[str, int]) -> float:
    """Calculate cost for model usage"""
    if not usage:
        return 0.0
    
    input_tokens = usage.get('prompt_tokens', 0)
    output_tokens = usage.get('completion_tokens', 0)
    
    # Find pricing for model
    pricing = None
    if model in PRICING:
        pricing = PRICING[model]
    else:
        # Try to find similar model
        for known_model in PRICING:
            if known_model in model.lower():
                pricing = PRICING[known_model]
                break
    
    if not pricing:
        return 0.0
    
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    
    return input_cost + output_cost

# Backward compatibility functions
def track_rag_operation(operation_name: str, metadata=None):
    """Decorator for backward compatibility - now does nothing"""
    def decorator(func):
        return func
    return decorator

# Simple usage stats
def get_usage_stats():
    """Get usage statistics"""
    if langfuse_tracker.enabled:
        return {
            "status": "enabled",
            "total_requests": "Check LangFuse dashboard",
            "total_cost": "Check LangFuse dashboard", 
            "total_tokens": "Check LangFuse dashboard",
            "dashboard_url": "https://cloud.langfuse.com"
        }
    else:
        return {
            "status": "disabled",
            "message": "LangFuse not configured. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY"
        }

# Legacy compatibility
class CostTracker:
    """Legacy cost tracker class"""
    
    @classmethod
    def calculate_cost(cls, model: str, input_tokens: int, output_tokens: int = 0) -> float:
        """Calculate cost for model usage"""
        usage = {
            'prompt_tokens': input_tokens,
            'completion_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens
        }
        return calculate_cost(model, usage) 
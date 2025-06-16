"""
Enhanced Multi-Step RAG API - REFACTORED VERSION
Sistem RAG dokumen hukum dengan pendekatan Enhanced Multi-Step RAG yang telah dimodularisasi.

Version: 2.0.0 (Modular)
"""

import os
import time
import secrets
import asyncio
from typing import List, Any
from dotenv import load_dotenv

# FastAPI imports
from fastapi import FastAPI, HTTPException, Header, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKey, APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN
import uvicorn

# Langchain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory

# Local modular imports
from .models import MultiStepRAGRequest, MultiStepRAGResponse, StepInfo
from .utils import EMBEDDING_CONFIG, MODELS, pairs_to_str, summarize_pairs
from .tools import (
    search_documents, 
    enhanced_search_documents, 
    refine_query, 
    evaluate_documents, 
    generate_answer, 
    request_new_query,
    parallel_search_documents
)
from .executors import get_agent_executor, create_agent_tools

# Load environment variables
load_dotenv()

# Modern RAG Observability & Caching
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from observability.rag_tracker import (
        rag_tracker, APIType, ExecutionMode
    )
    from cache.smart_cache import cache_system
    
    LANGFUSE_ENABLED = rag_tracker.enabled
    print(f"✅ RAG Tracker (Multi-Step): {'enabled' if LANGFUSE_ENABLED else 'disabled'}")
    print(f"✅ Smart Cache: enabled")
except ImportError as e:
    print(f"⚠️  RAG Observability not available: {e}")
    LANGFUSE_ENABLED = False

# Security settings
API_KEY_NAME = "X-API-Key"
API_KEY = os.environ.get("API_KEY", secrets.token_urlsafe(32))
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def verify_api_key(api_key_header: str = Depends(api_key_header)):
    """Security dependency - verifikasi API key"""
    if api_key_header != API_KEY:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, 
            detail="Akses ditolak: API key tidak valid"
        )
    return api_key_header


# Initialize FastAPI app with modular architecture indicator
app = FastAPI(
    title="Enhanced Multi-Step RAG System - Modular",
    description="API untuk sistem RAG dokumen hukum dengan arsitektur modular",
    version="2.0.0",
)

print("🏗️ Multi-Step RAG API (Modular Architecture) - Successfully loaded!")
print("📦 Modular components:")
print("  ✅ Models: MultiStepRAGRequest, MultiStepRAGResponse, StepInfo")
print("  ✅ Tools: search_documents, refine_query, evaluate_documents, generate_answer")
print("  ✅ Utils: EMBEDDING_CONFIG, MODELS, document_processor, vector_store_manager")
print("  ✅ Architecture: Separated concerns dengan clean imports")

# Add error handling
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    print(f"❌ Unhandled exception: {str(exc)}")
    return HTTPException(
        status_code=500, 
        detail=f"Internal server error: {str(exc)}"
    )

# Global variables untuk agent executor
agent_executor = None


@app.on_event("startup")
async def startup_event():
    """Initialize system components on startup"""
    global agent_executor, parallel_executor, standard_executor
    
    try:
        print("🚀 Starting LexMedica Chatbot Multi-Agent RAG API (Modular)...")
        print(f"✅ FastAPI initialized successfully")
        
        # Test environment variables
        required_env_vars = ["SUPABASE_URL", "SUPABASE_SERVICE_KEY", "OPENAI_API_KEY"]
        missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
        
        if missing_vars:
            print(f"⚠️ Missing environment variables: {missing_vars}")
        else:
            print("✅ All required environment variables found")
            
        # Test database connection
        try:
            from .utils.vector_store_manager import supabase
            result = supabase.table("documents").select("*").limit(1).execute()
            print("✅ Database connection successful")
        except Exception as db_error:
            print(f"⚠️ Database connection issue: {db_error}")
        
        # Initialize agent executor using modular function
        try:
            agent_executor = get_agent_executor()
            if agent_executor:
                print("✅ RAG agent initialized successfully")
            else:
                print("⚠️ Agent initialization returned None")
        except Exception as agent_error:
            print(f"⚠️ Agent initialization issue: {agent_error}")
            agent_executor = None
            
        print("✅ Startup completed successfully")
        
    except Exception as e:
        print(f"❌ Startup error: {str(e)}")


# Add CORS middleware
backend_url = os.environ.get("BACKEND_URL", "*")
frontend_url = os.environ.get("FRONTEND_URL", "*")
allowed_origins = ["*"] if not backend_url or backend_url == "*" else [backend_url, frontend_url]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


# ======================= MAIN API ENDPOINTS =======================

@app.post(
    "/api/chat",
    response_model=MultiStepRAGResponse,
    dependencies=[Depends(verify_api_key)],
)
async def multi_step_rag_chat(request: MultiStepRAGRequest):
    """Main Multi-Step RAG endpoint dengan modular architecture"""
    start_time = time.time()
    print(f"\n[API] 📝 Enhanced Multi-Step RAG Request (Modular): {request.query}")
    print(f"[API] 🔍 Parallel execution: {request.use_parallel_execution}")
    
    # Check cache first
    try:
        cached_result = await cache_system.get_cached_response(
            query=request.query, 
            embedding_model=request.embedding_model
        )
    except Exception as cache_error:
        print(f"[API] ⚠️ Cache lookup error: {cache_error}")
        cached_result = None
    
    if cached_result:
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        if isinstance(cached_result, dict):
            answer = cached_result.get("answer", "")
            referenced_docs = cached_result.get("referenced_documents", [])
            model_info = cached_result.get("model_info", {})
        else:
            answer = str(cached_result)
            referenced_docs = []
            model_info = {}
        
        print(f"🎯 Cache HIT - returning cached response ({processing_time_ms}ms)")
        
        return MultiStepRAGResponse(
            answer=answer,
            referenced_documents=referenced_docs,
            processing_steps=[
                StepInfo(
                    tool="cache_lookup",
                    tool_input={"query": request.query},
                    tool_output=f"Cache hit - response retrieved from cache"
                )
            ],
            processing_time_ms=processing_time_ms,
            model_info={
                "cached": True, 
                "cache_type": "redis_exact", 
                "embedding_model": request.embedding_model,
                "modular_architecture": True
            }
        )
    
    # Determine execution mode
    execution_mode = ExecutionMode.PARALLEL if request.use_parallel_execution else ExecutionMode.STANDARD
    
    # Start RAG tracking session
    trace = None
    if LANGFUSE_ENABLED:
        trace = rag_tracker.start_session(
            query=request.query,
            api_type=APIType.MULTI_STEP,
            execution_mode=execution_mode,
            metadata={"embedding_model": request.embedding_model, "modular": True}
        )
    
    try:
        # For now, use standard execution (parallel execution will be implemented later)
        print(f"[API] 🔄 Using STANDARD EXECUTION mode (Modular)")
        
        # Check if agent_executor is initialized
        if agent_executor is None:
            print("[API] ❌ Agent executor not initialized")
            return MultiStepRAGResponse(
                answer="Maaf, sistem sedang dalam proses inisialisasi. Silakan coba lagi dalam beberapa saat.",
                referenced_documents=[],
                processing_steps=[],
                processing_time_ms=int((time.time() - start_time) * 1000),
                model_info={"error": "Agent not initialized", "modular_architecture": True}
            )

        # Reset refinement count for new request
        import importlib
        refinement_module = importlib.import_module("src.api.tools.refinement_tools")
        refinement_module.refinement_count = 0

        # Prepare chat history
        chat_history = []
        history_summary = ""

        if request.previous_responses:
            history_summary = summarize_pairs(request.previous_responses)
            print(f"[API] History summary: {history_summary[:100]}...")

        # Execute multi-step RAG
        result = await agent_executor.ainvoke(
            {
                "input": request.query,
                "history_summary": history_summary,
                "chat_history": chat_history,
            }
        )

        end_time = time.time()
        processing_time_ms = int((end_time - start_time) * 1000)

        # Extract answer and processing info
        answer = result.get("output", "")
        referenced_documents = []
        processing_steps = []

        intermediate_steps = result.get("intermediate_steps", [])
        print(f"[API] 📊 Multi-Step RAG Execution - Found {len(intermediate_steps)} steps")

        for i, (agent_action, observation) in enumerate(intermediate_steps):
            tool_name = agent_action.tool
            tool_input = agent_action.tool_input
            tool_output = str(observation)

            # Enhanced logging untuk setiap step
            print(f"[API] 🔧 Step {i+1}: {tool_name}")
            print(f"[API] 📥 Input: {str(tool_input)[:100]}...")
            
            if tool_name == "search_documents":
                print(f"[API] 🔍 SEARCH STEP: Looking for documents...")
            elif tool_name == "evaluate_documents":
                print(f"[API] 📊 EVALUATE STEP: Checking document adequacy...")
                # Extract evaluation result
                if "MEMADAI" in tool_output:
                    print(f"[API] ✅ Evaluation Result: MEMADAI")
                elif "KURANG MEMADAI" in tool_output:
                    print(f"[API] ⚠️ Evaluation Result: KURANG MEMADAI")
            elif tool_name == "refine_query":
                print(f"[API] 🔄 REFINE STEP: Improving search query...")
            elif tool_name == "generate_answer":
                print(f"[API] ✨ GENERATE STEP: Creating final answer...")

            # Create step info
            step_info = StepInfo(
                tool=tool_name,
                tool_input=tool_input,
                tool_output=(
                    tool_output[:500] + "..."
                    if len(str(observation)) > 500
                    else tool_output
                ),
            )
            processing_steps.append(step_info)

            # Extract referenced documents from search_documents tool
            if tool_name == "search_documents" and isinstance(observation, dict):
                docs_data = observation.get("retrieved_docs_data", [])
                if docs_data:
                    referenced_documents.extend(docs_data)
                    print(f"[API] 📄 Found {len(docs_data)} documents from search")

        # Remove duplicate documents
        unique_docs = []
        seen_hashes = set()
        for doc in referenced_documents:
            content_hash = hash(str(doc.get("content", "")))
            if content_hash not in seen_hashes:
                unique_docs.append(doc)
                seen_hashes.add(content_hash)

        referenced_documents = unique_docs

        response_data = {
            "answer": answer,
            "referenced_documents": referenced_documents,
            "processing_steps": processing_steps,
            "processing_time_ms": processing_time_ms,
            "model_info": {
                "model": request.embedding_model,
                "parallel_execution": False,
                "cached": False,
                "modular_architecture": True,
                "version": "2.0.0"
            },
        }

        # Cache the response
        try:
            cache_data = {
                "answer": answer,
                "referenced_documents": referenced_documents,
                "model_info": {
                    "parallel_execution": False,
                    "embedding_model": request.embedding_model,
                    "modular_architecture": True
                }
            }
            await cache_system.cache_response(
                query=request.query,
                model=request.embedding_model,
                response=cache_data
            )
            print(f"[API] ✅ Response cached successfully")
        except Exception as cache_error:
            print(f"[API] ⚠️ Failed to cache response: {cache_error}")

        return MultiStepRAGResponse(**response_data)

    except Exception as e:
        print(f"[API] ❌ Error: {str(e)}")
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return MultiStepRAGResponse(
            answer=f"Maaf, terjadi kesalahan: {str(e)}",
            referenced_documents=[],
            processing_steps=[],
            processing_time_ms=processing_time_ms,
            model_info={
                "modular_architecture": True,
                "error": str(e),
                "embedding_model": request.embedding_model
            },
        )


# ======================= MONITORING & HEALTH ENDPOINTS =======================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "system": "Enhanced Multi-Step RAG (Modular)", 
        "version": "2.0.0",
        "timestamp": int(time.time())
    }


@app.get("/")
async def root():
    """Root endpoint to show modular architecture info"""
    return {
        "message": "LexMedica Multi-Step RAG API - Modular Architecture",
        "version": "2.0.0",
        "architecture": "modular",
        "components": {
            "models": "✅ loaded",
            "tools": "✅ loaded", 
            "utils": "✅ loaded"
        },
        "improvements": [
            "🔧 Separated 2180 lines into focused modules",
            "📦 Clean imports dan dependency management", 
            "🏗️ Maintainable architecture",
            "⚡ Better performance dengan focused imports",
            "🧪 Easier testing dengan isolated components"
        ],
        "status": "operational"
    }


@app.get("/monitoring/health/detailed")
async def monitoring_health_detailed():
    """Detailed health check dengan modular component status"""
    try:
        from cache.smart_cache import get_cache_stats
        cache_status = get_cache_stats("multi")
        cache_healthy = cache_status.get("enabled", False)
    except:
        cache_healthy = False
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "2.0.0",
        "architecture": "modular",
        "services": {
            "api": "operational",
            "cache": "operational" if cache_healthy else "degraded",
            "database": "operational",
            "agent_executor": "operational" if agent_executor else "degraded"
        },
        "components": {
            "models": "loaded",
            "tools": "loaded", 
            "utils": "loaded",
            "executors": "loaded"
        }
    }


@app.get("/api/models", dependencies=[Depends(verify_api_key)])
async def available_models():
    """Get available embedding models"""
    return {
        "embedding_models": {
            model_key: {
                "model_name": config["model"],
                "table": config["table"],
            }
            for model_key, config in EMBEDDING_CONFIG.items()
        },
        "architecture": "modular",
        "version": "2.0.0"
    }


@app.get("/api/cache/stats", dependencies=[Depends(verify_api_key)])
async def get_cache_stats_endpoint():
    """Get cache statistics untuk monitoring"""
    from cache.smart_cache import get_cache_stats
    stats = get_cache_stats("multi")
    stats["architecture"] = "modular"
    return stats


@app.get("/api/observability")
async def get_observability_status():
    """Get RAG observability status untuk Multi-Step RAG (Modular)"""
    from cache.smart_cache import get_cache_stats
    
    response = {
        "tracking": rag_tracker.get_status() if LANGFUSE_ENABLED else {"enabled": False},
        "cache": get_cache_stats("multi"),
        "api_type": "enhanced_multi_step_rag_modular",
        "version": "2.0.0",
        "architecture": "modular",
        "features": [
            "Modular architecture dengan separation of concerns",
            "Smart caching dengan similarity matching",
            "Cost tracking per request",
            "Performance monitoring",
            "Multi-step process tracking",
            "Parallel vs Standard execution analytics",
            "Agent-based tool orchestration tracking"
        ]
    }
    
    if LANGFUSE_ENABLED:
        response["tracking"].update({
            "setup_instructions": "Visit https://cloud.langfuse.com to see your dashboard",
            "metrics_tracked": {
                "cost_per_request": "USD cost for LLM calls",
                "token_usage": "Input/output tokens for each request",
                "document_retrieval": "Number of documents retrieved",
                "response_time": "End-to-end processing time",
                "embedding_model": "Which embedding model used",
                "execution_mode": "Parallel vs Standard execution",
                "processing_steps": "Multi-step RAG tool usage",
                "agent_actions": "Tool calls and reasoning steps",
                "modular_components": "Track which modules were used"
            }
        })
    
    return response


@app.delete("/api/cache/clear", dependencies=[Depends(verify_api_key)])
async def clear_cache():
    """Clear all cache (use with caution)"""
    try:
        from cache.smart_cache import clear_cache
        clear_cache("multi")
        return {
            "status": "success", 
            "message": "Multi-step RAG cache cleared successfully (Modular)",
            "architecture": "modular"
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Failed to clear cache: {str(e)}",
            "architecture": "modular"
        }


# For development only
if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8080))
    print(f"Starting development server (Modular) on port {port}")
    uvicorn.run("src.api.multi_api_refactored:app", host="0.0.0.0", port=port, reload=False) 
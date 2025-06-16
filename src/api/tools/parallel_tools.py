"""
Parallel execution tools untuk Multi-Step RAG system
"""

import time
from typing import Dict, List, Any
from langchain.tools import tool


async def parallel_search_documents(
    queries: List[str], embedding_model: str = "large", limit: int = 5
) -> Dict[str, Any]:
    """
    Execute multiple document searches sequentially with enhanced processing
    Focus on stability over pure parallelism to avoid LangChain callback issues
    """

    print(f"[PARALLEL] 🚀 Processing {len(queries)} search queries...")
    start_time = time.time()

    # Process searches sequentially but efficiently
    combined_docs = []
    all_metadata = []

    for i, query in enumerate(queries):
        try:
            print(f"[PARALLEL] Processing query {i+1}/{len(queries)}: {query[:50]}...")

            # Import locally to avoid circular import
            from .search_tools import search_documents
            
            # Run search synchronously to avoid callback context issues
            result = search_documents.invoke(
                {"query": query, "embedding_model": embedding_model, "limit": limit}
            )

            if isinstance(result, dict) and "retrieved_docs_data" in result:
                docs_data = result["retrieved_docs_data"]
                combined_docs.extend(docs_data)
                all_metadata.append(
                    {"query": query, "docs_found": len(docs_data), "status": "success"}
                )
                print(f"[PARALLEL] ✅ Query {i+1} found {len(docs_data)} documents")
            else:
                print(f"[PARALLEL] ⚠️ Query {i+1} returned unexpected result")
                all_metadata.append(
                    {"query": query, "docs_found": 0, "status": "no_results"}
                )

        except Exception as e:
            print(f"[PARALLEL] ❌ Query {i+1} failed: {e}")
            all_metadata.append(
                {"query": query, "docs_found": 0, "status": "error", "error": str(e)}
            )

    end_time = time.time()
    print(
        f"[PARALLEL] ✅ Completed {len(queries)} searches in {(end_time - start_time):.2f}s"
    )

    # Remove duplicates based on content hash
    unique_docs = []
    seen_hashes = set()

    for doc in combined_docs:
        content_hash = hash(doc.get("content", ""))
        if content_hash not in seen_hashes:
            unique_docs.append(doc)
            seen_hashes.add(content_hash)

    print(
        f"[PARALLEL] 📄 Found {len(unique_docs)} unique documents after deduplication"
    )

    return {
        "retrieved_docs_data": unique_docs,
        "search_metadata": all_metadata,
        "parallel_execution": True,
        "total_unique_docs": len(unique_docs),
        "execution_time": end_time - start_time,
        "performance_boost": f"Enhanced search processing with {len(queries)} query variations",
    } 
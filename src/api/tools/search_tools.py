"""
Search tools untuk Multi-Step RAG system
"""

import time
from typing import Dict, List, Any
from langchain.tools import tool
from ..utils.vector_store_manager import get_vector_store
from ..utils.document_processor import format_docs, extract_document_info, clean_control


@tool
def search_documents(
    query: str, embedding_model: str = "large", limit: int = 5
) -> Dict[str, Any]:
    """
    Mencari dokumen dari vectorstore berdasarkan kueri yang diberikan.

    Args:
        query: Query pencarian untuk menemukan dokumen yang relevan
        embedding_model: Model embedding yang digunakan ("small" atau "large")
        limit: Jumlah dokumen yang dikembalikan

    Returns:
        Dictionary berisi dokumen yang diformat untuk LLM dan data dokumen terstruktur
    """
    try:
        print(f"\n[TOOL] Searching for documents with query: {query}")

        # Gunakan vector store dari global cache
        vector_store = get_vector_store(embedding_model)
        retriever = vector_store.as_retriever(search_kwargs={"k": limit})
        docs = retriever.invoke(query)

        if not docs:
            return {
                "formatted_docs_for_llm": "Tidak ditemukan dokumen yang relevan dengan query tersebut.",
                "retrieved_docs_data": [],
            }

        # Format dokumen untuk konteks LLM
        formatted_docs_for_llm = format_docs(docs)

        # Siapkan data terstruktur untuk setiap dokumen yang diambil
        retrieved_docs_data = []
        for i, doc in enumerate(docs):
            metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
            content = clean_control(doc.page_content)  # Bersihkan karakter kontrol

            # Ekstrak informasi dasar dari metadata
            jenis_peraturan = metadata.get("jenis_peraturan", "")
            nomor_peraturan = metadata.get("nomor_peraturan", "")
            tahun_peraturan = metadata.get("tahun_peraturan", "")
            tipe_bagian = metadata.get("tipe_bagian", "")
            status = metadata.get("status", "berlaku")
            bagian_dari = metadata.get("bagian_dari", "")
            judul_peraturan = metadata.get("judul_peraturan", "")

            # Jika metadata tidak lengkap, coba ekstrak dari konten
            if not all([jenis_peraturan, nomor_peraturan, tahun_peraturan]):
                extracted_info = extract_document_info(content)
                jenis_peraturan = extracted_info.get("jenis_peraturan", jenis_peraturan)
                nomor_peraturan = extracted_info.get("nomor_peraturan", nomor_peraturan)
                tahun_peraturan = extracted_info.get("tahun_peraturan", tahun_peraturan)
                tipe_bagian = extracted_info.get("tipe_bagian", tipe_bagian)
                status = extracted_info.get("status", status)
                bagian_dari = extracted_info.get("bagian_dari", bagian_dari)
                judul_peraturan = extracted_info.get("judul_peraturan", judul_peraturan)

            # Buat nama dokumen
            doc_name = ""
            if jenis_peraturan and nomor_peraturan and tahun_peraturan:
                doc_name = (
                    f"{jenis_peraturan} No. {nomor_peraturan} Tahun {tahun_peraturan}"
                )
                if tipe_bagian:
                    doc_name += f" {tipe_bagian}"
                if judul_peraturan:
                    doc_name += f" tentang {judul_peraturan}"

            # Buat metadata terstruktur
            structured_metadata = {
                "status": status,
                "bagian_dari": bagian_dari,
                "tipe_bagian": tipe_bagian,
                "jenis_peraturan": jenis_peraturan,
                "judul_peraturan": judul_peraturan,
                "nomor_peraturan": nomor_peraturan,
                "tahun_peraturan": tahun_peraturan,
            }

            # Buat label peraturan untuk ditampilkan di frontend
            peraturan_label = ""
            if jenis_peraturan and nomor_peraturan and tahun_peraturan:
                peraturan_label = (
                    f"{jenis_peraturan} No. {nomor_peraturan} Tahun {tahun_peraturan}"
                )
                if tipe_bagian:
                    peraturan_label += f" {tipe_bagian}"
                if judul_peraturan:
                    peraturan_label += f" tentang {judul_peraturan}"

            retrieved_docs_data.append(
                {
                    "name": f"Dokumen #{i+1}",
                    "source": doc_name,
                    "content": content,
                    "metadata": {
                        **structured_metadata,
                        "label": peraturan_label,  # Tambahkan label untuk frontend
                    },
                }
            )

        print(f"[TOOL] Found {len(docs)} documents")

        return {
            "formatted_docs_for_llm": formatted_docs_for_llm,
            "retrieved_docs_data": retrieved_docs_data,
        }
    except Exception as e:
        print(f"[ERROR] Error pada pencarian dokumen: {str(e)}")
        return {
            "formatted_docs_for_llm": f"Error pada pencarian dokumen: {str(e)}",
            "retrieved_docs_data": [],
        }


@tool
async def enhanced_search_documents(
    query: str,
    embedding_model: str = "large",
    limit: int = 5,
) -> Dict[str, Any]:
    """
    Enhanced search dengan parallel execution untuk complex queries
    Automatically generates search variations untuk better coverage
    """
    print(f"[TOOL] Enhanced search for: {query}")

    # Generate multiple search variations untuk parallel execution
    search_variations = [
        query,  # Original query
        f"definisi {query}",  # Definition variant
        f"peraturan {query}",  # Regulation variant
        f"{query} Indonesia",  # Localized variant
    ]

    # Remove duplicates while preserving order
    unique_queries = []
    seen = set()
    for q in search_variations:
        if q.lower() not in seen:
            unique_queries.append(q)
            seen.add(q.lower())

    # Limit to max 3 variations untuk efficiency
    unique_queries = unique_queries[:3]

    try:
        # Import parallel_search_documents locally to avoid circular import
        from .parallel_tools import parallel_search_documents
        
        # Execute parallel search
        result = await parallel_search_documents(unique_queries, embedding_model, limit)
        print(
            f"[TOOL] Parallel search completed: {result['total_unique_docs']} unique docs"
        )
        return result
    except Exception as e:
        print(
            f"[TOOL] ⚠️ Parallel search failed, falling back to standard search: {str(e)}"
        )
        # Fallback to standard search
        return search_documents.invoke(
            {"query": query, "embedding_model": embedding_model, "limit": limit}
        ) 
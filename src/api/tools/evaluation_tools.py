"""
Document Evaluation Tools untuk Multi-Step RAG system
"""

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from typing import Union, Dict, Any, List
import json
from ..utils.config_manager import MODELS

def safe_parse(input_str: str):
    """
    Safely parse a JSON string, with fallback handling
    """
    try:
        return json.loads(input_str)
    except (json.JSONDecodeError, TypeError):
        return input_str

@tool
def evaluate_documents(
    query: str, documents: Union[str, Dict[str, Any], List[Any]]
) -> str:
    """
    Mengevaluasi kualitas dan relevansi dokumen untuk query.

    Args:
        query: Query yang perlu dijawab (wajib)
        documents: Dict, list, atau string JSON berisi dokumen hasil pencarian

    Returns:
        Hasil evaluasi dokumen: "MEMADAI" atau "KURANG MEMADAI" dengan alasan
    """
    try:
        print(f"\n[TOOL] Evaluating document quality for query: {query}")
        print(f"[DEBUG] Input type: {type(documents)}")

        # --- PARSING INPUT ---
        if isinstance(documents, (dict, list)):
            json_data = (
                {"retrieved_docs_data": documents}
                if isinstance(documents, list)
                else documents
            )
        else:  # string
            # First, try to extract retrieved_docs_data from string
            if isinstance(documents, str):
                # Check if it's a formatted string from agent
                if "retrieved_docs_data" in documents:
                    try:
                        parsed = safe_parse(documents)
                        json_data = parsed if isinstance(parsed, dict) else {"retrieved_docs_data": []}
                    except:
                        json_data = {"retrieved_docs_data": []}
                else:
                    # It's a formatted docs string, skip evaluation for now
                    print(f"[EVAL] ⚠️ Received formatted docs string instead of structured data")
                    return "MEMADAI: Dokumen telah tersedia untuk evaluasi."
            else:
                json_data = {"retrieved_docs_data": []}

        retrieved_docs = json_data.get("retrieved_docs_data", [])
        if not retrieved_docs:
            return "KURANG MEMADAI: Tidak ditemukan dokumen yang relevan."

        # Debug: check what documents we received
        print(f"\n[DEBUG] 📋 Documents received for evaluation:")
        for i, doc in enumerate(retrieved_docs):
            source = doc.get('source', 'Unknown')
            metadata = doc.get('metadata', {})
            status = metadata.get('status')
            print(f"[DEBUG] Doc {i+1}: {source} (Status: {status})")

        # Hitung jumlah dokumen dan status
        doc_count = len(retrieved_docs)
        berlaku_count = sum(
            1
            for doc in retrieved_docs
            if doc.get("metadata", {}).get("status", "").lower() == "berlaku"
        )
        dicabut_count = sum(
            1
            for doc in retrieved_docs
            if doc.get("metadata", {}).get("status", "").lower() == "dicabut"
        )

        print("\n=== EVALUASI DOKUMEN ===")
        print(f"Total dokumen: {doc_count}")
        print(f"Dokumen berlaku: {berlaku_count}")
        print(f"Dokumen dicabut: {dicabut_count}")

        # Cek heuristik tambahan sebelum memanggil LLM
        # Jika query jelas membutuhkan peraturan terkini dan hanya ada yang dicabut
        need_current_info = any(
            term in query.lower()
            for term in [
                "terbaru",
                "saat ini",
                "sekarang",
                "terkini",
                "berlaku",
                "yang masih berlaku",
                "yang masih digunakan",
                "peraturan baru",
            ]
        )

        if need_current_info:
            print("\nQuery membutuhkan informasi peraturan terkini")
            if berlaku_count == 0 and dicabut_count > 0:
                print("⚠️ Peringatan: Hanya ditemukan peraturan yang sudah dicabut")
                return "KURANG MEMADAI: Hanya ditemukan peraturan yang sudah dicabut, sementara query membutuhkan informasi peraturan yang masih berlaku."

        # Format dokumen untuk evaluasi
        formatted_docs = []
        for doc in retrieved_docs:
            metadata = doc.get("metadata", {})
            status = metadata.get("status", "status tidak diketahui")
            jenis = metadata.get("jenis_peraturan", "")
            nomor = metadata.get("nomor_peraturan", "")
            tahun = metadata.get("tahun_peraturan", "")

            # Format referensi dokumen
            ref = (
                f"[{jenis} No. {nomor} Tahun {tahun}] ({status})"
                if all([jenis, nomor, tahun])
                else f"[Dokumen] ({status})"
            )

            formatted_docs.append(f"{ref}\n{doc.get('content', '')}")

        formatted_docs_str = "\n\n".join(formatted_docs)

        # Use LLM to evaluate document adequacy for the query
        evaluator = ChatOpenAI(**MODELS["REF_EVAL"])

        evaluation_prompt = f"""Evaluasi SECARA OBJEKTIF apakah kumpulan dokumen berikut memberikan informasi YANG MEMADAI untuk menjawab query:

Query: {query}

Dokumen:
{formatted_docs_str}

1. Apakah dokumen berisi informasi relevan dengan query tersebut?
2. Apakah Anda dapat memberikan jawaban lengkap berdasarkan dokumen-dokumen ini?
3. Apakah dokumen-dokumen ini mencakup topik utama dari query?
4. Apakah ada peraturan yang masih "berlaku" (bukan dicabut) yang membahas query?

Berikan jawaban MEMADAI atau KURANG MEMADAI diikuti oleh alasan singkat.
Format: "MEMADAI: [alasan]" ATAU "KURANG MEMADAI: [alasan]"
Jawaban harus singkat dan langsung ke poin utama!"""

        result = evaluator.invoke(evaluation_prompt)
        evaluation = result.content.strip()

        print(f"\nHasil evaluasi LLM: {evaluation}")

        # Heuristik tambahan: jika evaluasi menyatakan MEMADAI tapi tidak ada dokumen yang berlaku
        # dan query membutuhkan peraturan terkini, maka override menjadi KURANG MEMADAI
        if (
            "MEMADAI" in evaluation
            and berlaku_count == 0
            and dicabut_count > 0
            and need_current_info
        ):
            evaluation = "KURANG MEMADAI: Hanya ditemukan peraturan yang sudah dicabut, sementara query membutuhkan informasi peraturan yang masih berlaku."
            print(
                f"\n⚠️ Override evaluasi: {evaluation}"
            )

        return evaluation

    except Exception as e:
        print(f"[ERROR] Error pada evaluasi dokumen: {str(e)}")
        return "KURANG MEMADAI: Terjadi error dalam evaluasi dokumen." 
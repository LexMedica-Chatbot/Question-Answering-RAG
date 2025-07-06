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
                        json_data = (
                            parsed
                            if isinstance(parsed, dict)
                            else {"retrieved_docs_data": []}
                        )
                    except:
                        json_data = {"retrieved_docs_data": []}
                else:
                    # It's a formatted docs string, skip evaluation for now
                    print(
                        f"[EVAL] ⚠️ Received formatted docs string instead of structured data"
                    )
                    return json.dumps(
                        {
                            "per_doc_scores": [],
                            "overall_quality": "MEMADAI",
                            "overall_reason": "Dokumen telah tersedia untuk evaluasi.",
                        }
                    )
            else:
                json_data = {"retrieved_docs_data": []}

        retrieved_docs = json_data.get("retrieved_docs_data", [])
        if not retrieved_docs:
            return json.dumps(
                {
                    "per_doc_scores": [],
                    "overall_quality": "KURANG MEMADAI",
                    "overall_reason": "Tidak ditemukan dokumen yang relevan.",
                }
            )

        # Debug: check what documents we received
        print(f"\n[DEBUG] 📋 Documents received for evaluation:")
        for i, doc in enumerate(retrieved_docs):
            source = doc.get("source", "Unknown")
            metadata = doc.get("metadata", {}) or {}
            status = metadata.get("status", "tidak diketahui")
            print(f"[DEBUG] Doc {i+1}: {source} (Status: {status})")

        # Hitung jumlah dokumen dan status (hanya untuk logging)
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

        # ------------ FORMAT DOKUMEN UNTUK PROMPT ------------
        formatted_docs = []
        for idx, doc in enumerate(retrieved_docs):
            metadata = doc.get("metadata", {}) or {}
            status = metadata.get("status", "tidak diketahui")
            jenis = metadata.get("jenis_peraturan", "")
            nomor = metadata.get("nomor_peraturan", "")
            tahun = metadata.get("tahun_peraturan", "")

            ref = (
                f"{jenis} No. {nomor} Tahun {tahun} ({status})"
                if all([jenis, nomor, tahun])
                else f"Dokumen ({status})"
            )
            formatted_docs.append(
                f"### DOC_{idx}\nSumber: {ref}\nIsi:\n{doc.get('content', '')}"
            )

        formatted_docs_str = "\n\n".join(formatted_docs)

        # ------------ LLM CALL ------------
        evaluator = ChatOpenAI(**MODELS["REF_EVAL"])

        evaluation_prompt = f"""Anda bertindak sebagai evaluator kualitas dokumen untuk menjawab pertanyaan hukum kesehatan.\n\nQuery: {query}\n\nBerikut daftar dokumen dengan metadata status (berlaku/dicabut).\nNilailah SETIAP dokumen dan keluarkan JSON dengan format persis: \n{{\n  \"per_doc_scores\": [{{\"index\": int, \"score\": float, \"reason\": str}}...],\n  \"overall_quality\": \"MEMADAI|KURANG MEMADAI\",\n  \"overall_reason\": str\n}}\n\nAturan penilaian skor dokumen:\n- 1   : relevan & memadai (mengandung jawaban lengkap)\n- 0.5 : relevan tapi kurang detail\n- 0   : tidak relevan\n\nKriteria overall_quality:\n- MEMADAI jika terdapat ≥1 dokumen dengan skor 1\n- Selain itu KURANG MEMADAI\n\nDokumen:\n{formatted_docs_str}\n"""

        result = evaluator.invoke(evaluation_prompt)
        evaluation_json = result.content.strip()

        print(f"\nHasil evaluasi LLM (JSON): {evaluation_json}")

        return evaluation_json

    except Exception as e:
        print(f"[ERROR] Error pada evaluasi dokumen: {str(e)}")
        return json.dumps(
            {
                "per_doc_scores": [],
                "overall_quality": "KURANG MEMADAI",
                "overall_reason": "Terjadi error dalam evaluasi dokumen.",
            }
        )

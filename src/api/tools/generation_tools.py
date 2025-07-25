"""
Answer Generation Tools untuk Multi-Step RAG system
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


def extract_metadata_with_prompt(source: str, content: str) -> Dict[str, str]:
    """
    Extract metadata from source and content using prompt engineering
    No regex - pure LLM approach
    """
    if not source and not content:
        return {"status": "tidak diketahui", "jenis_peraturan": "tidak diketahui"}

    try:
        llm = ChatOpenAI(
            **{**MODELS["GENERATOR"], "temperature": 0.1, "max_tokens": 200}
        )

        prompt = f"""Ekstrak informasi metadata dari teks hukum berikut:

SOURCE: {source}
CONTENT (first 200 chars): {content[:200]}...

Ekstrak dan berikan dalam format JSON:
{{
    "status": "berlaku atau dicabut",
    "jenis_peraturan": "UU/PP/Perpres/Permenkes/dll",
    "nomor_peraturan": "nomor saja tanpa 'No.'",
    "tahun_peraturan": "tahun 4 digit",
    "tipe_bagian": "Pasal X atau Bab X"
}}

ATURAN:
- Status "berlaku" jika tahun >= 2023 atau tidak ada indikasi dicabut
- Status "dicabut" jika ada kata "dicabut" atau UU lama yang diganti
- Jika tidak bisa ekstrak, gunakan "tidak diketahui"

JSON:"""

        response = llm.invoke(prompt)
        metadata = json.loads(response.content.strip())
        return metadata

    except Exception as e:
        print(f"[METADATA] Error extracting with prompt: {e}")
        return {"status": "tidak diketahui", "jenis_peraturan": "tidak diketahui"}


# Soft filtering function removed - using only hard filtering + LLM evaluation


def filter_documents_by_hard_filtering(
    documents: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Apply only hard filtering to eliminate revoked documents
    Let LLM evaluation handle relevance assessment
    """
    # Hard filtering: eliminate revoked documents
    active_docs = []
    eliminated_count = 0

    for doc in documents:
        status = doc.get("metadata", {}).get("status", "tidak diketahui").lower()

        if status == "dicabut":
            eliminated_count += 1
        else:
            active_docs.append(doc)

    # Log filtering results only if documents were eliminated
    if eliminated_count > 0:
        print(
            f"[FILTER] 🚫 Eliminated {eliminated_count} revoked documents, {len(active_docs)} remaining"
        )

    # Fallback to all documents if no active documents (prevent no-answer)
    if not active_docs:
        print("[FILTER] ⚠️ No active documents found, using all documents as fallback")
        active_docs = documents

    return active_docs


@tool
def generate_answer(
    documents: Union[str, Dict[str, Any], List[Any]],
    query: str = None,
    evaluation_result: str = None,
) -> str:
    """
    Generate comprehensive answer with smart document filtering based on evaluation

    Args:
        documents: Documents from search (can be string, dict, or list)
        query: The original query for context
        evaluation_result: Result from evaluate_documents to guide filtering

    Returns:
        JSON string containing both answer and filtered documents:
        {
            "answer": "Generated answer...",
            "filtered_documents": [...],  # Only documents actually used
            "total_documents_processed": 5,
            "documents_used": 2
        }
    """
    print(f"[TOOL] Generating answer for query: {query}")

    # Parse documents input
    if isinstance(documents, str):
        try:
            documents = json.loads(documents)
        except json.JSONDecodeError:
            print("[ERROR] Failed to parse documents string")
            return json.dumps(
                {
                    "answer": "Maaf, terjadi kesalahan dalam memproses dokumen.",
                    "filtered_documents": [],
                    "total_documents_processed": 0,
                    "documents_used": 0,
                }
            )

    # Extract document list - preserve metadata
    if isinstance(documents, dict):
        doc_list = documents.get("retrieved_docs_data", [])
    elif isinstance(documents, list):
        doc_list = documents
    else:
        print(f"[ERROR] Unexpected documents type: {type(documents)}")
        return json.dumps(
            {
                "answer": "Maaf, format dokumen tidak dapat diproses.",
                "filtered_documents": [],
                "total_documents_processed": 0,
                "documents_used": 0,
            }
        )

    # Ensure all documents have proper structure with metadata
    standardized_docs = []
    for i, doc in enumerate(doc_list):
        if isinstance(doc, dict):
            # Preserve existing structure or create minimal structure
            standardized_doc = {
                "name": doc.get("name", f"Dokumen #{i+1}"),
                "source": doc.get("source", "Unknown"),
                "content": doc.get("content", ""),
                "metadata": doc.get("metadata", {}),
            }

            # Pastikan metadata berisi status; default ke "tidak diketahui" jika hilang
            meta = standardized_doc.get("metadata", {}) or {}
            if not meta.get("status"):
                meta["status"] = "tidak diketahui"
            standardized_doc["metadata"] = meta

            standardized_docs.append(standardized_doc)
        else:
            print(f"[WARNING] Document {i+1} is not a dict: {type(doc)}")

    doc_list = standardized_docs
    original_docs = list(doc_list)  # Simpan salinan sebelum filtering lanjutan

    # Hard filtering: eliminate revoked documents
    active_docs = []
    eliminated_count = 0

    for doc in doc_list:
        status = doc.get("metadata", {}).get("status", "tidak diketahui").lower()

        if status == "dicabut":
            eliminated_count += 1
        else:
            active_docs.append(doc)

    # Log filtering results only if documents were eliminated
    if eliminated_count > 0:
        print(
            f"[GENERATE] 🚫 Eliminated {eliminated_count} revoked documents, {len(active_docs)} remaining"
        )

    doc_list = active_docs

    # Return early if no active documents found
    if not doc_list:
        return json.dumps(
            {
                "answer": "Maaf, semua dokumen yang relevan sudah tidak berlaku (dicabut). Silakan coba dengan pertanyaan yang lebih spesifik atau hubungi ahli hukum untuk informasi terkini.",
                "filtered_documents": [],
                "total_documents_processed": len(original_docs),
                "documents_used": 0,
            }
        )

    if not doc_list:
        return json.dumps(
            {
                "answer": "Maaf, tidak ada dokumen yang tersedia untuk menjawab pertanyaan Anda.",
                "filtered_documents": [],
                "total_documents_processed": 0,
                "documents_used": 0,
            }
        )

    total_documents_processed = len(doc_list)

    # SIMPLIFIED FILTERING: Use only LLM evaluation results
    if query and evaluation_result:
        try:
            parsed_eval = safe_parse(evaluation_result)
            if isinstance(parsed_eval, str):
                # Jika evaluation_result adalah string sederhana seperti "MEMADAI"
                print(
                    f"[GENERATE] 💡 evaluation_result is simple string: {evaluation_result}"
                )
                # Gunakan semua dokumen yang sudah di-hard filter
                print(f"[GENERATE] 📄 Using all {len(doc_list)} active documents")
            else:
                # JSON evaluation result - gunakan dokumen dengan LLM score >= 0.5
                per_doc_scores = {
                    item["index"]: item
                    for item in parsed_eval.get("per_doc_scores", [])
                }

                # Filter dokumen berdasarkan LLM evaluation score
                filtered_docs = []
                for idx, doc in enumerate(doc_list):
                    eval_info = per_doc_scores.get(idx, {"score": 0})
                    eval_score = eval_info.get("score", 0)

                    if eval_score >= 0.5:  # Gunakan dokumen dengan score >= 0.5
                        filtered_docs.append(doc)

                # Urutkan berdasarkan LLM score (1.0 > 0.5)
                filtered_docs.sort(
                    key=lambda doc: per_doc_scores.get(
                        doc_list.index(doc), {"score": 0}
                    ).get("score", 0),
                    reverse=True,
                )

                if filtered_docs:
                    doc_list = filtered_docs
                    print(
                        f"[GENERATE] ✅ Using {len(doc_list)} documents with LLM score >= 0.5"
                    )
                else:
                    # Fallback: gunakan semua dokumen jika tidak ada yang score >= 0.5
                    print(
                        f"[GENERATE] ⚠️ No documents with score >= 0.5, using all {len(doc_list)} documents"
                    )

        except Exception as parse_err:
            print(f"[WARNING] Error processing evaluation_result: {parse_err}")
            print(f"[GENERATE] 🔄 Using all documents as fallback")
            # Fallback: gunakan semua dokumen yang sudah di-hard filter
    else:
        # Fallback: gunakan semua dokumen jika tidak ada evaluation result
        print(
            f"[GENERATE] 📄 No evaluation result, using all {len(doc_list)} documents"
        )

    if not doc_list:
        return json.dumps(
            {
                "answer": "Maaf, tidak ada dokumen yang cukup relevan untuk menjawab pertanyaan Anda.",
                "filtered_documents": [],
                "total_documents_processed": total_documents_processed,
                "documents_used": 0,
            }
        )

    documents_used = len(doc_list)

    # Process each document for enhanced metadata and citation
    processed_docs = []
    reference_mapping = {}

    for i, doc in enumerate(doc_list, 1):
        # Get or extract metadata using prompt engineering
        metadata = doc.get("metadata", {})
        if not metadata or metadata.get("status") == "tidak diketahui":
            source = doc.get("source", "")
            content = doc.get("content", "")
            extracted_metadata = extract_metadata_with_prompt(source, content)
            metadata.update(extracted_metadata)

        # Create reference key
        ref_key = f"DOC_{i}"

        # Build citation string
        jenis = metadata.get("jenis_peraturan", "Dokumen")
        nomor = metadata.get("nomor_peraturan", "")
        tahun = metadata.get("tahun_peraturan", "")
        tipe = metadata.get("tipe_bagian", "")
        status = metadata.get("status", "tidak diketahui")

        if nomor and tahun and tipe:
            citation = f"[{jenis} No. {nomor} Tahun {tahun} {tipe}] ({status})"
        else:
            citation = f"[{doc.get('source', 'Dokumen tidak dikenal')}] ({status})"

        reference_mapping[ref_key] = citation

        processed_docs.append(
            {
                "ref_key": ref_key,
                "citation": citation,
                "content": doc.get("content", ""),
                "source": doc.get("source", ""),
                "metadata": metadata,
            }
        )

    # Prepare context for LLM with focused, relevant documents (sorted by relevance)
    context_parts = []
    reference_list = []

    for doc in processed_docs:
        context_parts.append(
            f"{doc['ref_key']}: {doc['content'][:1000]}..."
        )  # Increased content length
        reference_list.append(f"{doc['ref_key']} = {doc['citation']}")

    document_context = "\n\n".join(context_parts)
    reference_context = "\n".join(reference_list)

    # Enhanced prompt for focused citation WITH BOLD FORMATTING
    prompt = f"""Anda adalah asisten hukum kesehatan Indonesia yang ahli dalam memberikan jawaban yang akurat dengan sitasi yang tepat.

INSTRUKSI KHUSUS:
1. Jawab pertanyaan berdasarkan HANYA dokumen yang diberikan di bawah
2. WAJIB menggunakan sitasi dalam format [DOC_X] untuk setiap fakta atau informasi
3. Fokus pada dokumen yang status "berlaku" jika tersedia
4. Buat jawaban yang ringkas tapi komprehensif
5. Gunakan SEMUA dokumen yang relevan dalam jawaban Anda
6. Urutkan informasi dari yang paling penting/relevan ke yang kurang penting

INSTRUKSI FORMATTING (WAJIB – OUTPUT ANDA AKAN DINILAI):
- Gunakan **bold** untuk kata-kata PENTING seperti: **nama peraturan**, **pasal**, **definisi kunci**, **status hukum**, **sanksi**, **kewajiban**, dll
- Bold untuk **angka/nominal** penting, **tanggal**, **persyaratan**, **prosedur**
- Bold untuk **istilah teknis** dan **konsep hukum** yang penting
- Pastikan **sitasi [DOC_X]** juga di-bold
- Contoh: "Berdasarkan **PP No. 28 Tahun 2024** **[DOC_1]**, **definisi kesehatan** adalah..."
- Jika setelah menulis jawaban Anda melihat ada kata penting yang belum ditebalkan, **perbaiki jawaban Anda sebelum mengirim**.

PERTANYAAN: {query}

DOKUMEN YANG TERSEDIA (diurutkan berdasarkan relevance):
{document_context}

REFERENSI MAPPING:
{reference_context}

JAWABAN (dengan sitasi yang tepat dan formatting bold untuk kata-kata penting):"""

    try:
        llm = ChatOpenAI(
            **{**MODELS["GENERATOR"], "temperature": 0.3, "max_tokens": 1200}
        )

        response = llm.invoke(prompt)
        answer = response.content.strip()

        print(f"[TOOL] Generated answer (preview): {answer[:100]}...")
        print(f"[TOOL] ✅ Answer generation completed with enhanced citation")
        print(
            f"[TOOL] 📊 Documents: {total_documents_processed} processed → {documents_used} used"
        )

        # Return structured result with filtered documents
        result = {
            "answer": answer,
            "filtered_documents": doc_list,  # Original documents that were actually used
            "total_documents_processed": total_documents_processed,
            "documents_used": documents_used,
        }

        return json.dumps(result)

    except Exception as e:
        print(f"[ERROR] Generation failed: {str(e)}")
        return json.dumps(
            {
                "answer": f"Maaf, terjadi kesalahan dalam menghasilkan jawaban: {str(e)}",
                "filtered_documents": [],
                "total_documents_processed": total_documents_processed,
                "documents_used": 0,
            }
        )


@tool
def request_new_query(reason: str) -> str:
    """
    Membuat permintaan kepada pengguna untuk query baru karena query saat ini tidak dapat dijawab.

    Args:
        reason: Alasan mengapa query saat ini tidak dapat dijawab

    Returns:
        Pesan permintaan query baru untuk pengguna
    """
    return f"""Mohon maaf, saya tidak dapat menjawab query Anda saat ini karena: {reason}

Silakan coba pertanyaan lain yang lebih spesifik atau dengan kata kunci yang berbeda. Beberapa saran:
1. Gunakan terminologi hukum kesehatan yang lebih spesifik
2. Sebutkan nomor peraturan atau UU jika Anda mengetahuinya
3. Fokuskan pertanyaan pada aspek tertentu dari topik Anda
4. Coba formulasikan pertanyaan dengan cara berbeda"""

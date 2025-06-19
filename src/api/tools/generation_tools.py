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
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.1, max_tokens=200)

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


def calculate_document_relevance(doc: Dict[str, Any], query: str) -> float:
    """
    Calculate relevance score for a document based on query
    Returns score between 0.0 and 1.0
    """
    content = doc.get("content", "").lower()
    query_lower = query.lower()

    # Basic keyword matching
    query_words = set(query_lower.split())
    content_words = set(content.split())

    # Calculate word overlap
    overlap = len(query_words.intersection(content_words))
    word_score = overlap / len(query_words) if query_words else 0

    # Boost for documents with key definition terms
    definition_terms = [
        "adalah",
        "dimaksud dengan",
        "yang disebut",
        "pengertian",
        "definisi",
    ]
    definition_score = 0.3 if any(term in content for term in definition_terms) else 0

    # Boost for current law (status berlaku)
    status = doc.get("metadata", {}).get("status", "")
    status_score = 0.2 if status == "berlaku" else 0

    # Content length consideration
    content_length = len(content)
    if content_length < 50:
        length_penalty = 0.2
    elif content_length > 2000:
        length_penalty = 0.1
    else:
        length_penalty = 0

    total_score = word_score + definition_score + status_score - length_penalty
    return min(1.0, max(0.0, total_score))


def filter_documents_by_evaluation(
    documents: List[Dict[str, Any]], evaluation_result: str, query: str
) -> List[Dict[str, Any]]:
    """
    Filter documents based on evaluation result and sort by relevance
    Only keep documents that are considered adequate ("MEMADAI")
    """
    print("\n=== DOKUMEN SEBELUM FILTERING ===")
    for i, doc in enumerate(documents, 1):
        source = doc.get("source", "Unknown")
        status = doc.get("metadata", {}).get("status", "tidak diketahui")
        print(f"\nDokumen {i}:")
        print(f"Source: {source}")
        print(f"Status: {status}")
        print(f"Preview: {doc.get('content', '')[:100]}...")

    print("\n=== PERHITUNGAN SKOR RELEVANSI ===")
    # Calculate relevance scores for all documents
    scored_docs = []
    for doc in documents:
        score = calculate_document_relevance(doc, query)
        scored_docs.append((score, doc))
        source = doc.get("source", "Unknown")
        status = doc.get("metadata", {}).get("status", "tidak diketahui")
        print(f"\nDokumen: {source}")
        print(f"Status: {status}")
        print(f"Skor Relevansi: {score:.3f}")

    # Sort by relevance score (descending) - highest relevance first
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    print("\n=== HASIL FILTERING ===")
    # Check evaluation result to determine filtering logic
    if "KURANG MEMADAI" in evaluation_result.upper():
        # If evaluation says inadequate, we might need refinement
        # But still return some docs for potential refinement
        print(
            "[FILTER] Evaluation indicates documents are inadequate - keeping top documents for refinement"
        )
        filtered_docs = [
            doc for score, doc in scored_docs[:3]
        ]  # Keep top 3 for refinement
    else:
        # Evaluation says adequate - filter based on relevance
        print(
            "[FILTER] Evaluation indicates documents are adequate - filtering by relevance"
        )
        min_score = 0.2  # Lower threshold since evaluation approved
        filtered_docs = []
        for score, doc in scored_docs:
            source = doc.get("source", "Unknown")
            status = doc.get("metadata", {}).get("status", "tidak diketahui")
            if score >= min_score and len(filtered_docs) < 5:  # Max 5 docs
                filtered_docs.append(doc)
                print(f"\n✅ Lolos Filtering:")
                print(f"Source: {source}")
                print(f"Status: {status}")
                print(f"Skor: {score:.3f}")
            else:
                print(f"\n❌ Tidak Lolos Filtering:")
                print(f"Source: {source}")
                print(f"Status: {status}")
                print(f"Skor: {score:.3f}")
                print(
                    f"Alasan: {'Skor terlalu rendah' if score < min_score else 'Melebihi batas maksimum dokumen'}"
                )

    print(f"\n=== RINGKASAN ===")
    print(f"Total dokumen awal: {len(documents)}")
    print(f"Dokumen yang lolos filtering: {len(filtered_docs)}")
    print(f"Hasil evaluasi: {evaluation_result}")

    return filtered_docs


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
    print(f"[DEBUG] Input type: {type(documents)}")

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
            standardized_docs.append(standardized_doc)
        else:
            print(f"[WARNING] Document {i+1} is not a dict: {type(doc)}")

    doc_list = standardized_docs

    # Debug: log received documents
    print(f"\n[DEBUG] 📋 Received {len(doc_list)} documents for filtering:")
    for i, doc in enumerate(doc_list):
        source = doc.get("source", "Unknown")
        metadata = doc.get("metadata", {})
        status = metadata.get("status", "tidak diketahui")
        print(f"[DEBUG] Doc {i+1}: {source} (Status: {status})")

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

    # SMART FILTERING: Filter based on evaluation result and sort by relevance
    if query and evaluation_result:
        doc_list = filter_documents_by_evaluation(doc_list, evaluation_result, query)
    else:
        # Fallback: sort by relevance if no evaluation result
        if query:
            scored_docs = [
                (calculate_document_relevance(doc, query), doc) for doc in doc_list
            ]
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            doc_list = [doc for score, doc in scored_docs[:3]]

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

        print(
            f"[DEBUG] {ref_key} final metadata: status={status}, jenis={jenis}, nomor={nomor}, tahun={tahun}, tipe={tipe}"
        )

    print(f"[DEBUG] Enhanced reference mapping:")
    for key, citation in reference_mapping.items():
        print(f"[DEBUG]   {key} -> {citation}")

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

INSTRUKSI FORMATTING (WAJIB):
- Gunakan **bold** untuk kata-kata PENTING seperti: **nama peraturan**, **pasal**, **definisi kunci**, **status hukum**, **sanksi**, **kewajiban**, dll
- Bold untuk **angka/nominal** penting, **tanggal**, **persyaratan**, **prosedur**
- Bold untuk **istilah teknis** dan **konsep hukum** yang penting
- Pastikan **sitasi [DOC_X]** juga di-bold
- Contoh: "Berdasarkan **PP No. 28 Tahun 2024** **[DOC_1]**, **definisi kesehatan** adalah..."

PERTANYAAN: {query}

DOKUMEN YANG TERSEDIA (diurutkan berdasarkan relevance):
{document_context}

REFERENSI MAPPING:
{reference_context}

JAWABAN (dengan sitasi yang tepat dan formatting bold untuk kata-kata penting):"""

    try:
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3, max_tokens=1200)

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

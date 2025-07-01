"""
Query Rewriting Tools untuk Multi-Step RAG system
Implementasi Query Rewriting/Transformation yang mempertimbangkan chat history
"""

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Any, Dict
from ..utils.config_manager import MODELS


@tool
def rewrite_query_with_history(
    current_query: str, chat_history: str = "", previous_responses: List[str] = None
) -> str:
    """
    Rewrite query dengan konteks history untuk meningkatkan kualitas retrieval.
    SELALU menggunakan LLM untuk rewriting, tidak ada fallback.
    """
    try:
        print(f"\n[QUERY REWRITING] 🔄 Original query: {current_query}")
        print(f"[QUERY REWRITING] 📚 Chat history available: {bool(chat_history)}")

        # Gabungkan previous_responses - selalu ada dan berupa array 3 string
        prev_join = "\n\n".join(previous_responses[-2:]) if previous_responses else ""

        combined_history = chat_history or ""
        if prev_join:
            combined_history += f"\n\nJAWABAN SEBELUMNYA:\n{prev_join}"

        print(f"[QUERY REWRITING] 📚 Combined history: {combined_history[:200]}...")

        # Initialize rewriter LLM (model lebih kapabel, di-set via config)
        rewriter = ChatOpenAI(**MODELS["REF_EVAL"])  # model untuk rewriting

        # Create comprehensive rewriting prompt
        rewriting_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Anda adalah expert dalam query rewriting untuk sistem hukum kesehatan Indonesia. 

TUGAS ANDA:
Tulis ulang query pengguna menjadi query yang mandiri (standalone) dan optimal untuk retrieval, dengan mempertimbangkan konteks percakapan sebelumnya.

PRINSIP QUERY REWRITING:
1. **Resolve References**: Ganti pronoun/referensi implisit dengan entitas eksplisit
2. **Add Context**: Tambahkan konteks penting dari percakapan sebelumnya
3. **Maintain Intent**: Pertahankan maksud asli pengguna
4. **Enhance Specificity**: Buat query lebih spesifik untuk domain hukum kesehatan
5. **Preserve Completeness**: Pastikan query bisa dipahami tanpa konteks tambahan

CONTOH TRANSFORMASI DENGAN KONTEKS:
Konteks: "Aborsi diperbolehkan maksimal 40 hari dalam kasus perkosaan"
- "Jadi boleh/engga?" → "Apakah aborsi diperbolehkan secara hukum dalam kasus perkosaan di Indonesia?"
- "Bagaimana sanksinya?" → "Apa sanksi hukum bagi pelaku aborsi yang melebihi batas 40 hari dalam kasus perkosaan?"
- "Terus gimana dong?" → "Bagaimana prosedur hukum aborsi dalam kasus perkosaan menurut peraturan Indonesia?"

CONTOH LAIN:
- "Bagaimana dengan sanksinya?" → "Bagaimana sanksi hukum bagi dokter yang melanggar kewajiban informed consent?"
- "Apa bedanya dengan yang sebelumnya?" → "Apa perbedaan UU Kesehatan terbaru dengan UU Kesehatan sebelumnya?"
- "Jelaskan lebih detail" → "Jelaskan lebih detail tentang prosedur pengajuan izin praktik dokter spesialis"

INSTRUKSI SPESIFIK:
1. WAJIB analisis konteks percakapan untuk mencari topik utama (misal: aborsi, BPJS, obat generik).
2. WAJIB integrasikan topik tersebut ke dalam query rewrite jika query mengandung kata seperti "jadi", "bagaimana", "terus", "boleh/engga".
3. Jangan hanya menambah "hukum kesehatan Indonesia" - gunakan topik spesifik dari konteks.
4. Query hasil harus bisa dipahami tanpa konteks tambahan.
5. Prioritaskan topik dari jawaban terakhir dalam percakapan.

OUTPUT: Hanya berikan query yang sudah ditulis ulang, tanpa penjelasan tambahan.""",
                ),
                (
                    "human",
                    """KONTEKS PERCAKAPAN:
{combined_history}

QUERY ASLI: {current_query}

Tulis ulang query ini menjadi query yang mandiri dan optimal untuk retrieval dokumen hukum kesehatan Indonesia:""",
                ),
            ]
        )

        # Format the prompt
        formatted_prompt = rewriting_prompt.format_prompt(
            combined_history=combined_history,
            current_query=current_query,
        )

        print(f"[QUERY REWRITING] 🔍 Formatted prompt preview:")
        print(f"  - Combined history: {combined_history[:150]}...")
        print(f"  - Current query: {current_query}")

        # Invoke rewriter
        result = rewriter.invoke(formatted_prompt)
        rewritten_query = result.content.strip().replace('"', "").replace("'", "")

        print(f"[QUERY REWRITING] ✅ Rewritten query: {rewritten_query}")
        return rewritten_query

    except Exception as e:
        print(f"[QUERY REWRITING] ❌ Error in rewriting: {str(e)}")
        # Jika error, tetap return query yang sudah ditingkatkan sedikit
        return f"{current_query} hukum kesehatan Indonesia"


@tool
def analyze_query_context_dependency(
    current_query: str, chat_history: str = ""
) -> Dict[str, Any]:
    """
    Analisis apakah query saat ini bergantung pada konteks percakapan sebelumnya.

    Args:
        current_query: Query yang akan dianalisis
        chat_history: Konteks percakapan sebelumnya

    Returns:
        Dictionary dengan analisis dependency dan rekomendasi
    """
    try:
        print(f"\n[CONTEXT ANALYSIS] 🔍 Analyzing query dependency...")

        # Indicators of context dependency
        dependency_indicators = {
            "pronouns": [
                "itu",
                "ini",
                "tersebut",
                "dia",
                "mereka",
                "nya",
                "nya",
                "mereka",
            ],
            "references": [
                "yang tadi",
                "sebelumnya",
                "di atas",
                "yang disebutkan",
                "tadi",
                "barusan",
            ],
            "comparisons": [
                "bedanya",
                "perbedaan",
                "dibanding",
                "versus",
                "vs",
                "beda",
            ],
            "follow_ups": [
                "bagaimana dengan",
                "lalu",
                "kemudian",
                "selanjutnya",
                "jadi",
                "terus",
                "nah",
            ],
            "clarifications": [
                "maksudnya",
                "jelaskan lebih",
                "detail",
                "spesifik",
                "gimana",
                "bagaimana",
            ],
            "implicit_references": [
                "boleh",
                "engga",
                "bisa",
                "tidak",
                "ya",
                "iya",
                "nggak",
            ],
            "continuation": ["dan", "atau", "tapi", "tetapi", "namun", "sedangkan"],
            "short_queries": [],  # Will be handled separately
        }

        query_lower = current_query.lower()
        dependencies_found = {}

        for category, indicators in dependency_indicators.items():
            found_indicators = [ind for ind in indicators if ind in query_lower]
            if found_indicators:
                dependencies_found[category] = found_indicators

        # Special handling for very short queries (likely context-dependent)
        is_short_query = len(current_query.split()) <= 3
        if is_short_query:
            dependencies_found["short_queries"] = ["short_query_detected"]

        # Special patterns for Indonesian context-dependent queries
        context_patterns = [
            r"\bjadi\b.*\?",  # "Jadi boleh/engga?"
            r"\b(boleh|bisa|tidak|engga|nggak)\s*\?",  # "Boleh?", "Bisa?", "Tidak?"
            r"^\w{1,10}\s*\?$",  # Very short questions
            r"\b(gimana|bagaimana)\s*(dengan|nya)\b",  # "Gimana dengan...", "Bagaimana nya"
            r"\b(terus|lalu|kemudian)\b.*\?",  # "Terus gimana?", "Lalu boleh?"
        ]

        import re

        for pattern in context_patterns:
            if re.search(pattern, query_lower):
                if "context_patterns" not in dependencies_found:
                    dependencies_found["context_patterns"] = []
                dependencies_found["context_patterns"].append(pattern)

        # Calculate dependency score with enhanced logic
        total_indicators = sum(
            len(indicators) for indicators in dependencies_found.values()
        )

        # Base score from indicators
        base_score = min(total_indicators / 3.0, 1.0)

        # Boost score for short queries (likely context-dependent)
        if is_short_query:
            base_score += 0.4

        # Boost score if chat history exists and query is ambiguous
        if chat_history and len(current_query.split()) <= 5:
            base_score += 0.3

        dependency_score = min(base_score, 1.0)

        # More aggressive rewriting threshold
        needs_rewriting = (
            dependency_score > 0.2 or len(dependencies_found) > 0 or is_short_query
        )

        analysis_result = {
            "needs_rewriting": needs_rewriting,
            "dependency_score": dependency_score,
            "dependencies_found": dependencies_found,
            "has_chat_history": bool(chat_history),
            "recommendation": "rewrite" if needs_rewriting else "enhance",
        }

        print(f"[CONTEXT ANALYSIS] 📊 Dependency score: {dependency_score:.2f}")
        print(f"[CONTEXT ANALYSIS] 🎯 Needs rewriting: {needs_rewriting}")

        return analysis_result

    except Exception as e:
        print(f"[CONTEXT ANALYSIS] ❌ Error in context analysis: {str(e)}")
        return {
            "needs_rewriting": True,  # Safe default
            "dependency_score": 0.5,
            "dependencies_found": {},
            "has_chat_history": bool(chat_history),
            "recommendation": "rewrite",
        }


@tool
def smart_query_preprocessing(
    current_query: str, chat_history: str = "", previous_responses: List[Any] = None
) -> Dict[str, Any]:
    """
    Preprocessing query cerdas yang menggabungkan analisis konteks dan rewriting.

    Workflow:
    1. Analisis dependency terhadap chat history
    2. Jika perlu, lakukan query rewriting
    3. Return query yang optimal untuk retrieval

    Args:
        current_query: Query asli dari user
        chat_history: Konteks percakapan (bisa dari history_summary agent)
        previous_responses: Response sebelumnya

    Returns:
        Dictionary dengan query yang sudah diproses dan metadata
    """
    try:
        print(f"\n[SMART PREPROCESSING] 🧠 Starting smart query preprocessing...")
        print(f"[SMART PREPROCESSING] 📝 Current query: {current_query}")
        print(f"[SMART PREPROCESSING] 📚 Chat history available: {bool(chat_history)}")

        # Fallback untuk chat_history dari agent context (dipertahankan)
        if not chat_history:
            try:
                import inspect

                frame = inspect.currentframe()
                while frame:
                    if "history_summary" in frame.f_locals:
                        chat_history = frame.f_locals["history_summary"]
                        break
                    frame = frame.f_back
            except Exception:
                pass

        # STEP 1: Analisis context dependency dulu
        context_analysis = analyze_query_context_dependency.func(
            current_query, chat_history
        )

        needs_rewriting = context_analysis.get("needs_rewriting", False)

        # STEP 2: Conditional rewriting berdasarkan analysis
        if needs_rewriting:
            processed_query = rewrite_query_with_history.func(
                current_query, chat_history, previous_responses
            )
            processing_method = "rewritten"
            print(f"[SMART PREPROCESSING] ✅ Query rewritten due to context dependency")
        else:
            processed_query = current_query
            processing_method = "original"
            print(
                f"[SMART PREPROCESSING] ✅ Query kept original - no context dependency detected"
            )

        result = {
            "original_query": current_query,
            "processed_query": processed_query,
            "processing_method": processing_method,
            "context_analysis": context_analysis,
            "improvement_ratio": (
                len(processed_query) / len(current_query) if current_query else 1.0
            ),
        }

        print(f"[SMART PREPROCESSING] ✅ Processing complete:")
        print(f"  - Method: {processing_method}")
        print(f"  - Needs rewriting: {needs_rewriting}")
        print(
            f"  - Dependency score: {context_analysis.get('dependency_score', 0):.2f}"
        )
        return result

    except Exception as e:
        print(f"[SMART PREPROCESSING] ❌ Error in smart preprocessing: {str(e)}")
        return {
            "original_query": current_query,
            "processed_query": current_query,
            "processing_method": "error_fallback",
            "context_analysis": {},
            "improvement_ratio": 1.0,
        }


@tool
def smart_query_preprocessing_with_history(
    current_query: str, history_summary: str = "", previous_responses: List[Any] = None
) -> Dict[str, Any]:
    """
    Preprocessing query cerdas dengan history_summary eksplisit dari agent.

    Tool ini dirancang khusus untuk dipanggil oleh agent dengan history_summary
    yang sudah tersedia dalam context agent.

    Args:
        current_query: Query asli dari user
        history_summary: Ringkasan percakapan sebelumnya dari agent context
        previous_responses: Response sebelumnya untuk konteks tambahan

    Returns:
        Dictionary dengan query yang sudah diproses dan metadata
    """
    try:
        print(f"\n[SMART PREPROCESSING WITH HISTORY] 🧠 Starting preprocessing...")
        print(f"[SMART PREPROCESSING WITH HISTORY] 📝 Current query: {current_query}")
        print(
            f"[SMART PREPROCESSING WITH HISTORY] 📚 History summary: {history_summary[:100] if history_summary else 'None'}..."
        )

        # STEP 1: Analisis context dependency dulu
        context_analysis = analyze_query_context_dependency.func(
            current_query, history_summary
        )

        needs_rewriting = context_analysis.get("needs_rewriting", False)

        # STEP 2: Conditional rewriting berdasarkan analysis
        if needs_rewriting:
            processed_query = rewrite_query_with_history.func(
                current_query, history_summary, previous_responses
            )
            processing_method = "rewritten"
            print(
                f"[SMART PREPROCESSING WITH HISTORY] ✅ Query rewritten due to context dependency"
            )
        else:
            processed_query = current_query
            processing_method = "original"
            print(
                f"[SMART PREPROCESSING WITH HISTORY] ✅ Query kept original - no context dependency detected"
            )

        result = {
            "original_query": current_query,
            "processed_query": processed_query,
            "processing_method": processing_method,
            "context_analysis": context_analysis,
            "improvement_ratio": (
                len(processed_query) / len(current_query) if current_query else 1.0
            ),
        }

        print(f"[SMART PREPROCESSING WITH HISTORY] ✅ Processing complete:")
        print(f"  - Method: {processing_method}")
        print(f"  - Needs rewriting: {needs_rewriting}")
        print(
            f"  - Dependency score: {context_analysis.get('dependency_score', 0):.2f}"
        )
        return result

    except Exception as e:
        print(f"[SMART PREPROCESSING WITH HISTORY] ❌ Error: {str(e)}")
        return {
            "original_query": current_query,
            "processed_query": current_query,
            "processing_method": "error_fallback",
            "context_analysis": {},
            "improvement_ratio": 1.0,
        }

"""
Enhanced Agent Executor untuk Multi-Step RAG system
"""

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Dict, Any, List
from ..utils.config_manager import MODELS
from ..tools import (
    search_documents,
    refine_query,
    evaluate_documents,
    generate_answer,
    request_new_query,
)
from ..tools.query_rewriting_tools import (
    rewrite_query_with_history,
    analyze_query_context_dependency,
    smart_query_preprocessing,
    smart_query_preprocessing_with_history,
)


def create_agent_tools():
    """
    Create tools list for the agent
    """
    return [
        smart_query_preprocessing,  # NEW: Smart query preprocessing with history
        smart_query_preprocessing_with_history,  # NEW: Smart query preprocessing with explicit history
        rewrite_query_with_history,  # NEW: Query rewriting tool
        search_documents,
        refine_query,
        evaluate_documents,
        generate_answer,
        request_new_query,
    ]


def create_enhanced_agent_executor(
    tools: List, model_config: Dict[str, Any]
) -> AgentExecutor:
    """
    Create enhanced agent executor dengan konfigurasi khusus untuk multi-step RAG
    """

    # Enhanced system prompt dengan instruksi yang lebih spesifik
    system_prompt = """Anda adalah asisten hukum kesehatan Indonesia berbasis AI yang menggunakan pendekatan Enhanced Multi-Step RAG (Retrieval-Augmented Generation) untuk menjawab pertanyaan.

TUGAS ANDA:
1. Memahami pertanyaan pengguna tentang hukum kesehatan Indonesia 
2. Mencari dokumen yang relevan dengan pertanyaan (maksimal 3 dokumen)
3. Mengevaluasi apakah dokumen yang ditemukan memadai untuk menjawab pertanyaan
4. Jika evaluasi menunjukkan "KURANG MEMADAI", sempurnakan query HANYA SEKALI dan cari lagi
5. Setelah pencarian kedua, langsung hasilkan jawaban berdasarkan dokumen yang ada
6. JANGAN melakukan evaluasi kedua setelah penyempurnaan query

ATURAN WAJIB (MANDATORY RULES):
1. 🔍 STEP 1 - SEARCH: Selalu mulai dengan `search_documents` untuk mencari dokumen relevan dengan query yang spesifik
2. �� STEP 2 - EVALUATE: Panggil `evaluate_documents` dengan **SELURUH** `search_result['retrieved_docs_data']` (jangan subset)  
3. 🔧 STEP 3 - REFINE (jika perlu): Jika dokumen kurang memadai, gunakan `refine_query` SEKALI saja lalu `search_documents` lagi
4. ✨ STEP 4 - GENERATE: Gunakan `generate_answer` dengan parameter:
   • documents = search_result['retrieved_docs_data']  
   • evaluation_result = evaluation  (JSON lengkap dari previous step)  
   Tool akan otomatis memilih & mererank dokumen relevan (top 3)

ATURAN SELEKSI DOKUMEN:
- Hanya pilih dokumen yang BENAR-BENAR menjawab pertanyaan
- Prioritaskan dokumen dengan status "berlaku" daripada "dicabut"  
- Jangan sertakan dokumen yang hanya tangensial atau tidak langsung relevan
- Untuk generate_answer, berikan HANYA dokumen yang akan dikutip dalam jawaban

KRITERIA EVALUASI:
- MEMADAI: Dokumen mengandung informasi langsung yang menjawab pertanyaan, dengan detail yang cukup
- KURANG MEMADAI: Dokumen terlalu umum, tidak langsung menjawab, atau kurang detail

⚠️ INSTRUKSI PENTING UNTUK DATA PASSING:
WAJIB: Saat memanggil `evaluate_documents` dan `generate_answer`:
- Gunakan search_result['retrieved_docs_data'] (list of documents dengan metadata lengkap)
- JANGAN gunakan search_result['formatted_docs_for_llm'] (string format)
- Pastikan setiap dokumen memiliki: name, source, content, metadata

CONTOH BENAR:
```
search_result = search_documents(query="...")
evaluate_documents(query="...", documents=search_result['retrieved_docs_data'])
generate_answer(documents=search_result['retrieved_docs_data'], query="...")
```

❌ SALAH: menggunakan formatted_docs_for_llm atau string lainnya

LARANGAN:
❌ JANGAN evaluasi berulang-ulang
❌ JANGAN gunakan semua dokumen untuk generate_answer jika tidak relevan
❌ JANGAN lewatkan parameter query saat memanggil generate_answer
❌ JANGAN sertakan dokumen yang tidak akan dikutip dalam jawaban
❌ JANGAN kirim formatted string ke evaluate_documents/generate_answer
❌ WAJIB kirim structured list dari retrieved_docs_data

Jawab dengan profesional dalam Bahasa Indonesia, gunakan sitasi yang akurat, dan pastikan menjawab pertanyaan secara langsung."""

    # Create the chat prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("system", "{history_summary}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Create the ChatOpenAI model
    llm = ChatOpenAI(**model_config)

    # Create the agent
    agent = create_openai_tools_agent(llm, tools, prompt)

    # Create and return the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10,
        early_stopping_method="generate",
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    return agent_executor


def get_agent_executor():
    """
    Initialize and return agent executor
    """
    try:
        print("✅ Initializing RAG agent...")

        # Initialize LLM
        llm = ChatOpenAI(**MODELS["MAIN"])

        # Create tools list
        tools = create_agent_tools()

        # Create system prompt for agent
        system_prompt = """Anda adalah asisten hukum kesehatan Indonesia. Tugas Anda adalah menjawab pertanyaan dengan menggunakan tools yang tersedia.

WORKFLOW ENHANCED DENGAN QUERY REWRITING:
1. 🧠 PREPROCESS: Gunakan smart_query_preprocessing_with_history untuk menganalisis dan memperbaiki query berdasarkan history_summary
2. 🔍 SEARCH: Gunakan search_documents dengan query yang sudah diproses
3. 📊 EVALUATE: Panggil `evaluate_documents` dengan **SELURUH** `search_result['retrieved_docs_data']` (jangan subset)  
4. ✨ GENERATE: Gunakan generate_answer untuk membuat jawaban final

ATURAN PENTING:
- SELALU mulai dengan smart_query_preprocessing_with_history jika ada history_summary
- Gunakan processed_query dari preprocessing untuk search_documents
- Jika evaluasi menunjukkan "MEMADAI", langsung generate jawaban
- Jika evaluasi menunjukkan "KURANG MEMADAI", coba refine_query SEKALI saja, lalu search lagi, lalu generate
- JANGAN melakukan evaluasi berulang-ulang
- SELALU akhiri dengan generate_answer

INSTRUKSI DATA PASSING:
- Saat memanggil smart_query_preprocessing_with_history: sertakan current_query dan history_summary
- Saat memanggil search_documents: gunakan processed_query dari preprocessing
- Saat memanggil generate_answer: WAJIB sertakan parameter 'documents', 'query', dan 'evaluation_result'
- Format: generate_answer(documents=hasil_search, query=query_asli, evaluation_result=hasil_evaluasi)

CONTOH WORKFLOW:
1. preprocessing_result = smart_query_preprocessing_with_history(current_query="Jadi boleh/engga?", history_summary="{history_summary}", previous_responses={previous_responses})
2. search_result = search_documents(query=preprocessing_result['processed_query'])
3. evaluation = evaluate_documents(query="...", documents=search_result['retrieved_docs_data'])
4. answer = generate_answer(documents=search_result['retrieved_docs_data'], query=preprocessing_result['original_query'], evaluation_result=evaluation)

PENTING: 
- Gunakan {history_summary} dan {previous_responses} yang tersedia dalam context
- previous_responses berisi array 3 string dengan format "Question: X\nAnswer: Y"

Jawab dalam Bahasa Indonesia dengan sitasi yang akurat."""

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("system", "History Summary: {history_summary}"),
                ("system", "Previous Responses: {previous_responses}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        # Create OpenAI tools agent
        agent = create_openai_tools_agent(llm, tools, prompt)

        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,  # Enable verbose logging untuk debugging
            handle_parsing_errors=True,
            return_intermediate_steps=True,
            max_execution_time=120,
            max_iterations=10,
        )

        print("✅ RAG agent initialized successfully")
        return agent_executor

    except Exception as agent_error:
        print(f"⚠️ Agent initialization issue: {agent_error}")
        return None

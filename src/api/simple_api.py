# import basics
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKey, APIKeyHeader
from pydantic import BaseModel, Field
import uvicorn
from typing import List, Dict, Any, Optional, Literal
import secrets
from starlette.status import HTTP_403_FORBIDDEN
import time
from operator import itemgetter  # NEW: untuk ekstrak field dari dict

# RAG components
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Supabase
from supabase.client import Client, create_client

# Load environment variables
load_dotenv()

# Modern RAG Observability (No Cache - Simple API)
try:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from observability.rag_tracker import rag_tracker, APIType, ExecutionMode

    LANGFUSE_ENABLED = rag_tracker.enabled
    print(f"✅ RAG Tracker: {'enabled' if LANGFUSE_ENABLED else 'disabled'}")
    print(f"ℹ️  Cache: disabled (Simple API doesn't use cache)")
except ImportError as e:
    print(f"⚠️  RAG Observability not available: {e}")
    LANGFUSE_ENABLED = False

# Security settings - Simplifikasi keamanan
API_KEY_NAME = "X-API-Key"
API_KEY = os.environ.get("API_KEY", secrets.token_urlsafe(32))

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


# Security dependency - verifikasi API key saja
async def verify_api_key(
    api_key_header: str = Depends(api_key_header),
):
    if api_key_header != API_KEY:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Akses ditolak: API key tidak valid"
        )
    return api_key_header


# Initialize FastAPI app
app = FastAPI(
    title="Pure RAG API - Sistem Dokumen Hukum",
    description="API Pure RAG untuk sistem dokumentasi hukum Indonesia (tanpa history processing)",
    version="1.0.0",
)

# Add CORS middleware - hanya izinkan backend yang terdaftar
backend_url = os.environ.get("BACKEND_URL")
frontend_url = os.environ.get("FRONTEND_URL")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[backend_url, frontend_url],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Initialize Supabase database
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")

supabase: Client = create_client(supabase_url, supabase_key)

# Model embedding configuration
# Mapping tabel dan model embedding
EMBEDDING_CONFIG = {
    "small": {"model": "text-embedding-3-small", "table": "documents_small"},
    "large": {"model": "text-embedding-3-large", "table": "documents"},
}

# Initialize LLM
from .utils.config_manager import MODELS

llm = ChatOpenAI(**{**MODELS["MAIN"], "streaming": True, "temperature": 0})

# Create the custom prompt
custom_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """# PERAN DAN PEMBATASAN (Framework COSTAR)

## Context
Anda adalah asisten AI untuk dokumen-dokumen hukum kesehatan di Indonesia. Anda dirancang untuk membantu pengguna menemukan informasi terkait regulasi kesehatan, peraturan, dan ketentuan hukum di bidang kesehatan Indonesia. Anda hanya menjawab pertanyaan berdasarkan dokumen hukum kesehatan resmi yang tersedia dalam database.

## Objective
Tujuan utama Anda adalah memberikan informasi faktual, akurat dan komprehensif dari dokumen hukum kesehatan Indonesia, membantu pengguna memahami aturan, prosedur, ketentuan, dan informasi legal dalam bidang kesehatan.

## Scope
Anda hanya menjawab pertanyaan dalam ruang lingkup:
- Undang-Undang dan peraturan kesehatan Indonesia
- Keputusan terkait kebijakan kesehatan
- Prosedur hukum dan administratif kesehatan
- Definisi hukum dan penjelasan istilah legal kesehatan
- Hak dan kewajiban dalam bidang kesehatan menurut hukum Indonesia
- Interpretasi hukum kesehatan berdasarkan dokumen resmi

Anda TIDAK akan menjawab:
- Pertanyaan di luar konteks dokumen yang tersedia
- Memberikan nasihat medis atau diagnosa (Anda bukan dokter)
- Memberikan nasihat hukum personal (Anda bukan pengacara)
- Rekomendasi pengobatan atau prosedur medis spesifik
- Hal-hal terkait SARA (Suku, Agama, Ras, dan Antar-golongan)
- Pertanyaan politik yang memecah belah atau isu kontroversial
- Permintaan untuk melakukan tindakan ilegal atau tidak etis
- Memprediksi hasil kasus hukum spesifik

## Tone
Gunakan Bahasa Indonesia yang formal, jelas, presisi dan sesuai dengan terminologi hukum kesehatan. Jawaban Anda harus sistematis, terstruktur, dan mengikuti hierarki peraturan perundang-undangan Indonesia.

## Authority
Otoritas Anda berasal dari dokumen resmi dalam database. Jangan memberikan pendapat pribadi atau spekulasi. Jika informasi tidak tersedia dalam dokumen, jujurlah bahwa Anda tidak memiliki informasi tersebut.

## Role
Anda adalah asisten informasi dokumen hukum kesehatan, bukan penasihat medis atau hukum. Anda membantu menemukan dan menjelaskan informasi, bukan memberikan saran medis, diagnosa, atau interpretasi yang melampaui isi dokumen.

# INSTRUKSI KHUSUS UNTUK DOKUMEN HUKUM KESEHATAN
1. Jawaban harus mengutamakan hierarki peraturan perundang-undangan Indonesia: UUD 1945 → UU/Perpu → PP → Perpres → Permenkes → Perda
2. Saat mengutip peraturan, gunakan format baku: "Pasal X ayat (Y) [Nama Peraturan]"
3. Jika ada peraturan yang bertentangan, terapkan asas:
   - Lex superior derogat legi inferiori (hukum yang lebih tinggi mengalahkan yang lebih rendah)
   - Lex specialis derogat legi generali (hukum khusus mengalahkan hukum umum)
   - Lex posterior derogat legi priori (hukum baru mengalahkan hukum lama)
4. Berikan definisi istilah hukum kesehatan yang penting jika relevan dengan pertanyaan

# PENANGANAN STATUS PERATURAN (BERLAKU vs DICABUT)
1. Konteks akan berisi dokumen dari peraturan dengan status "berlaku" dan "dicabut"
2. PRIORITASKAN informasi dari dokumen berstatus "berlaku" dalam memberikan jawaban
3. Jika ada kontradiksi antara dokumen "berlaku" dan "dicabut", gunakan dokumen yang "berlaku"
4. Anda BOLEH mereferensikan dokumen yang sudah "dicabut" untuk:
   - Memberikan konteks historis perkembangan regulasi
   - Menjelaskan perubahan regulasi dari waktu ke waktu
   - Membandingkan peraturan lama dan baru bila relevan
5. SELALU sebutkan status peraturan saat Anda merujuk ke suatu dokumen, misalnya:
   - "Berdasarkan PP No. 28 Tahun 2024 (berlaku) Pasal 32, ..."
   - "Menurut UU No. 36 Tahun 2009 (dicabut) pada saat itu mengatur bahwa..."
6. Prioritaskan menyimpulkan berdasarkan peraturan terbaru yang masih berlaku

# INSTRUKSI PENTING
1. Jika pertanyaan di luar ruang lingkup atau tidak pantas, tolak dengan sopan dan sarankan untuk mengajukan pertanyaan terkait dokumen hukum kesehatan Indonesia.
2. Jangan menyertakan URL, referensi internal, atau nomor halaman dalam jawaban Anda.
3. SELALU mulai jawaban Anda dengan menyebutkan sumber peraturan yang relevan, termasuk statusnya, misalnya: "Berdasarkan PP No. 28 Tahun 2024 (berlaku) Pasal 32, ..." atau "Menurut UU No. 36 Tahun 2009 (dicabut) Pasal 128, ..."
4. Jika menggunakan beberapa sumber, sebutkan sumber-sumber utama di awal jawaban Anda, misalnya: "Berdasarkan PP No. 28 Tahun 2024 (berlaku) Pasal 24, 28, dan 32, serta UU No. 36 Tahun 2009 (dicabut) Pasal 128, ..."
5. Jangan pernah berhalusinasi atau membuat informasi yang tidak ada dalam dokumen.
6. Jika Anda tidak yakin, katakan bahwa informasi tersebut tidak ditemukan dalam dokumen yang tersedia.
7. JANGAN memberikan saran medis, diagnosa penyakit, atau rekomendasi pengobatan.

# INSTRUKSI FORMATTING (WAJIB UNTUK READABILITY):
- Gunakan **bold formatting** untuk kata-kata PENTING seperti: **nama peraturan**, **pasal**, **definisi kunci**, **status hukum**, **sanksi**, **kewajiban**, **persyaratan**, dll
- Bold untuk **angka/nominal** penting, **tanggal**, **batas waktu**, **prosedur**, **kriteria**
- Bold untuk **istilah teknis** dan **konsep hukum** yang penting
- Bold untuk **nama institusi**, **wewenang**, **tanggung jawab**
- Contoh formatting: "Berdasarkan **PP No. 28 Tahun 2024 (berlaku)** **Pasal 32**, **definisi kesehatan** adalah..."
- Contoh: "**Sanksi pidana** berupa **denda maksimal Rp 100 juta** atau **kurungan 1 tahun**"

Jawaban Anda harus faktual, akurat, dan hanya berdasarkan pada dokumen yang tersedia.""",
        ),
        ("human", "{question}"),
        (
            "system",
            "Berikut adalah dokumen-dokumen yang relevan dengan pertanyaan:\n\n{context}",
        ),
    ]
)


# RAG utility functions
def get_embeddings(embedding_model="large"):
    """Dapatkan objek embedding berdasarkan model yang dipilih"""
    if embedding_model not in EMBEDDING_CONFIG:
        raise ValueError(f"Model embedding tidak valid: {embedding_model}")

    model_name = EMBEDDING_CONFIG[embedding_model]["model"]
    return OpenAIEmbeddings(model=model_name)


def get_vector_store(embedding_model="large"):
    """Initialize vector store dengan model embedding yang dipilih"""
    if embedding_model not in EMBEDDING_CONFIG:
        raise ValueError(f"Model embedding tidak valid: {embedding_model}")

    # Dapatkan embeddings model
    embeddings = get_embeddings(embedding_model)
    print(
        f"[DEBUG] Initialized embeddings model: {EMBEDDING_CONFIG[embedding_model]['model']}"
    )

    # Dapatkan nama tabel sesuai model
    table_name = EMBEDDING_CONFIG[embedding_model]["table"]

    # Gunakan nama fungsi pencarian yang sesuai berdasarkan model
    if embedding_model == "small":
        query_name = "match_documents_small"
    else:
        query_name = "match_documents"

    print(f"[DEBUG] Initializing vector store with:")
    print(f"[DEBUG] - Table name: {table_name}")
    print(f"[DEBUG] - Query function: {query_name}")
    print(f"[DEBUG] - Supabase URL: {supabase_url}")
    print(
        f"[DEBUG] - Supabase key length: {len(supabase_key) if supabase_key else 'None'}"
    )

    try:
        # Buat vector store
        vector_store = SupabaseVectorStore(
            embedding=embeddings,
            client=supabase,
            table_name=table_name,
            query_name=query_name,
        )
        print("[DEBUG] Vector store initialized successfully")
        return vector_store
    except Exception as e:
        print(f"[ERROR] Failed to initialize vector store: {str(e)}")
        raise


def format_docs(docs):
    """Format retrieved documents for context dengan menambahkan metadata yang lebih lengkap"""
    formatted_docs = []

    for i, doc in enumerate(docs):
        # Ekstrak metadata penting
        jenis_peraturan = doc.metadata.get("jenis_peraturan", "")
        nomor_peraturan = doc.metadata.get("nomor_peraturan", "")
        tahun_peraturan = doc.metadata.get("tahun_peraturan", "")
        tipe_bagian = doc.metadata.get("tipe_bagian", "")
        status = doc.metadata.get("status", "berlaku")
        nomor_halaman = doc.metadata.get("nomor_halaman", "")

        # Buat header dokumen yang informatif untuk membantu model mengetahui sumbernya
        doc_header = f"Dokumen #{i+1}"

        if jenis_peraturan and nomor_peraturan and tahun_peraturan:
            doc_header += (
                f" ({jenis_peraturan} No. {nomor_peraturan} Tahun {tahun_peraturan}"
            )
            if tipe_bagian:
                doc_header += f" {tipe_bagian}"
            # Tambahkan status dan nomor halaman di header dokumen
            doc_header += f", Status: {status}"
            if nomor_halaman:
                doc_header += f", Hal: {nomor_halaman}"
            doc_header += ")"
        else:
            doc_header += ")"

        # Format dokumen dengan header yang lebih informatif
        formatted_docs.append(f"{doc_header}:\n{doc.page_content}\n")

    return "\n\n".join(formatted_docs)


def summarize_pairs(pairs: List[str]) -> str:
    """Ringkas max 3 pasangan Q-A (>2? diambil 3 terakhir)."""
    if not pairs:
        return ""
    limited = pairs[-3:]  # ambil 3 terakhir
    joined = "\n\n".join(limited)
    return small_llm.invoke(
        summarizer_prompt.format_prompt(chat_pairs=joined)
    ).content.strip()


def find_document_links(doc_names, embedding_model="large"):
    """Find document links based on document names extracted from retrieved docs"""
    print(f"\n[DEBUG] Searching for document links for: {doc_names}")
    print(f"[DEBUG] Using embedding model: {embedding_model}")
    document_links = []

    # Get embeddings model dengan model yang sesuai
    embeddings = get_embeddings(embedding_model)

    # Gunakan nama fungsi pencarian yang benar berdasarkan model
    if embedding_model == "small":
        search_function = "match_document_links_small"
    else:
        search_function = "match_document_links"

    print(f"[DEBUG] Using search function: {search_function}")

    for doc_name in doc_names:
        try:
            # Generate embedding dari nama dokumen
            query_embedding = embeddings.embed_query(doc_name)

            # Menggunakan fungsi yang sesuai berdasarkan model
            result = supabase.rpc(
                search_function, {"query_embedding": query_embedding}
            ).execute()

            # Ambil dokumen paling mirip (hasil pertama)
            if result.data:
                most_similar = result.data[0]
                doc_name = most_similar.get("document_name", "")
                doc_link = most_similar.get("document_link", "")

                # Skip jika tidak ada link
                if not doc_link:
                    continue

                # Cek apakah link sudah ada (hindari duplikat)
                already_exists = any(
                    existing["document_link"] == doc_link for existing in document_links
                )

                if not already_exists:
                    link_info = {
                        "document_name": doc_name,
                        "document_link": doc_link,
                    }
                    document_links.append(link_info)

        except Exception as e:
            print(f"[DEBUG] Error finding link for {doc_name}: {str(e)}")

    return document_links


# Define request and response models
class ChatRequest(BaseModel):
    query: str
    embedding_model: Literal["small", "large"] = "large"
    # previous_responses: List[Any] = []  # DIHAPUS: Pure RAG tanpa history


class ChatResponse(BaseModel):
    answer: str
    model_info: Dict[str, Any] = {}  # Ubah dari str ke Any untuk bisa menyimpan angka
    referenced_documents: List[Dict[str, Any]] = []
    processing_time_ms: Optional[int] = None


# Fungsi untuk mengekstrak informasi dokumen dari hasil retrieval
def extract_document_info(docs):
    """Ekstrak informasi dokumen dari hasil retrieval"""
    document_info = []

    print("[DEBUG] Starting document info extraction")
    print(f"[DEBUG] Number of documents: {len(docs)}")

    for i, doc in enumerate(docs):
        try:
            print(f"\n[DEBUG] Processing document #{i+1}")
            print(f"[DEBUG] Raw metadata: {doc.metadata}")
            print(f"[DEBUG] Document type: {type(doc)}")

            # Pastikan metadata adalah dictionary
            metadata = doc.metadata if isinstance(doc.metadata, dict) else {}

            # Data dari metadata dengan default values
            jenis_peraturan = metadata.get("jenis_peraturan", "Tidak Diketahui")
            nomor_peraturan = metadata.get("nomor_peraturan", "-")
            tahun_peraturan = metadata.get("tahun_peraturan", "-")
            tipe_bagian = metadata.get("tipe_bagian", "")
            status = metadata.get("status", "berlaku")
            nomor_halaman = metadata.get("nomor_halaman", "")

            # Buat nama dokumen urutan
            doc_name = f"Dokumen #{i+1}"

            # Buat deskripsi dokumen
            if jenis_peraturan != "Tidak Diketahui" and nomor_peraturan != "-":
                doc_description = (
                    f"{jenis_peraturan} No. {nomor_peraturan} Tahun {tahun_peraturan}"
                )
                if tipe_bagian:
                    doc_description += f" {tipe_bagian}"
                doc_description += f" (Status: {status}"
                if nomor_halaman:
                    doc_description += f", Hal: {nomor_halaman}"
                doc_description += ")"
            else:
                # Coba ekstrak informasi dari content jika metadata kosong
                content = doc.page_content if hasattr(doc, "page_content") else ""
                # Coba ekstrak informasi dari konten menggunakan regex
                import re

                peraturan_match = re.search(
                    r"(UU|PP|Perpres|Permenkes)\s+No[mor]?\.*\s+(\d+)\s+Tahun\s+(\d{4})",
                    content,
                    re.IGNORECASE,
                )
                if peraturan_match:
                    jenis = peraturan_match.group(1)
                    nomor = peraturan_match.group(2)
                    tahun = peraturan_match.group(3)
                    doc_description = (
                        f"{jenis} No. {nomor} Tahun {tahun} (Status: {status})"
                    )
                else:
                    doc_description = f"{doc_name} (Detail tidak tersedia)"

            # Buat source yang konsisten
            doc_source = f"{jenis_peraturan} No. {nomor_peraturan}/{tahun_peraturan} (Status: {status}"
            if nomor_halaman:
                doc_source += f", Hal: {nomor_halaman}"
            doc_source += ")"
            if doc_source.startswith("Tidak Diketahui"):
                doc_source = doc_description

            print(f"[DEBUG] Processed document info: {doc_description}")

            document_info.append(
                {
                    "name": doc_name,
                    "source": doc_source,
                    "content": doc.page_content if hasattr(doc, "page_content") else "",
                    "metadata": metadata,
                }
            )

        except Exception as e:
            print(f"[ERROR] Error processing document #{i+1}: {str(e)}")
            # Tambahkan dokumen dengan informasi minimal
            document_info.append(
                {
                    "name": f"Dokumen #{i+1}",
                    "description": "Error saat memproses dokumen",
                    "source": "Tidak tersedia",
                    "content": getattr(doc, "page_content", ""),
                    "metadata": {},
                }
            )

    print(f"[DEBUG] Finished processing {len(document_info)} documents")
    return document_info


# DIHAPUS: Pure RAG tanpa history processing
# format_chat_history() dan to_str_history() tidak diperlukan


# Create Pure RAG chain (tanpa history)
def create_rag_chain(embedding_model="large"):
    """Create a pure RAG chain using the specified model (no history processing)"""
    print(f"\n[DEBUG] Creating pure RAG chain with model: {embedding_model}")

    try:
        # Get vector store for the specified model
        vector_store = get_vector_store(embedding_model)
        print("[DEBUG] Vector store created successfully")

        # Create retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        print("[DEBUG] Retriever created successfully")

        # Create PURE RAG chain - langsung query → retrieval → generation
        rag_chain = (
            {
                "context": retriever | format_docs,  # langsung dari query
                "question": RunnablePassthrough(),  # query asli
            }
            | custom_prompt
            | llm
            | StrOutputParser()
        )
        print("[DEBUG] Pure RAG chain created successfully")

        return rag_chain, retriever, EMBEDDING_CONFIG[embedding_model]["model"]
    except Exception as e:
        print(f"[ERROR] Failed to create pure RAG chain: {str(e)}")
        raise


# API endpoint for chat
@app.post(
    "/api/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)]
)
async def chat(request: ChatRequest):
    print("\n" + "=" * 50)
    print("[DEBUG] Starting new chat request")
    print("=" * 50)

    try:
        start_time = time.time()  # Tambahkan waktu mulai

        # Note: Simple API doesn't use cache - always processes fresh requests

        # Start RAG tracking session
        trace = None
        if LANGFUSE_ENABLED:
            trace = rag_tracker.start_session(
                query=request.query,
                api_type=APIType.SIMPLE,
                execution_mode=ExecutionMode.STANDARD,
                metadata={"embedding_model": request.embedding_model},
            )

        print(f"[DEBUG] Request details:")
        print(f"- Query: {request.query}")
        print(f"- Embedding model: {request.embedding_model}")
        print(f"- Pure RAG: No history processing")
        print(f"- RAG Tracker: {'enabled' if trace else 'disabled'}")

        # Validasi model
        if request.embedding_model not in EMBEDDING_CONFIG:
            raise HTTPException(
                status_code=400,
                detail=f"Model embedding tidak valid. Harus 'small' atau 'large'",
            )

        # Dapatkan nama tabel dan model
        table_name = EMBEDDING_CONFIG[request.embedding_model]["table"]
        model_name = EMBEDDING_CONFIG[request.embedding_model]["model"]
        print(f"[DEBUG] Configuration:")
        print(f"- Table name: {table_name}")
        print(f"- Model name: {model_name}")

        # Cek dimensi embedding
        try:
            embeddings = get_embeddings(request.embedding_model)
            sample_embedding = embeddings.embed_query("Test query for dimension check")
            embedding_dimensions = len(sample_embedding)
            print(f"[DEBUG] Embedding test successful:")
            print(f"- Dimensions: {embedding_dimensions}")
        except Exception as e:
            print(f"[ERROR] Embedding test failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error saat menginisialisasi model embedding: {str(e)}",
            )

        # Create RAG chain
        try:
            rag_chain, retriever, model_name = create_rag_chain(request.embedding_model)
            print("[DEBUG] RAG chain created successfully")
        except Exception as e:
            print(f"[ERROR] Failed to create RAG chain: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error saat membuat RAG chain: {str(e)}"
            )

        # PURE RAG: Langsung gunakan query tanpa history processing
        # Tidak ada kombinasi dengan history atau previous responses

        # Get relevant documents using direct query
        try:
            docs = retriever.invoke(request.query)
            print(f"[DEBUG] Retrieved {len(docs)} relevant documents")

            # Track document retrieval
            if LANGFUSE_ENABLED and trace:
                rag_tracker.track_document_retrieval(
                    trace=trace,
                    query=request.query,
                    embedding_model=model_name,
                    num_docs=len(docs),
                    api_type=APIType.SIMPLE,
                    docs=[
                        (
                            doc.page_content[:200] + "..."
                            if len(doc.page_content) > 200
                            else doc.page_content
                        )
                        for doc in docs[:2]
                    ],
                )

            if docs:
                print("[DEBUG] Sample document info:")
                sample_doc = docs[0]
                print(f"- Type: {type(sample_doc)}")
                print(f"- Has metadata: {'metadata' in dir(sample_doc)}")
                if hasattr(sample_doc, "metadata"):
                    print(f"- Metadata type: {type(sample_doc.metadata)}")
                    print(
                        f"- Metadata keys: {sample_doc.metadata.keys() if isinstance(sample_doc.metadata, dict) else 'Not a dict'}"
                    )
        except Exception as e:
            print(f"[ERROR] Document retrieval failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error saat mengambil dokumen relevan: {str(e)}",
            )

        # Extract document information
        try:
            referenced_documents = extract_document_info(docs)
            print(f"[DEBUG] Extracted info from {len(referenced_documents)} documents")
        except Exception as e:
            print(f"[ERROR] Document info extraction failed: {str(e)}")
            referenced_documents = []  # Fallback to empty list

        # Generate answer using pure RAG (tanpa history)
        try:
            # PURE RAG: Langsung invoke dengan query saja
            answer = rag_chain.invoke(request.query)
            print("[DEBUG] Successfully generated answer using pure RAG")

            # Track LLM call (approximate token usage)
            estimated_input_tokens = (
                len(f"{request.query} {format_docs(docs)}".split()) * 1.3
            )
            estimated_output_tokens = len(answer.split()) * 1.3

            if LANGFUSE_ENABLED and trace:
                rag_tracker.track_llm_generation(
                    trace=trace,
                    model=MODELS["MAIN"]["model"],
                    input_messages=[{"role": "user", "content": request.query}],
                    response=answer,
                    api_type=APIType.SIMPLE,
                    usage={
                        "prompt_tokens": int(estimated_input_tokens),
                        "completion_tokens": int(estimated_output_tokens),
                        "total_tokens": int(
                            estimated_input_tokens + estimated_output_tokens
                        ),
                    },
                )

        except Exception as e:
            print(f"[ERROR] Answer generation failed: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error saat menghasilkan jawaban: {str(e)}"
            )

        # Prepare response
        model_info = {
            "model": model_name,
            "table": table_name,
            "dimensions": embedding_dimensions,
            "type": "pure_rag",  # Menandakan ini adalah Pure RAG
            "history_processing": False,  # Tidak ada history processing
        }

        # Hitung waktu pemrosesan
        end_time = time.time()
        processing_time_ms = int((end_time - start_time) * 1000)

        # Calculate estimated cost
        estimated_cost = (
            rag_tracker._calculate_cost(
                MODELS["MAIN"]["model"],
                {
                    "prompt_tokens": int(estimated_input_tokens),
                    "completion_tokens": int(estimated_output_tokens),
                    "total_tokens": int(
                        estimated_input_tokens + estimated_output_tokens
                    ),
                },
            )
            if LANGFUSE_ENABLED
            else 0.0
        )

        # Finalize RAG tracking
        if LANGFUSE_ENABLED and trace:
            rag_tracker.finalize_session(
                trace=trace,
                final_answer=answer,
                api_type=APIType.SIMPLE,
                execution_mode=ExecutionMode.STANDARD,
                processing_time_ms=processing_time_ms,
                estimated_cost=estimated_cost,
            )

        # Note: Simple API doesn't cache responses - always fresh processing

        print("[DEBUG] Request completed successfully")
        print("=" * 50 + "\n")

        return ChatResponse(
            answer=answer,
            model_info=model_info,
            referenced_documents=referenced_documents,
            processing_time_ms=processing_time_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Unexpected error in chat endpoint: {str(e)}")
        import traceback

        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error internal server: {str(e)}")


# Health check endpoint - No authentication needed
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/monitoring/health")
async def monitoring_health():
    """Basic health check endpoint for monitoring"""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/monitoring/health/detailed")
async def monitoring_health_detailed():
    """Detailed health check endpoint for external monitoring systems"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "api": "operational",
            "cache": "disabled",  # Simple API doesn't use cache
            "database": "operational",  # Assuming Pinecone is working if API works
        },
        "system": "Simple RAG",
        "uptime": time.time(),  # Simple uptime indicator
    }


# Documentation endpoint for available models and collections
@app.get("/api/models", dependencies=[Depends(verify_api_key)])
async def available_models():
    return {
        "embedding_models": {
            model_key: {
                "model_name": config["model"],
                "table": config["table"],
            }
            for model_key, config in EMBEDDING_CONFIG.items()
        }
    }


# RAG observability endpoint
@app.get("/api/observability")
async def get_observability_status():
    """Get RAG observability and cache status"""
    from cache.smart_cache import get_cache_stats

    response = {
        "tracking": (
            rag_tracker.get_status() if LANGFUSE_ENABLED else {"enabled": False}
        ),
        "cache": {
            "enabled": False,
            "reason": "Simple API designed for direct processing without cache",
        },
        "api_type": "simple_api",
        "features": [
            "Direct processing (no cache)",
            "Cost tracking per request",
            "Performance monitoring",
            "API-specific tracing",
            "Fast response times",
        ],
    }

    if LANGFUSE_ENABLED:
        response["tracking"].update(
            {
                "setup_instructions": "Visit https://cloud.langfuse.com to see your dashboard",
                "metrics_tracked": {
                    "cost_per_request": "USD cost for LLM calls",
                    "token_usage": "Input/output tokens for each request",
                    "document_retrieval": "Number of documents retrieved",
                    "response_time": "End-to-end processing time",
                    "embedding_model": "Which embedding model used",
                },
            }
        )
    else:
        response["setup_instructions"] = [
            "1. Get free account at https://cloud.langfuse.com",
            "2. Create new project and get API keys",
            "3. Set environment variables:",
            "   - LANGFUSE_PUBLIC_KEY=pk-...",
            "   - LANGFUSE_SECRET_KEY=sk-...",
            "4. Restart application",
        ]

    return response


# Run the API server
if __name__ == "__main__":
    # Print API key untuk setup
    print(f"API Key: {API_KEY}")
    uvicorn.run("simple_api:app", host="0.0.0.0", port=8000, reload=True)

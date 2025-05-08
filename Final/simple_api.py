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
    title="Sistem RAG Dokumen Hukum",
    description="API sederhana untuk sistem RAG dokumen hukum Indonesia",
    version="1.0.0",
)

# Add CORS middleware - hanya izinkan backend yang terdaftar
backend_url = os.environ.get("BACKEND_URL", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[backend_url],
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
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

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

# INSTRUKSI PENTING
1. Jika pertanyaan di luar ruang lingkup atau tidak pantas, tolak dengan sopan dan sarankan untuk mengajukan pertanyaan terkait dokumen hukum kesehatan Indonesia.
2. Jangan menyertakan URL, referensi internal, atau nomor halaman dalam jawaban Anda.
3. SELALU mulai jawaban Anda dengan menyebutkan sumber peraturan yang relevan, misalnya: "Berdasarkan PP No. 28 Tahun 2024 Pasal 32, ..." atau "Menurut UU No. 36 Tahun 2009 Pasal 128, ..."
4. Jika menggunakan beberapa sumber, sebutkan sumber-sumber utama di awal jawaban Anda, misalnya: "Berdasarkan PP No. 28 Tahun 2024 Pasal 24, 28, dan 32, serta UU No. 36 Tahun 2009 Pasal 128, ..."
5. Jangan pernah berhalusinasi atau membuat informasi yang tidak ada dalam dokumen.
6. Jika Anda tidak yakin, katakan bahwa informasi tersebut tidak ditemukan dalam dokumen yang tersedia.
7. JANGAN memberikan saran medis, diagnosa penyakit, atau rekomendasi pengobatan.

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

    # Dapatkan nama tabel sesuai model
    table_name = EMBEDDING_CONFIG[embedding_model]["table"]

    # Buat vector store
    return SupabaseVectorStore(
        embedding=embeddings,
        client=supabase,
        table_name=table_name,
        query_name="match_documents",
    )


def format_docs(docs):
    """Format retrieved documents for context dengan menambahkan metadata yang lebih lengkap"""
    formatted_docs = []

    for i, doc in enumerate(docs):
        # Ekstrak metadata penting
        jenis_peraturan = doc.metadata.get("jenis_peraturan", "")
        nomor_peraturan = doc.metadata.get("nomor_peraturan", "")
        tahun_peraturan = doc.metadata.get("tahun_peraturan", "")
        tipe_bagian = doc.metadata.get("tipe_bagian", "")

        # Buat header dokumen yang informatif untuk membantu model mengetahui sumbernya
        doc_header = f"Dokumen #{i+1}"

        if jenis_peraturan and nomor_peraturan and tahun_peraturan:
            doc_header += (
                f" ({jenis_peraturan} No. {nomor_peraturan} Tahun {tahun_peraturan}"
            )
            if tipe_bagian:
                doc_header += f" {tipe_bagian}"
            doc_header += ")"

        # Format dokumen dengan header yang lebih informatif
        formatted_docs.append(f"{doc_header}:\n{doc.page_content}\n")

    return "\n\n".join(formatted_docs)


def find_document_links(doc_names, embedding_model="large"):
    """Find document links based on document names extracted from retrieved docs"""
    print(f"\n[DEBUG] Searching for document links for: {doc_names}")
    document_links = []

    # Get embeddings model
    embeddings = get_embeddings(embedding_model)

    for doc_name in doc_names:
        try:
            # Generate embedding dari nama dokumen
            query_embedding = embeddings.embed_query(doc_name)

            # Menggunakan fungsi match_document_links untuk mendapatkan dokumen paling mirip
            result = supabase.rpc(
                "match_document_links", {"query_embedding": query_embedding}
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
    embedding_model: Literal["small", "large"] = (
        "large"  # Model embedding yang digunakan
    )
    conversation_id: Optional[str] = None
    previous_responses: List[str] = []  # Menambahkan riwayat chat sebelumnya


class ChatResponse(BaseModel):
    answer: str
    document_links: List[Dict[str, Any]] = []
    model_info: Dict[str, str] = {}
    referenced_documents: List[Dict[str, Any]] = (
        []
    )  # Menambahkan informasi dokumen yang direferensikan


# Fungsi untuk mengekstrak informasi dokumen dari hasil retrieval
def extract_document_info(docs):
    """Ekstrak informasi dokumen dari hasil retrieval"""
    document_info = []
    for i, doc in enumerate(docs):
        # Debug: print seluruh metadata untuk mengetahui key apa saja yang tersedia
        print(f"[DEBUG] Document #{i} metadata: {doc.metadata}")

        # Coba ambil informasi dari berbagai kemungkinan metadata key
        source = doc.metadata.get("source", None)
        filename = doc.metadata.get("filename", None)
        title = doc.metadata.get("title", None)
        path = doc.metadata.get("path", None)
        document_id = doc.metadata.get("id", None)

        # Data dari metadata tambahan yang mungkin berguna untuk nama dokumen
        jenis_peraturan = doc.metadata.get("jenis_peraturan", "")
        nomor_peraturan = doc.metadata.get("nomor_peraturan", "")
        tahun_peraturan = doc.metadata.get("tahun_peraturan", "")
        tipe_bagian = doc.metadata.get("tipe_bagian", "")
        judul_peraturan = doc.metadata.get("judul_peraturan", "")

        # Buat nama dokumen urutan (sesuai permintaan)
        doc_name = f"Dokumen #{i+1}"

        # Buat nama deskriptif untuk informasi tambahan
        if jenis_peraturan and nomor_peraturan and tahun_peraturan:
            doc_description = (
                f"{jenis_peraturan} No. {nomor_peraturan} Tahun {tahun_peraturan}"
            )
            if tipe_bagian:
                doc_description += f" {tipe_bagian}"
        elif title:
            doc_description = title
        elif filename:
            doc_description = filename
        elif source:
            # Ekstrak nama file dari path jika ada
            doc_description = source.split("\\")[-1] if "\\" in source else source
        elif path:
            doc_description = path.split("/")[-1]
        else:
            # Fallback ke nama dokumen urutan
            doc_description = doc_name

        # Buat source yang bermakna jika metadata standar tidak tersedia
        if source or path:
            doc_source = source or path
        else:
            # Buat source deskriptif dari metadata yang tersedia
            doc_source = (
                f"{jenis_peraturan} No. {nomor_peraturan}/{tahun_peraturan}"
                if jenis_peraturan and nomor_peraturan and tahun_peraturan
                else "Metadata tidak lengkap"
            )

        # Tambahkan metadata lain yang mungkin berguna (semua metadata)
        additional_metadata = {}
        for key, value in doc.metadata.items():
            additional_metadata[key] = value

        document_info.append(
            {
                "name": doc_name,
                "description": doc_description,
                "source": doc_source,
                "content": doc.page_content,  # Kembalikan seluruh konten
                "metadata": additional_metadata,
            }
        )

    return document_info


# Fungsi untuk format chat history menjadi prompt
def format_chat_history(previous_responses, query):
    """Format riwayat chat untuk konteks tambahan"""
    if not previous_responses:
        return ""

    # Batasi jumlah respons sebelumnya yang digunakan (misal: 3 terakhir)
    limited_responses = (
        previous_responses[-3:] if len(previous_responses) > 3 else previous_responses
    )

    history = "\n\nBerikut adalah riwayat percakapan sebelumnya:\n\n"
    for i, response in enumerate(limited_responses):
        history += f"Respons sebelumnya #{i+1}:\n{response}\n\n"

    history += f"Pertanyaan saat ini: {query}"
    return history


# Create RAG chain
def create_rag_chain(embedding_model="large"):
    """Create a RAG chain using the specified model"""
    # Get vector store for the specified model
    vector_store = get_vector_store(embedding_model)

    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # Create RAG chain dengan pemrosesan context yang diubah
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever, EMBEDDING_CONFIG[embedding_model]["model"]


# API endpoint for chat
@app.post(
    "/api/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)]
)
async def chat(request: ChatRequest):
    try:
        print(f"\n[DEBUG] Processing chat request: {request.query}")
        print(f"[DEBUG] Using model: {request.embedding_model}")
        print(f"[DEBUG] Previous responses: {len(request.previous_responses)}")

        # Validasi model
        if request.embedding_model not in EMBEDDING_CONFIG:
            raise HTTPException(
                status_code=400,
                detail=f"Model embedding tidak valid. Harus 'small' atau 'large'",
            )

        # Dapatkan nama tabel
        table_name = EMBEDDING_CONFIG[request.embedding_model]["table"]
        print(f"[DEBUG] Using table: {table_name}")

        # Create RAG chain for the specified model
        rag_chain, retriever, model_name = create_rag_chain(request.embedding_model)

        # Get documents for finding links later
        docs = retriever.get_relevant_documents(request.query)

        # Debug: Print sample document untuk analisis
        if docs:
            print(f"[DEBUG] Retrieved {len(docs)} documents")
            sample_doc = docs[0]
            print(f"[DEBUG] Sample document metadata: {sample_doc.metadata}")
            print(
                f"[DEBUG] Sample document content (first 100 chars): {sample_doc.page_content[:100]}"
            )

        # Ekstrak informasi dokumen untuk response
        referenced_documents = extract_document_info(docs)
        print(f"[DEBUG] Extracted info from {len(referenced_documents)} documents")

        # Tambahkan konteks riwayat chat jika ada
        query_with_history = request.query
        if request.previous_responses:
            # Gunakan riwayat chat untuk memperkaya konteks jika diperlukan
            chat_history = format_chat_history(
                request.previous_responses, request.query
            )
            print(
                f"[DEBUG] Added chat history context with {len(request.previous_responses)} previous responses"
            )
        else:
            chat_history = ""

        # Execute RAG chain
        answer = rag_chain.invoke(query_with_history)

        # Extract document names for link retrieval
        doc_names = [
            doc.metadata.get("source", "").split("\\")[-1]
            for doc in docs
            if "source" in doc.metadata
        ]

        # Get document links using same embedding model for consistency
        document_links = find_document_links(
            doc_names, embedding_model=request.embedding_model
        )

        # Informasi model untuk respons
        model_info = {
            "model": model_name,
            "table": table_name,
        }

        return ChatResponse(
            answer=answer,
            document_links=document_links,
            model_info=model_info,
            referenced_documents=referenced_documents,
        )

    except Exception as e:
        print(f"[ERROR] Exception in chat endpoint: {str(e)}")
        import traceback

        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint - No authentication needed
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


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


# Run the API server
if __name__ == "__main__":
    # Print API key untuk setup
    print(f"API Key: {API_KEY}")
    uvicorn.run("simple_api:app", host="0.0.0.0", port=8000, reload=True)

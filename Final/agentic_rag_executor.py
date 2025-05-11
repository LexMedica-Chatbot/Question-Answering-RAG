import os
import time
import re
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKey, APIKeyHeader
from pydantic import BaseModel, Field
import uvicorn
from typing import List, Dict, Any, Optional, Literal, Union
import secrets
from starlette.status import HTTP_403_FORBIDDEN

# Langchain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.tools import BaseTool, tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory

# Supabase
from supabase.client import Client, create_client

# Load environment variables
load_dotenv()

# Security settings
API_KEY_NAME = "X-API-Key"
API_KEY = os.environ.get("API_KEY", secrets.token_urlsafe(32))
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


# Security dependency - verifikasi API key
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
    title="Sistem Agentic RAG Executor",
    description="API untuk sistem RAG dokumen hukum dengan pendekatan Agentic Executor",
    version="1.0.0",
)

# Add CORS middleware
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
EMBEDDING_CONFIG = {
    "small": {"model": "text-embedding-3-small", "table": "documents_small"},
    "large": {"model": "text-embedding-3-large", "table": "documents"},
}

# Initialize LLM with context window large enough for document processing
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# ======================= UTILITY FUNCTIONS =======================


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

    embeddings = get_embeddings(embedding_model)
    table_name = EMBEDDING_CONFIG[embedding_model]["table"]
    query_name = (
        "match_documents_small" if embedding_model == "small" else "match_documents"
    )

    return SupabaseVectorStore(
        embedding=embeddings,
        client=supabase,
        table_name=table_name,
        query_name=query_name,
    )


def format_docs(docs):
    """Format retrieved documents for context"""
    formatted_docs = []

    for i, doc in enumerate(docs):
        jenis_peraturan = doc.metadata.get("jenis_peraturan", "")
        nomor_peraturan = doc.metadata.get("nomor_peraturan", "")
        tahun_peraturan = doc.metadata.get("tahun_peraturan", "")
        tipe_bagian = doc.metadata.get("tipe_bagian", "")
        status = doc.metadata.get(
            "status", "berlaku"
        )  # Default ke "berlaku" jika tidak ada

        doc_header = f"Dokumen #{i+1}"

        if jenis_peraturan and nomor_peraturan and tahun_peraturan:
            doc_header += (
                f" ({jenis_peraturan} No. {nomor_peraturan} Tahun {tahun_peraturan}"
            )
            if tipe_bagian:
                doc_header += f" {tipe_bagian}"
            # Tambahkan status di header dokumen
            doc_header += f", Status: {status})"
        else:
            doc_header += ")"

        formatted_docs.append(f"{doc_header}:\n{doc.page_content}\n")

    return "\n\n".join(formatted_docs)


def extract_document_info(docs):
    """Ekstrak informasi dokumen dari hasil retrieval"""
    document_info = []
    for i, doc in enumerate(docs):
        jenis_peraturan = doc.metadata.get("jenis_peraturan", "")
        nomor_peraturan = doc.metadata.get("nomor_peraturan", "")
        tahun_peraturan = doc.metadata.get("tahun_peraturan", "")
        tipe_bagian = doc.metadata.get("tipe_bagian", "")
        judul_peraturan = doc.metadata.get("judul_peraturan", "")
        source = doc.metadata.get("source", "")
        status = doc.metadata.get(
            "status", "berlaku"
        )  # Default ke "berlaku" jika tidak ada

        doc_name = f"Dokumen #{i+1}"

        if jenis_peraturan and nomor_peraturan and tahun_peraturan:
            doc_description = (
                f"{jenis_peraturan} No. {nomor_peraturan} Tahun {tahun_peraturan}"
            )
            if tipe_bagian:
                doc_description += f" {tipe_bagian}"
            # Tambahkan status
            doc_description += f" (Status: {status})"
        else:
            doc_description = (
                source.split("\\")[-1] if "\\" in source else (source or doc_name)
            )

        additional_metadata = {}
        for key, value in doc.metadata.items():
            additional_metadata[key] = value

        document_info.append(
            {
                "name": doc_name,
                "description": doc_description,
                "source": (
                    source
                    or f"{jenis_peraturan} No. {nomor_peraturan}/{tahun_peraturan} (Status: {status})"
                    if jenis_peraturan and nomor_peraturan and tahun_peraturan
                    else "Metadata tidak lengkap"
                ),
                "content": doc.page_content,
                "metadata": additional_metadata,
            }
        )

    return document_info


def find_document_links(doc_names, embedding_model="large"):
    """Find document links based on document names"""
    document_links = []

    try:
        print(f"[DEBUG] Mencari link untuk dokumen: {doc_names}")

        # Karena match_document_links tidak tersedia, gunakan pendekatan yang sederhana
        # Tanpa memanggil fungsi Supabase yang tidak ada

        # Buat mapping manual untuk dokumen-dokumen
        document_mapping = {
            "PP No. 36 Tahun 2009": {
                "document_name": "Peraturan Pemerintah Nomor 36 Tahun 2009 tentang Kesehatan",
                "document_link": "https://peraturan.bpk.go.id/Home/Details/4986/pp-no-36-tahun-2009",
            },
            "PP No. 61 Tahun 2014": {
                "document_name": "Peraturan Pemerintah Nomor 61 Tahun 2014 tentang Kesehatan Reproduksi",
                "document_link": "https://peraturan.bpk.go.id/Home/Details/5548/pp-no-61-tahun-2014",
            },
            "PP No. 28 Tahun 2024": {
                "document_name": "Peraturan Pemerintah Nomor 28 Tahun 2024 tentang Kesehatan",
                "document_link": "https://peraturan.bpk.go.id/Home/Details/243856/pp-no-28-tahun-2024",
            },
            "UU No. 36 Tahun 2009": {
                "document_name": "Undang-Undang Nomor 36 Tahun 2009 tentang Kesehatan",
                "document_link": "https://peraturan.bpk.go.id/Home/Details/38778/uu-no-36-tahun-2009",
            },
        }

        for doc_name in doc_names:
            # Cari dokumen berdasarkan nama
            for key in document_mapping:
                if key in doc_name:
                    link_info = document_mapping[key]
                    # Periksa duplikat
                    already_exists = any(
                        existing["document_link"] == link_info["document_link"]
                        for existing in document_links
                    )

                    if not already_exists:
                        document_links.append(link_info)
                    break

        return document_links
    except Exception as e:
        print(f"[ERROR] Error finding document links: {str(e)}")
        return []


def extract_legal_entities(docs):
    """Extract legal entities from documents using pattern matching"""
    if not docs:
        return []

    entities = set()
    patterns = [
        r"(?:Undang-Undang|UU)(?:\s+Nomor|\s+No\.?)?(?:\s+\d+(?:/\d+)?)?(?:\s+Tahun\s+\d{4})?",
        r"(?:Peraturan\s+Pemerintah|PP)(?:\s+Nomor|\s+No\.?)?(?:\s+\d+(?:/\d+)?)?(?:\s+Tahun\s+\d{4})?",
        r"(?:Peraturan\s+Presiden|Perpres)(?:\s+Nomor|\s+No\.?)?(?:\s+\d+(?:/\d+)?)?(?:\s+Tahun\s+\d{4})?",
        r"(?:Peraturan\s+Menteri\s+Kesehatan|Permenkes)(?:\s+Nomor|\s+No\.?)?(?:\s+\d+(?:/\d+)?)?(?:\s+Tahun\s+\d{4})?",
        r"(?:Keputusan\s+Menteri\s+Kesehatan|Kepmenkes)(?:\s+Nomor|\s+No\.?)?(?:\s+\d+(?:/\d+)?)?(?:\s+Tahun\s+\d{4})?",
        r"Pasal\s+\d+(?:\s+[aA]yat\s+\d+)?",
    ]

    for doc in docs:
        text = doc.page_content
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                entities.add(match.strip())

    return list(entities)


# ======================= AGENT TOOLS DEFINITION =======================


@tool
def search_documents(query: str, embedding_model: str = "large", limit: int = 5) -> str:
    """
    Mencari dokumen dari vectorstore berdasarkan kueri yang diberikan.

    Args:
        query: Query pencarian untuk menemukan dokumen yang relevan
        embedding_model: Model embedding yang digunakan ("small" atau "large")
        limit: Jumlah dokumen yang dikembalikan

    Returns:
        String berformat yang berisi dokumen yang ditemukan
    """
    try:
        print(f"\n[TOOL] Searching for documents with query: {query}")
        vector_store = get_vector_store(embedding_model)
        retriever = vector_store.as_retriever(search_kwargs={"k": limit})

        # Perbarui dengan menggunakan metode invoke sebagai pengganti get_relevant_documents
        docs = retriever.invoke(query)

        if not docs:
            return "Tidak ditemukan dokumen yang relevan dengan query tersebut."

        formatted_docs = format_docs(docs)
        print(f"[TOOL] Found {len(docs)} documents")

        return formatted_docs
    except Exception as e:
        return f"Error pada pencarian dokumen: {str(e)}"


@tool
def refine_query(original_query: str, reason: str = "") -> str:
    """
    Menyempurnakan query pencarian untuk mendapatkan hasil yang lebih baik.

    Args:
        original_query: Query asli yang ingin disempurnakan
        reason: Alasan mengapa query perlu disempurnakan

    Returns:
        Query yang telah disempurnakan
    """
    try:
        print(f"\n[TOOL] Refining query: {original_query}")
        print(f"[TOOL] Reason for refinement: {reason}")

        # Use LLM to refine the query
        refiner = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

        refiner_prompt = f"""Sebagai asisten informasi hukum kesehatan Indonesia, sempurnakan query pencarian berikut untuk mendapatkan dokumen yang lebih relevan dan akurat.

Query asli: {original_query}

Alasan penyempurnaan: {reason}

Panduan penyempurnaan:
1. Tambahkan kata kunci spesifik terkait hukum kesehatan Indonesia
2. Fokuskan pada istilah teknis/legal yang tepat
3. Jika perlu, tambahkan referensi ke peraturan atau pasal spesifik
4. Sertakan frasa yang menunjukkan preferensi terhadap peraturan yang "berlaku" (jika sesuai)
5. Pertimbangkan sinonim dan variasi terminologi hukum
6. Hindari kata-kata ambigu

Kriteria hasil:
1. Query lebih spesifik dan tepat sasaran
2. Tetap mempertahankan maksud asli pengguna
3. Tidak lebih dari 2-3 kali panjang query asli
4. Menggunakan Bahasa Indonesia baku/formal
5. Terfokus pada dokumen dengan status "berlaku" kecuali query secara eksplisit mencari informasi historis

Berikan HANYA query yang sudah disempurnakan, tanpa penjelasan tambahan."""

        # Invoke the LLM
        result = refiner.invoke(refiner_prompt)
        refined_query = result.content.strip().replace('"', "").replace("'", "")

        # Jika query eksplisit tentang peraturan yang sudah dicabut, tetap pertahankan konteks tersebut
        if any(
            term in original_query.lower()
            for term in [
                "dicabut",
                "tidak berlaku",
                "sebelumnya",
                "lama",
                "historis",
                "dulu",
                "dahulu",
            ]
        ):
            # Jika memang tentang peraturan lama, pastikan konteks ini tetap ada
            pass
        # Jika query tidak eksplisit tentang peraturan lama dan refined_query tidak menyebutkan status
        elif (
            "berlaku" not in refined_query.lower()
            and "terbaru" not in refined_query.lower()
        ):
            # Tambahkan konteks untuk mencari peraturan yang berlaku
            refined_query += " yang masih berlaku"

        print(f"[TOOL] Refined query: {refined_query}")
        return refined_query
    except Exception as e:
        print(f"[ERROR] Error pada penyempurnaan query: {str(e)}")
        # Jika terjadi error, tambahkan term "yang berlaku" ke query asli sebagai fallback
        return f"{original_query} yang berlaku"


@tool
def evaluate_documents(documents: str, query: str) -> str:
    """
    Mengevaluasi kualitas dan relevansi dokumen untuk query.

    Args:
        documents: String berisi dokumen hasil pencarian
        query: Query yang perlu dijawab

    Returns:
        Hasil evaluasi dokumen: "MEMADAI" atau "KURANG MEMADAI" dengan alasan
    """
    try:
        print(f"\n[TOOL] Evaluating document quality for query: {query}")

        if (
            not documents
            or documents
            == "Tidak ditemukan dokumen yang relevan dengan query tersebut."
        ):
            return "KURANG MEMADAI: Tidak ditemukan dokumen yang relevan."

        # Periksa apakah dokumen cukup panjang - heuristik sederhana
        if len(documents) < 200:
            return "KURANG MEMADAI: Dokumen terlalu pendek untuk menjawab pertanyaan dengan baik."

        # Count the number of documents
        doc_count = documents.count("Dokumen #")

        # Periksa apakah ada dokumen berlaku (heuristic sederhana)
        berlaku_count = documents.count("Status: berlaku")
        dicabut_count = documents.count("Status: dicabut")

        print(
            f"[TOOL] Document evaluation - Doc count: {doc_count}, Berlaku: {berlaku_count}, Dicabut: {dicabut_count}"
        )

        # Use LLM to evaluate document adequacy for the query
        evaluator = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        evaluation_prompt = f"""Evaluasi SECARA OBJEKTIF apakah kumpulan dokumen berikut memberikan informasi YANG MEMADAI untuk menjawab query:

Query: {query}

Dokumen:
{documents}

1. Apakah dokumen berisi informasi relevan dengan query tersebut?
2. Apakah Anda dapat memberikan jawaban lengkap berdasarkan dokumen-dokumen ini?
3. Apakah dokumen-dokumen ini mencakup topik utama dari query?
4. Apakah ada peraturan yang masih "berlaku" (bukan dicabut) yang membahas query?

Berikan jawaban MEMADAI atau KURANG MEMADAI diikuti oleh alasan singkat.
Format: "MEMADAI: [alasan]" ATAU "KURANG MEMADAI: [alasan]"
Jawaban harus singkat dan langsung ke poin utama!"""

        result = evaluator.invoke(evaluation_prompt)
        evaluation = result.content.strip()

        print(f"[TOOL] Document evaluation result: {evaluation}")

        # Heuristik tambahan: jika evaluasi menyatakan MEMADAI tapi tidak ada dokumen yang berlaku
        # dan query membutuhkan peraturan terkini, maka override menjadi KURANG MEMADAI
        if "MEMADAI" in evaluation and berlaku_count == 0 and dicabut_count > 0:
            # Periksa jika query membutuhkan informasi terkini/terbaru
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
                evaluation = "KURANG MEMADAI: Hanya ditemukan peraturan yang sudah dicabut, sementara query membutuhkan informasi peraturan yang masih berlaku."
                print(
                    f"[TOOL] Overriding evaluation due to missing current regulations: {evaluation}"
                )

        return evaluation
    except Exception as e:
        print(f"[ERROR] Error pada evaluasi dokumen: {str(e)}")
        return "KURANG MEMADAI: Terjadi error dalam evaluasi dokumen."


@tool
def generate_answer(documents: str, query: str) -> str:
    """
    Menghasilkan jawaban berdasarkan dokumen yang ditemukan.

    Args:
        documents: String berisi dokumen hasil pencarian
        query: Query yang perlu dijawab

    Returns:
        Jawaban lengkap berdasarkan dokumen
    """
    try:
        print(f"\n[TOOL] Generating answer for query: {query}")

        if (
            not documents
            or documents
            == "Tidak ditemukan dokumen yang relevan dengan query tersebut."
        ):
            return "Mohon maaf, tidak ada informasi yang cukup dalam database kami untuk menjawab pertanyaan Anda. Silakan coba pertanyaan lain atau hubungi admin sistem untuk informasi lebih lanjut."

        # Extract entities from documents - contoh pola sederhana untuk entitas hukum
        # Tanpa menggunakan fungsi extract_legal_entities
        legal_entities = []
        patterns = [
            r"(?:Undang-Undang|UU)(?:\s+Nomor|\s+No\.?)?(?:\s+\d+(?:/\d+)?)?(?:\s+Tahun\s+\d{4})?",
            r"(?:Peraturan\s+Pemerintah|PP)(?:\s+Nomor|\s+No\.?)?(?:\s+\d+(?:/\d+)?)?(?:\s+Tahun\s+\d{4})?",
            r"(?:Peraturan\s+Presiden|Perpres)(?:\s+Nomor|\s+No\.?)?(?:\s+\d+(?:/\d+)?)?(?:\s+Tahun\s+\d{4})?",
            r"(?:Peraturan\s+Menteri\s+Kesehatan|Permenkes)(?:\s+Nomor|\s+No\.?)?(?:\s+\d+(?:/\d+)?)?(?:\s+Tahun\s+\d{4})?",
            r"(?:Keputusan\s+Menteri\s+Kesehatan|Kepmenkes)(?:\s+Nomor|\s+No\.?)?(?:\s+\d+(?:/\d+)?)?(?:\s+Tahun\s+\d{4})?",
            r"Pasal\s+\d+(?:\s+[aA]yat\s+\d+)?",
        ]

        # Langsung ekstrak dari string dokumen
        for pattern in patterns:
            matches = re.findall(pattern, documents)
            for match in matches:
                if match.strip() and match.strip() not in legal_entities:
                    legal_entities.append(match.strip())

        entities_str = ", ".join(legal_entities)

        # Generate answer - panggil LLM langsung tanpa chain
        generator = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

        generator_prompt = f"""# PERAN DAN PEMBATASAN (Framework COSTAR)

## Context
Anda adalah asisten AI untuk dokumen-dokumen hukum kesehatan di Indonesia. Anda dirancang untuk membantu pengguna menemukan informasi terkait regulasi kesehatan, peraturan, dan ketentuan hukum di bidang kesehatan Indonesia. Anda hanya menjawab pertanyaan berdasarkan dokumen hukum kesehatan resmi yang tersedia dalam database.

## Objective
Tujuan utama Anda adalah memberikan informasi faktual, akurat dan komprehensif dari dokumen hukum kesehatan Indonesia, membantu pengguna memahami aturan, prosedur, ketentuan, dan informasi legal dalam bidang kesehatan.

## Scope
Anda hanya menjawab pertanyaan dalam ruang lingkup dokumen yang diberikan.
Anda TIDAK akan menjawab pertanyaan di luar konteks dokumen yang tersedia, memberikan nasihat medis atau hukum personal, atau memprediksi hasil kasus hukum spesifik.

## Tone
Gunakan Bahasa Indonesia yang formal, jelas, presisi dan sesuai dengan terminologi hukum kesehatan. Jawaban Anda harus sistematis, terstruktur, dan mengikuti hierarki peraturan perundang-undangan Indonesia.

## Authority
Otoritas Anda berasal dari dokumen resmi yang diberikan. Jangan memberikan pendapat pribadi atau spekulasi.

## Role
Anda adalah asisten informasi dokumen hukum kesehatan, bukan penasihat medis atau hukum.

# PENANGANAN STATUS PERATURAN (BERLAKU vs DICABUT)
1. Konteks akan berisi dokumen dari peraturan dengan status "berlaku" dan "dicabut"
2. PRIORITASKAN informasi dari dokumen berstatus "berlaku" dalam memberikan jawaban
3. Jika ada kontradiksi antara dokumen "berlaku" dan "dicabut", gunakan dokumen yang "berlaku"
4. Anda BOLEH mereferensikan dokumen yang sudah "dicabut" untuk:
   - Memberikan konteks historis perkembangan regulasi
   - Menjelaskan perubahan regulasi dari waktu ke waktu
   - Membandingkan peraturan lama dan baru bila relevan
5. SELALU sebutkan status peraturan saat Anda merujuk ke suatu dokumen, misalnya:
   - "[Dok#1] PP No. 28 Tahun 2024 (berlaku) menyatakan bahwa..."
   - "[Dok#2] UU No. 36 Tahun 2009 (dicabut) pada saat itu mengatur bahwa..."
6. Prioritaskan menyimpulkan berdasarkan peraturan terbaru yang masih berlaku

# INSTRUKSI CONTEXTUAL CITATION
Untuk setiap pernyataan faktual, HARUS merujuk ke dokumen sumber dengan format:
[Dok#N] dimana N adalah nomor dokumen dalam konteks yang disediakan

# INSTRUKSI PENTING
1. SELALU mulai jawaban Anda dengan menyebutkan sumber peraturan yang relevan termasuk statusnya
2. Jangan berhalusinasi atau membuat informasi yang tidak ada dalam dokumen
3. Jika informasi tidak cukup, nyatakan dengan jujur bahwa dokumen tidak memuat informasi yang cukup

Pertanyaan pengguna: {query}

Berikut adalah dokumen-dokumen yang relevan dengan pertanyaan:

{documents}

Berdasarkan analisis entitas dalam dokumen, perhatikan entitas-entitas hukum berikut:
{entities_str}"""

        # Panggil LLM langsung
        result = generator.invoke(generator_prompt)
        answer = result.content

        print(f"[TOOL] Generated answer (preview): {answer[:100]}...")
        return answer
    except Exception as e:
        print(f"[ERROR] Error pada pembuatan jawaban: {str(e)}")
        return f"Error pada pembuatan jawaban: {str(e)}"


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


# ======================= AGENT DEFINITION =======================

# Setup system prompt for the agent
system_prompt = """Anda adalah asisten hukum kesehatan Indonesia berbasis AI yang menggunakan pendekatan RAG (Retrieval-Augmented Generation) untuk menjawab pertanyaan.

TUGAS ANDA:
1. Memahami pertanyaan pengguna tentang hukum kesehatan Indonesia 
2. Mencari dokumen yang relevan dengan pertanyaan
3. Mengevaluasi apakah dokumen yang ditemukan memadai untuk menjawab pertanyaan
4. Jika tidak memadai, sempurnakan query dan cari lagi
5. Jika setelah upaya penyempurnaan masih tidak memadai, minta pengguna memberikan query baru
6. Jika memadai, hasilkan jawaban yang komprehensif berdasarkan dokumen

ALUR KERJA ANDA:
1. Selalu mulai dengan mencari dokumen menggunakan query asli
2. Evaluasi kualitas dan relevansi dokumen yang ditemukan
3. Jika dokumen "KURANG MEMADAI", sempurnakan query dan cari lagi
4. Jika dokumen "MEMADAI", hasilkan jawaban
5. Jika setelah 2 kali penyempurnaan masih "KURANG MEMADAI", minta query baru

ATURAN PENTING:
1. Jawaban HARUS berdasarkan dokumen yang ditemukan, bukan pengetahuan umum Anda
2. Jangan berikan jawaban parsial jika dokumen tidak memadai
3. Gunakan Bahasa Indonesia formal dan terminologi hukum yang tepat
4. Selalu rujuk ke dokumen dengan format [Dok#N] dalam jawaban Anda
5. Fokus pada aspek hukum, bukan aspek medis atau klinis
6. Hanya informasi dari database dokumen hukum kesehatan yang dapat digunakan

PENANGANAN STATUS PERATURAN:
1. Dokumen yang ditemukan bisa memiliki status "berlaku" atau "dicabut"
2. Prioritaskan informasi dari dokumen dengan status "berlaku"
3. Ketika memberikan jawaban, selalu sebutkan status peraturan yang dirujuk
4. Peraturan yang "dicabut" hanya digunakan untuk konteks historis

INGAT: Jika tidak menemukan informasi yang cukup setelah beberapa upaya, lebih baik meminta query baru daripada memberikan jawaban yang tidak akurat!"""

# Create the agent with tools
tools = [
    search_documents,
    refine_query,
    evaluate_documents,
    generate_answer,
    request_new_query,
]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="output",  # Tetapkan output_key secara eksplisit
)

# Buat agent langsung tanpa perlu menggunakan format_to_openai_tool_messages
agent = create_openai_tools_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
)

# ======================= API MODELS =======================


class AgenticRequest(BaseModel):
    query: str
    embedding_model: Literal["small", "large"] = "large"
    conversation_id: Optional[str] = None
    debug_mode: bool = False


class StepInfo(BaseModel):
    tool: str
    tool_input: Dict[str, Any]
    tool_output: str


class CitationInfo(BaseModel):
    text: str
    source_doc: str
    source_text: str


class AgenticResponse(BaseModel):
    answer: str
    document_links: List[Dict[str, Any]] = []
    referenced_documents: List[Dict[str, Any]] = []
    citations: List[CitationInfo] = []
    agent_steps: Optional[List[StepInfo]] = None
    processing_time_ms: Optional[int] = None


# ======================= API ENDPOINTS =======================


@app.post(
    "/api/chat",
    response_model=AgenticResponse,
    dependencies=[Depends(verify_api_key)],
)
async def agentic_chat(request: AgenticRequest):
    try:
        start_time = time.time()
        print(f"\n[API] Processing chat request: {request.query}")

        # Execute agent with conversation memory
        result = agent_executor.invoke(
            {
                "input": request.query,
            }
        )

        # Extract the final answer from the agent result
        answer = result["output"]
        print(f"\n[API] Agent output: {answer[:100]}...")

        # Extract document links by analyzing the agent steps
        doc_names = []
        referenced_documents = []
        all_docs = []

        # Parse intermediate steps to extract documents
        for step in result["intermediate_steps"]:
            tool_name = step[0].tool

            # Collect documents from search_documents tool calls
            if tool_name == "search_documents" and isinstance(step[1], str):
                if "Tidak ditemukan dokumen" not in step[1]:
                    # Parse the documents from the formatted output
                    docs_list = []
                    current_doc = ""
                    current_header = ""

                    for line in step[1].split("\n"):
                        if line.startswith("Dokumen #"):
                            if current_doc and current_header:
                                docs_list.append(
                                    {
                                        "header": current_header,
                                        "content": current_doc.strip(),
                                    }
                                )
                                # Extract document name for links
                                if "(" in current_header and ")" in current_header:
                                    doc_name = current_header.split("(")[1].split(")")[
                                        0
                                    ]
                                    if doc_name not in doc_names:
                                        doc_names.append(doc_name)
                            current_header = line
                            current_doc = ""
                        else:
                            current_doc += line + "\n"

                    if current_doc and current_header:
                        docs_list.append(
                            {"header": current_header, "content": current_doc.strip()}
                        )
                        # Extract document name for links
                        if "(" in current_header and ")" in current_header:
                            doc_name = current_header.split("(")[1].split(")")[0]
                            if doc_name not in doc_names:
                                doc_names.append(doc_name)

                    # Add to all docs for referenced_documents
                    all_docs.extend(docs_list)

        # Create referenced_documents in the right format
        for i, doc in enumerate(all_docs):
            doc_info = {
                "name": f"Dokumen #{i+1}",
                "description": doc["header"]
                .replace(f"Dokumen #{i+1}", "")
                .strip()
                .strip("()"),
                "source": doc["header"],
                "content": doc["content"],
                "metadata": {},
            }
            referenced_documents.append(doc_info)

        # Find document links
        document_links = find_document_links(
            doc_names, embedding_model=request.embedding_model
        )

        # Extract citations from answer
        citations = []
        citation_pattern = r"\[Dok#(\d+)\]"
        citation_matches = re.findall(citation_pattern, answer)

        for match in citation_matches:
            doc_idx = int(match) - 1
            if 0 <= doc_idx < len(all_docs):
                citation = CitationInfo(
                    text=f"Referensi ke Dokumen #{int(match)}",
                    source_doc=f"Dokumen #{int(match)}",
                    source_text=(
                        all_docs[doc_idx]["content"][:100] + "..."
                        if len(all_docs[doc_idx]["content"]) > 100
                        else all_docs[doc_idx]["content"]
                    ),
                )
                citations.append(citation)

        # Format agent steps for debug mode
        agent_steps = None
        if request.debug_mode:
            agent_steps = []
            for step in result["intermediate_steps"]:
                step_info = StepInfo(
                    tool=step[0].tool,
                    tool_input=step[0].tool_input,
                    tool_output=(
                        step[1][:500] + "..." if len(step[1]) > 500 else step[1]
                    ),
                )
                agent_steps.append(step_info)

        # Calculate processing time
        end_time = time.time()
        processing_time_ms = int((end_time - start_time) * 1000)

        return AgenticResponse(
            answer=answer,
            document_links=document_links,
            referenced_documents=referenced_documents,
            citations=citations,
            agent_steps=agent_steps,
            processing_time_ms=processing_time_ms,
        )

    except Exception as e:
        print(f"[ERROR] Exception in chat endpoint: {str(e)}")
        import traceback

        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy", "system": "agentic RAG executor"}


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
    print(f"API Key: {API_KEY}")
    uvicorn.run("agentic_rag_executor:app", host="0.0.0.0", port=8000, reload=True)

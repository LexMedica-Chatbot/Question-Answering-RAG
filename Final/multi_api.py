import os
import time
import re
import json
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
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
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
EMBEDDING_CONFIG = {
    "small": {"model": "text-embedding-3-small", "table": "documents_small"},
    "large": {"model": "text-embedding-3-large", "table": "documents"},
}

# Initialize LLM with context window large enough for document processing
# Definisikan konfigurasi model
MODELS = {
    "MAIN": {"model": "gpt-4.1-mini", "temperature": 0.2},
    # gabungkan refiner + evaluator:
    "REF_EVAL": {"model": "gpt-4.1-nano", "temperature": 0.25},
    "GENERATOR": {"model": "gpt-4o-mini", "temperature": 0.2},
}

# Initialize LLM dengan model utama
llm = ChatOpenAI(**MODELS["MAIN"])

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


def extract_document_info(doc_content_str: str):
    """
    Mengekstrak informasi dari KONTEN STRING sebuah dokumen tunggal.
    Ini digunakan sebagai fallback atau pelengkap jika metadata dari header tidak cukup.
    """
    info = {
        "jenis_peraturan": "",
        "nomor_peraturan": "",
        "tahun_peraturan": "",
        "tipe_bagian": "",
        "judul_peraturan": "",
        "status": "berlaku",  # Default status ke berlaku
        "source": "",
        "doc_name": "",
    }

    if not isinstance(doc_content_str, str) or not doc_content_str.strip():
        return info

    # 1. Coba ekstrak Jenis, Nomor, Tahun Peraturan dari konten
    # Regex yang lebih komprehensif untuk menangkap berbagai format peraturan
    peraturan_patterns = [
        # Pattern untuk format lengkap: "UU No. 1 Tahun 2023"
        r"(undang-undang|uu|peraturan\s+pemerintah|pp|peraturan\s+presiden|perpres|peraturan\s+menteri\s+kesehatan|permenkes|keputusan\s+menteri\s+kesehatan|kepmenkes)(?:\s+republik\s+indonesia)?(?:\s+nomor|\s+no\.|\s+no)?\s*(\d+(?:/\d+)?)(?:\s+tahun)?\s*(\d{4})?",
        # Pattern untuk format singkat: "UU 1/2023"
        r"(undang-undang|uu|peraturan\s+pemerintah|pp|peraturan\s+presiden|perpres|peraturan\s+menteri\s+kesehatan|permenkes|keputusan\s+menteri\s+kesehatan|kepmenkes)(?:\s+republik\s+indonesia)?\s*(\d+(?:/\d+)?)(?:/\d{4})?",
    ]

    for pattern in peraturan_patterns:
        peraturan_match = re.search(pattern, doc_content_str, re.IGNORECASE)
        if peraturan_match:
            jenis_raw = peraturan_match.group(1).lower()
            # Normalisasi jenis peraturan
            if "undang-undang" in jenis_raw or "uu" in jenis_raw:
                info["jenis_peraturan"] = "UU"
            elif "peraturan pemerintah" in jenis_raw or "pp" in jenis_raw:
                info["jenis_peraturan"] = "PP"
            elif "peraturan presiden" in jenis_raw or "perpres" in jenis_raw:
                info["jenis_peraturan"] = "PERPRES"
            elif "peraturan menteri kesehatan" in jenis_raw or "permenkes" in jenis_raw:
                info["jenis_peraturan"] = "PERMENKES"
            elif "keputusan menteri kesehatan" in jenis_raw or "kepmenkes" in jenis_raw:
                info["jenis_peraturan"] = "KEPMENKES"
            else:
                info["jenis_peraturan"] = jenis_raw.upper()

            # Ekstrak nomor peraturan
            if len(peraturan_match.groups()) >= 2:
                nomor = peraturan_match.group(2).strip()
                # Jika format "1/2023", pisahkan nomor dan tahun
                if "/" in nomor and len(peraturan_match.groups()) < 3:
                    nomor, tahun = nomor.split("/")
                    info["nomor_peraturan"] = nomor.strip()
                    info["tahun_peraturan"] = tahun.strip()
                else:
                    info["nomor_peraturan"] = nomor

            # Ekstrak tahun peraturan jika ada
            if len(peraturan_match.groups()) >= 3 and peraturan_match.group(3):
                info["tahun_peraturan"] = peraturan_match.group(3).strip()
            break

    # 2. Coba ekstrak Judul Peraturan (setelah TENTANG)
    judul_patterns = [
        # Pattern untuk "TENTANG" diikuti judul
        r"(?:tentang|TENTANG)\s+(.+?)(?:Menimbang:|Mengingat:|Pasal\s+1|BAB\s+I|\n\n)",
        # Pattern untuk judul dalam tanda kutip
        r"\"(.+?)\"",
        # Pattern untuk judul setelah nomor peraturan
        r"(?:No\.\s*\d+(?:/\d+)?\s+Tahun\s+\d{4})\s+(.+?)(?:Menimbang:|Mengingat:|Pasal\s+1|BAB\s+I|\n\n)",
    ]

    for pattern in judul_patterns:
        judul_match = re.search(pattern, doc_content_str, re.IGNORECASE | re.DOTALL)
        if judul_match:
            judul = judul_match.group(1).strip()
            # Bersihkan judul dari karakter yang tidak diinginkan
            judul = re.sub(
                r"\s+", " ", judul
            )  # Ganti multiple spaces dengan single space
            judul = judul.replace("\n", " ").strip()
            if len(judul) > 5:  # Pastikan judul memiliki panjang yang masuk akal
                info["judul_peraturan"] = judul
                break

    # 3. Coba ekstrak Tipe Bagian (Pasal, BAB, dll.)
    tipe_bagian_patterns = [
        # Pattern untuk BAB
        r"(?:BAB|Bab)\s+([IVXLC\d]+)(?:\s+[A-Z\s]+)?",
        # Pattern untuk Pasal
        r"(?:Pasal|PASAL)\s+(\d+(?:[a-z])?)",
        # Pattern untuk Bagian
        r"(?:Bagian|BAGIAN)\s+([IVXLC\d]+)",
        # Pattern untuk Paragraf
        r"(?:Paragraf|PARAGRAF)\s+([IVXLC\d]+)",
    ]

    for pattern in tipe_bagian_patterns:
        tipe_match = re.search(pattern, doc_content_str, re.IGNORECASE)
        if tipe_match:
            tipe = tipe_match.group(0).strip()
            info["tipe_bagian"] = tipe
            break

    # 4. Coba ekstrak Status (berlaku/dicabut)
    status_patterns = [
        r"(?:dicabut|DICABUT|tidak berlaku|TIDAK BERLAKU)",
        r"(?:berlaku|BERLAKU|masih berlaku|MASIH BERLAKU)",
    ]

    for pattern in status_patterns:
        status_match = re.search(pattern, doc_content_str, re.IGNORECASE)
        if status_match:
            status_text = status_match.group(0).lower()
            if "dicabut" in status_text or "tidak berlaku" in status_text:
                info["status"] = "dicabut"
            elif "berlaku" in status_text:
                info["status"] = "berlaku"
            break

    # 5. Buat doc_name dari informasi yang diekstrak
    if info["jenis_peraturan"] and info["nomor_peraturan"] and info["tahun_peraturan"]:
        info["doc_name"] = (
            f"{info['jenis_peraturan']} No. {info['nomor_peraturan']} Tahun {info['tahun_peraturan']}"
        )
        if info["judul_peraturan"]:
            info["doc_name"] += f" tentang {info['judul_peraturan']}"
    elif info["judul_peraturan"]:  # Fallback jika hanya judul yang ada
        info["doc_name"] = info["judul_peraturan"]

    # 6. Ekstrak source (nama file) jika ada polanya
    source_patterns = [
        r"(?:source:|Sumber Dokumen:|Nama File:)\s*([^\n]+)",
        r"(?:file:|dokumen:)\s*([^\n]+)",
    ]

    for pattern in source_patterns:
        source_match = re.search(pattern, doc_content_str, re.IGNORECASE)
        if source_match:
            info["source"] = source_match.group(1).strip()
            break

    # Bersihkan spasi ekstra dari semua field
    for key, value in info.items():
        if isinstance(value, str):
            info[key] = value.strip()

    return info


def find_document_links(doc_names, embedding_model="large"):
    """Find document links based on document names"""
    # Fungsi ini tetap sebagai placeholder
    # Tidak lagi menggunakan document_mapping hardcoded
    print(f"[DEBUG] Referensi dokumen: {doc_names}")

    # Return array kosong karena tidak ada mapping hardcoded
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
        String JSON yang berisi dokumen yang diformat untuk LLM dan data dokumen terstruktur
    """
    try:
        print(f"\n[TOOL] Searching for documents with query: {query}")

        # Gunakan vector store dari global cache
        vector_store = get_vector_store(embedding_model)
        retriever = vector_store.as_retriever(search_kwargs={"k": limit})
        docs = retriever.invoke(query)

        if not docs:
            return json.dumps(
                {
                    "formatted_docs_for_llm": "Tidak ditemukan dokumen yang relevan dengan query tersebut.",
                    "retrieved_docs_data": [],
                }
            )

        # Format dokumen untuk konteks LLM
        formatted_docs_for_llm = format_docs(docs)

        # Siapkan data terstruktur untuk setiap dokumen yang diambil
        retrieved_docs_data = []
        for i, doc in enumerate(docs):
            metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
            content = doc.page_content

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

            # Buat deskripsi lengkap
            description = (
                f"{doc_name} (Status: {status})"
                if doc_name
                else f"Dokumen #{i+1} (Status: {status})"
            )

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

            retrieved_docs_data.append(
                {
                    "name": doc_name or f"Dokumen #{i+1}",
                    "description": description,
                    "source": doc_name or f"Dokumen #{i+1}",
                    "content": content,
                    "metadata": structured_metadata,
                }
            )

        print(f"[TOOL] Found {len(docs)} documents")

        return json.dumps(
            {
                "formatted_docs_for_llm": formatted_docs_for_llm,
                "retrieved_docs_data": retrieved_docs_data,
            }
        )
    except Exception as e:
        print(f"[ERROR] Error pada pencarian dokumen: {str(e)}")
        return json.dumps(
            {
                "formatted_docs_for_llm": f"Error pada pencarian dokumen: {str(e)}",
                "retrieved_docs_data": [],
            }
        )


# Tambahkan variabel untuk melacak jumlah penyempurnaan
refinement_count = 0


# Modifikasi fungsi refine_query untuk membatasi jumlah penyempurnaan
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
    global refinement_count

    # Jika sudah mencapai batas penyempurnaan, kembalikan query asli
    if refinement_count >= 1:
        print(
            f"\n[TOOL] Mencapai batas maksimum penyempurnaan query ({refinement_count})"
        )
        return original_query

    refinement_count += 1
    try:
        print(f"\n[TOOL] Refining query: {original_query}")
        print(f"[TOOL] Reason for refinement: {reason}")

        # Use LLM to refine the query
        refiner = ChatOpenAI(**MODELS["REF_EVAL"])

        refiner_prompt = f"""Sebagai asisten informasi hukum kesehatan Indonesia, sempurnakan query pencarian berikut untuk mendapatkan dokumen yang lebih relevan dan akurat.

Query asli: {original_query}

Alasan penyempurnaan: {reason}

Panduan penyempurnaan:
1. Tambahkan kata kunci spesifik terkait hukum kesehatan Indonesia
2. Fokuskan pada istilah teknis/legal yang tepat
3. Jika perlu, tambahkan referensi ke peraturan atau pasal spesifik
4. Sertakan frasa yang menunjukkan preferensi terhadap peraturan yang "berlaku"
5. Pertimbangkan sinonim dan variasi terminologi hukum
6. Hindari kata-kata ambigu

Kriteria hasil:
1. Query lebih spesifik dan tepat sasaran
2. Tetap mempertahankan maksud asli pengguna
3. Tidak lebih dari 2-3 kali panjang query asli
4. Menggunakan Bahasa Indonesia baku/formal
5. Terfokus pada dokumen dengan status "berlaku"

Berikan HANYA query yang sudah disempurnakan, tanpa penjelasan tambahan."""

        # Invoke the LLM
        result = refiner.invoke(refiner_prompt)
        refined_query = result.content.strip().replace('"', "").replace("'", "")

        # Jika query eksplisit tentang peraturan yang sudah dicabut, tetap pertahankan konteks tersebut
        historical_terms = [
            "dicabut",
            "tidak berlaku",
            "sebelumnya",
            "lama",
            "historis",
            "dulu",
            "dahulu",
        ]
        if any(term in original_query.lower() for term in historical_terms):
            # Jika memang tentang peraturan lama, pastikan konteks ini tetap ada
            pass
        # Jika query tidak eksplisit tentang peraturan lama dan refined_query tidak menyebutkan status
        elif "berlaku" not in refined_query.lower():
            # Tambahkan konteks untuk mencari peraturan yang berlaku
            refined_query += " yang masih berlaku"

        print(f"[TOOL] Refined query: {refined_query}")
        return refined_query
    except Exception as e:
        print(f"[ERROR] Error pada penyempurnaan query: {str(e)}")
        # Jika terjadi error, tambahkan term "yang berlaku" ke query asli sebagai fallback
        return f"{original_query} yang berlaku"


@tool
def evaluate_documents(query: str, documents: str) -> str:
    """
    Mengevaluasi kualitas dan relevansi dokumen untuk query.

    Args:
        query: Query yang perlu dijawab (wajib)
        documents: String JSON berisi dokumen hasil pencarian dengan format:
            {
                "formatted_docs_for_llm": string,
                "retrieved_docs_data": [
                    {
                        "name": string,
                        "description": string,
                        "source": string,
                        "content": string,
                        "metadata": {
                            "status": string,
                            "bagian_dari": string,
                            "tipe_bagian": string,
                            "jenis_peraturan": string,
                            "judul_peraturan": string,
                            "nomor_peraturan": string,
                            "tahun_peraturan": string
                        }
                    }
                ]
            }

    Returns:
        Hasil evaluasi dokumen: "MEMADAI" atau "KURANG MEMADAI" dengan alasan
    """
    try:
        print(f"\n[TOOL] Evaluating document quality for query: {query}")

        # Parse JSON input
        try:
            json_data = json.loads(documents)
            if not isinstance(json_data, dict):
                raise ValueError("Input bukan JSON object yang valid")

            retrieved_docs = json_data.get("retrieved_docs_data", [])
            if not retrieved_docs:
                return "KURANG MEMADAI: Tidak ditemukan dokumen yang relevan."

        except json.JSONDecodeError as e:
            print(f"[ERROR] Gagal parse JSON input: {str(e)}")
            return "KURANG MEMADAI: Format input dokumen tidak valid"

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

        print(
            f"[TOOL] Document evaluation - Doc count: {doc_count}, Berlaku: {berlaku_count}, Dicabut: {dicabut_count}"
        )

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

        if need_current_info and berlaku_count == 0 and dicabut_count > 0:
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

        print(f"[TOOL] Document evaluation result: {evaluation}")

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
                f"[TOOL] Overriding evaluation due to missing current regulations: {evaluation}"
            )

        return evaluation

    except Exception as e:
        print(f"[ERROR] Error pada evaluasi dokumen: {str(e)}")
        return "KURANG MEMADAI: Terjadi error dalam evaluasi dokumen."


def process_documents(formatted_docs_string: str):
    """
    Memproses string dokumen yang diformat (dari format_docs atau output search_documents)
    dan mengekstrak informasi untuk setiap dokumen.
    Mengembalikan list of dictionaries, setiap dict punya 'name', 'description', 'source', 'content', 'metadata'.
    """
    document_info_list = []
    if not isinstance(formatted_docs_string, str) or not formatted_docs_string.strip():
        print("[PROCESS_DOCUMENTS] Input string kosong atau bukan string.")
        return []

    try:
        # Coba parse sebagai JSON terlebih dahulu
        json_data = json.loads(formatted_docs_string)
        if isinstance(json_data, dict) and "metadata" in json_data:
            # Jika ini adalah output dari search_documents yang baru
            for doc_metadata in json_data["metadata"]:
                # Buat nama dokumen dari metadata
                doc_name = doc_metadata.get("doc_name", "")
                if not doc_name and doc_metadata.get("jenis_peraturan"):
                    doc_name = f"{doc_metadata['jenis_peraturan']} No. {doc_metadata.get('nomor_peraturan', '')} Tahun {doc_metadata.get('tahun_peraturan', '')}"
                    if doc_metadata.get("tipe_bagian"):
                        doc_name += f" {doc_metadata['tipe_bagian']}"

                # Buat deskripsi dokumen
                description = doc_name
                if doc_metadata.get("status"):
                    description += f" (Status: {doc_metadata['status']})"

                # Buat source
                source = doc_metadata.get("source", doc_name)

                document_info_list.append(
                    {
                        "name": doc_name
                        or f"Dokumen Tidak Dikenal {len(document_info_list) + 1}",
                        "description": description,
                        "source": source,
                        "content": doc_metadata.get("content", ""),
                        "metadata": doc_metadata,
                    }
                )
            return document_info_list
    except json.JSONDecodeError:
        # Jika bukan JSON, proses sebagai string format lama
        pass

    # Proses sebagai string format lama
    doc_pattern = re.compile(
        r"^(Dokumen #\d+ .*?\((?:.*?Status: (berlaku|dicabut))\)):?\n(.*?)(?=^Dokumen #\d+ .*?:?\n|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    matches = doc_pattern.findall(formatted_docs_string)

    if not matches:
        print(
            f"[PROCESS_DOCUMENTS] Tidak ada dokumen yang cocok dengan pola dari input string: {formatted_docs_string[:200]}..."
        )

    for idx, (full_header, status_from_header, content_str) in enumerate(matches):
        content_str = content_str.strip()
        full_header = full_header.strip().rstrip(":")

        # Ekstrak metadata dari konten menggunakan regex
        metadata_from_content_regex = extract_document_info(content_str)

        final_status = status_from_header.lower()

        # Ekstrak nama dokumen dari header (lebih diutamakan jika ada)
        parsed_name_from_header = ""
        name_header_match = re.search(
            r"Dokumen #\d+ \((.*?)(?:, Status: (?:berlaku|dicabut))\)", full_header
        )
        if name_header_match and name_header_match.group(1):
            parsed_name_from_header = name_header_match.group(1).strip()

        # Ekstrak informasi peraturan dari nama dokumen
        peraturan_match = re.search(
            r"(UU|PP|PERPRES|PERMENKES|KEPMENKES)\s+No\.\s+(\d+)\s+Tahun\s+(\d+)(?:\s+(.*))?",
            parsed_name_from_header,
        )

        if peraturan_match:
            jenis_peraturan = peraturan_match.group(1)
            nomor_peraturan = peraturan_match.group(2)
            tahun_peraturan = peraturan_match.group(3)
            tipe_bagian = peraturan_match.group(4) if peraturan_match.group(4) else ""

            # Cari bagian_dari dari konten
            bagian_dari = ""
            bagian_match = re.search(r"Bagian\s+[IVXLC\d]+\s*-\s*([^\n]+)", content_str)
            if bagian_match:
                bagian_dari = f"Bab {bagian_match.group(0)}"

            # Update metadata dengan informasi yang diekstrak
            metadata_from_content_regex = {
                "status": final_status,
                "bagian_dari": bagian_dari,
                "tipe_bagian": tipe_bagian,
                "jenis_peraturan": jenis_peraturan,
                "judul_peraturan": "",
                "nomor_peraturan": nomor_peraturan,
                "tahun_peraturan": tahun_peraturan,
            }

        # Gunakan nama dari regex konten jika lebih detail, jika tidak gunakan dari header
        doc_name_from_content_regex = metadata_from_content_regex.get("doc_name")
        final_doc_name = (
            doc_name_from_content_regex
            if doc_name_from_content_regex
            else parsed_name_from_header
        )

        if not final_doc_name:  # Fallback jika nama masih kosong
            doc_num_match = re.search(r"^(Dokumen #\d+)", full_header)
            final_doc_name = (
                doc_num_match.group(1)
                if doc_num_match
                else f"Dokumen Tidak Dikenal {idx + 1}"
            )

        document_info_list.append(
            {
                "name": final_doc_name,
                "description": full_header,
                "source": parsed_name_from_header or final_doc_name,
                "content": content_str,
                "metadata": metadata_from_content_regex,
            }
        )

    return document_info_list


def format_reference(doc_info: dict):
    """
    Memformat referensi dokumen sesuai dengan format yang diinginkan.
    doc_info adalah sebuah dictionary dari list yang dihasilkan oleh process_documents.
    """
    try:
        doc_name = doc_info.get("name", "Nama Dokumen Tidak Diketahui")
        # Ambil status dari metadata yang sudah diproses di process_documents
        status = (
            doc_info.get("metadata", {}).get("status", "status tidak diketahui").lower()
        )
        return f"[{doc_name}] ({status})"
    except Exception as e:
        print(f"Error in format_reference: {str(e)}")
        return "[Dokumen] (status tidak diketahui)"


@tool
def generate_answer(documents: str, query: str = None) -> str:
    """
    Menghasilkan jawaban berdasarkan dokumen yang ditemukan.

    Args:
        documents: String JSON berisi dokumen hasil pencarian dengan format:
            {
                "formatted_docs_for_llm": string,
                "retrieved_docs_data": [
                    {
                        "name": string,
                        "description": string,
                        "source": string,
                        "content": string,
                        "metadata": {
                            "status": string,
                            "bagian_dari": string,
                            "tipe_bagian": string,
                            "jenis_peraturan": string,
                            "judul_peraturan": string,
                            "nomor_peraturan": string,
                            "tahun_peraturan": string
                        }
                    }
                ]
            }
        query: Query yang perlu dijawab (wajib)

    Returns:
        Jawaban lengkap berdasarkan dokumen
    """
    try:
        if not query:
            return "Error: Query tidak boleh kosong. Query harus berasal dari pertanyaan pengguna."

        print(f"\n[TOOL] Generating answer for query: {query}")

        # Parse JSON input
        try:
            json_data = json.loads(documents)
            if not isinstance(json_data, dict):
                raise ValueError("Input bukan JSON object yang valid")

            retrieved_docs = json_data.get("retrieved_docs_data", [])
            if not retrieved_docs:
                return "Mohon maaf, tidak ada informasi yang cukup dalam database kami untuk menjawab pertanyaan Anda. Silakan coba pertanyaan lain atau hubungi admin sistem untuk informasi lebih lanjut."

        except json.JSONDecodeError as e:
            print(f"[ERROR] Gagal parse JSON input: {str(e)}")
            return "Error: Format input dokumen tidak valid"

        # Format dokumen untuk prompt dengan metadata terstruktur
        formatted_docs = []
        for i, doc in enumerate(retrieved_docs):
            metadata = doc.get("metadata", {})
            status = metadata.get("status", "status tidak diketahui")
            jenis = metadata.get("jenis_peraturan", "")
            nomor = metadata.get("nomor_peraturan", "")
            tahun = metadata.get("tahun_peraturan", "")

            # Format referensi dokumen
            ref = (
                f"[{jenis} No. {nomor} Tahun {tahun}] ({status})"
                if all([jenis, nomor, tahun])
                else f"[Dokumen {i+1}] ({status})"
            )

            formatted_docs.append(
                f"Dokumen {i+1}:\n{doc.get('content', '')}\nReferensi: {ref}"
            )

        formatted_docs_str = "\n\n".join(formatted_docs)

        # Extract legal entities from documents
        legal_entities = set()
        for doc in retrieved_docs:
            content = doc.get("content", "")
            # Pattern sederhana untuk entitas hukum
            patterns = [
                r"(?:Undang-Undang|UU)(?:\s+Nomor|\s+No\.?)?(?:\s+\d+(?:/\d+)?)?(?:\s+Tahun\s+\d{4})?",
                r"(?:Peraturan\s+Pemerintah|PP)(?:\s+Nomor|\s+No\.?)?(?:\s+\d+(?:/\d+)?)?(?:\s+Tahun\s+\d{4})?",
                r"(?:Peraturan\s+Presiden|Perpres)(?:\s+Nomor|\s+No\.?)?(?:\s+\d+(?:/\d+)?)?(?:\s+Tahun\s+\d{4})?",
                r"(?:Peraturan\s+Menteri\s+Kesehatan|Permenkes)(?:\s+Nomor|\s+No\.?)?(?:\s+\d+(?:/\d+)?)?(?:\s+Tahun\s+\d{4})?",
                r"(?:Keputusan\s+Menteri\s+Kesehatan|Kepmenkes)(?:\s+Nomor|\s+No\.?)?(?:\s+\d+(?:/\d+)?)?(?:\s+Tahun\s+\d{4})?",
                r"Pasal\s+\d+(?:\s+[aA]yat\s+\d+)?",
            ]

            for pattern in patterns:
                matches = re.findall(pattern, content)
                legal_entities.update(
                    match.strip() for match in matches if match.strip()
                )

        entities_str = ", ".join(sorted(legal_entities))

        # Generate answer using LLM
        generator = ChatOpenAI(**MODELS["GENERATOR"])

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
   - "UU No. 17 Tahun 2023 (berlaku) menyatakan bahwa..."
   - "UU No. 36 Tahun 2009 (dicabut) pada saat itu mengatur bahwa..."
6. Prioritaskan menyimpulkan berdasarkan peraturan terbaru yang masih berlaku

# INSTRUKSI CONTEXTUAL CITATION
Untuk setiap pernyataan faktual, HARUS merujuk ke dokumen sumber dengan format:
[Nama Peraturan] (status) dimana status adalah "berlaku" atau "dicabut"

# INSTRUKSI PENTING
1. SELALU mulai jawaban Anda dengan menyebutkan sumber peraturan yang relevan termasuk statusnya
2. Jangan berhalusinasi atau membuat informasi yang tidak ada dalam dokumen
3. Jika informasi tidak cukup, nyatakan dengan jujur bahwa dokumen tidak memuat informasi yang cukup
4. JANGAN menggunakan format [Dok#X] dalam jawaban, selalu gunakan format [Nama Peraturan] (status)
5. Contoh format referensi yang benar:
   - "[UU No. 17 Tahun 2023] (berlaku)"
   - "[PP No. 61 Tahun 2014] (dicabut)"
   - "[UU No. 36 Tahun 2009] (dicabut)"

Pertanyaan pengguna: {query}

Berikut adalah dokumen-dokumen yang relevan dengan pertanyaan:

{formatted_docs_str}

Berdasarkan analisis entitas dalam dokumen, perhatikan entitas-entitas hukum berikut:
{entities_str}"""

        # Panggil LLM
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
4. Jika evaluasi menunjukkan "KURANG MEMADAI", sempurnakan query HANYA SEKALI dan cari lagi
5. Setelah pencarian kedua, langsung hasilkan jawaban berdasarkan dokumen yang ada
6. JANGAN melakukan evaluasi kedua setelah penyempurnaan query

ALUR KERJA ANDA:
1. Cari dokumen menggunakan query asli
2. Evaluasi dokumen yang ditemukan
3. Jika evaluasi menunjukkan "KURANG MEMADAI", sempurnakan query SEKALI dan cari lagi
4. Setelah pencarian kedua, langsung hasilkan jawaban
5. JANGAN melakukan evaluasi kedua

ATURAN PENTING:
1. Jawaban HARUS berdasarkan dokumen yang ditemukan
2. Gunakan Bahasa Indonesia formal dan terminologi hukum yang tepat
3. Hanya informasi dari database dokumen hukum kesehatan yang dapat digunakan
4. SELALU sertakan query asli saat memanggil evaluate_documents
5. HANYA BOLEH melakukan penyempurnaan query SEKALI
6. JANGAN melakukan evaluasi kedua setelah penyempurnaan query

PENANGANAN STATUS PERATURAN:
1. Prioritaskan informasi dari dokumen dengan status "berlaku"
2. Ketika memberikan jawaban, selalu sebutkan status peraturan yang dirujuk
3. Peraturan yang "dicabut" hanya digunakan untuk konteks historis

INGAT: 
1. Setelah penyempurnaan query, langsung hasilkan jawaban tanpa evaluasi kedua
2. JANGAN melakukan penyempurnaan query lebih dari sekali
3. Lebih baik memberikan jawaban berdasarkan dokumen yang ada daripada terus melakukan evaluasi"""

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

# Buat agent dengan parameter optimize= untuk performa
agent = create_openai_tools_agent(llm, tools, prompt)

# Create agent executor with faster execution settings
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,  # Kurangi logging untuk kecepatan
    handle_parsing_errors=True,
    return_intermediate_steps=True,
    max_execution_time=120,  # Naikkan batas waktu eksekusi maksimum (dalam detik)
    max_iterations=4,  # Batasi maksimal 4 iterasi (1 pencarian awal + 1 evaluasi + 1 penyempurnaan + 1 pencarian kedua)
)

# Tambahkan variabel untuk melacak jumlah penyempurnaan
refinement_count = 0

# ======================= API MODELS =======================


class AgenticRequest(BaseModel):
    query: str
    embedding_model: Literal["small", "large"] = "large"
    previous_responses: Optional[List[Dict[str, Any]]] = Field(default_factory=list)


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
    referenced_documents: List[Dict[str, Any]] = []
    agent_steps: Optional[List[StepInfo]] = None
    processing_time_ms: Optional[int] = None
    model_info: Dict[str, Any] = {}  # Tambahkan model_info


# ======================= API ENDPOINTS =======================


@app.post(
    "/api/chat",
    response_model=AgenticResponse,
    dependencies=[Depends(verify_api_key)],
)
async def agentic_chat(request: AgenticRequest):
    global refinement_count
    refinement_count = 0  # Reset counter setiap request baru

    try:
        start_time = time.time()
        print(f"\n[API] Processing chat request: {request.query}")

        # Dapatkan nama tabel dan model
        table_name = EMBEDDING_CONFIG[request.embedding_model]["table"]
        model_name = EMBEDDING_CONFIG[request.embedding_model]["model"]

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

        # Format chat history dari previous_responses
        chat_history = []
        if request.previous_responses:
            for response in request.previous_responses:
                if "query" in response and "answer" in response:
                    chat_history.extend(
                        [
                            HumanMessage(content=response["query"]),
                            AIMessage(content=response["answer"]),
                        ]
                    )

        # Execute agent with conversation memory
        result = agent_executor.invoke(
            {"input": request.query, "chat_history": chat_history}
        )

        # Extract the final answer from the agent result
        answer = result["output"]
        print(f"\n[API] Agent output (raw): {answer[:200]}...")

        # Post-processing untuk menghapus referensi internal [Dok#X]
        # Pola regex untuk mencari [Dok#angka] atau [Dok# angka]
        dok_pattern = r"(\[Dok#\s*\d+\])+"

        # Ganti pola yang ditemukan dengan string kosong
        cleaned_answer = re.sub(dok_pattern, "", answer).strip()

        # Bersihkan spasi ganda yang mungkin muncul setelah penghapusan
        cleaned_answer = re.sub(r"\s{2,}", " ", cleaned_answer)

        print(f"\n[API] Agent output (cleaned): {cleaned_answer[:200]}...")
        answer = cleaned_answer  # Gunakan jawaban yang sudah bersih

        # Kumpulkan semua dokumen dari langkah-langkah agent
        all_retrieved_documents_data = []
        processed_content_hashes = set()  # Untuk menghindari duplikasi dokumen

        for step in result["intermediate_steps"]:
            tool_name = step[0].tool
            if tool_name == "search_documents" and isinstance(step[1], str):
                try:
                    # Coba parse sebagai JSON
                    json_data = json.loads(step[1])
                    if (
                        isinstance(json_data, dict)
                        and "retrieved_docs_data" in json_data
                    ):
                        for doc_data in json_data["retrieved_docs_data"]:
                            # Cek duplikasi berdasarkan hash konten
                            content_hash = hash(doc_data.get("content", ""))
                            if content_hash not in processed_content_hashes:
                                all_retrieved_documents_data.append(doc_data)
                                processed_content_hashes.add(content_hash)
                except json.JSONDecodeError:
                    print(
                        f"[API] Peringatan: Output search_documents bukan JSON valid: {step[1][:100]}"
                    )

        # Gunakan dokumen yang sudah terstruktur untuk referenced_documents
        referenced_documents = all_retrieved_documents_data
        print(
            f"[API] Total referenced documents collected: {len(referenced_documents)}"
        )

        # Format agent steps
        agent_steps = []
        for step in result["intermediate_steps"]:
            if not hasattr(step[0], "tool") or not hasattr(step[0], "tool_input"):
                continue

            output_text = step[1] if isinstance(step[1], str) else str(step[1])

            step_info = StepInfo(
                tool=step[0].tool,
                tool_input=step[0].tool_input,
                tool_output=(
                    output_text[:500] + "..." if len(output_text) > 500 else output_text
                ),
            )
            agent_steps.append(step_info)

        # Calculate processing time
        end_time = time.time()
        processing_time_ms = int((end_time - start_time) * 1000)

        # Prepare model info
        model_info = {
            "model": model_name,
            "table": table_name,
            "dimensions": embedding_dimensions,
        }

        return AgenticResponse(
            answer=answer,
            referenced_documents=referenced_documents,
            agent_steps=agent_steps,
            processing_time_ms=processing_time_ms,
            model_info=model_info,
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
    uvicorn.run("multi_api:app", host="0.0.0.0", port=8000, reload=True)

# import basics
import os
from dotenv import load_dotenv
import re

# import langchain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

# import supabase
from supabase import create_client, Client

# load environment variables
load_dotenv()

# initialize supabase db
supabase_url: str = os.getenv("SUPABASE_URL")
supabase_key: str = os.getenv("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# initialize embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# ------------------------------------------------------------------------------------
# 1. Muat dokumen PDF
# ------------------------------------------------------------------------------------
loader = PyPDFDirectoryLoader("documents")
pages = loader.load()

# Gabungkan seluruh halaman agar urutannya terjaga
full_text = "\n".join([page.page_content for page in pages])

# ------------------------------------------------------------------------------------
# 2. Ekstraksi Judul Peraturan (setelah kata TENTANG)
# ------------------------------------------------------------------------------------
judul_peraturan_match = re.search(
    r"TENTANG\s+([A-Z0-9 ,\.\-()]+)", full_text, re.IGNORECASE
)
if judul_peraturan_match:
    judul_peraturan = f"Tentang {judul_peraturan_match.group(1).strip().title()}"
else:
    judul_peraturan = "Tentang -"

# ------------------------------------------------------------------------------------
# 3. Iterasi baris untuk mendeteksi BAB & PASAL, membentuk dokumen per Pasal
# ------------------------------------------------------------------------------------
jenis_peraturan = "UU"
nomor_peraturan = "4"
tahun_peraturan = "1953"

current_bab = ""
current_pasal = ""
current_lines: list[str] = []

pasal_docs: list[Document] = []

for raw_line in full_text.splitlines():
    line = raw_line.strip()

    # Lewati baris kosong atau nomor halaman
    if not line or re.match(r"^\d+\s*$", line):
        continue

    # Deteksi BAB (mis. "BAB IV KETENTUAN UMUM")
    bab_match = re.match(
        r"^BAB\s+([IVXLCDM]+)(?:\s*[-\.]*\s*(.*))?", line, re.IGNORECASE
    )
    if bab_match:
        roman, title = bab_match.group(1), bab_match.group(2) or ""
        title = title.title().strip()
        current_bab = f"Bab {roman.strip()}" + (f" - {title}" if title else "")
        continue  # lanjut ke baris berikutnya tanpa menambahkan ke konten pasal

    # Deteksi PASAL
    pasal_match = re.match(r"^Pasal\s+(\d+[A-Z]?)", line, re.IGNORECASE)
    if pasal_match:
        # Jika sebelumnya sudah ada pasal, simpan dokumennya terlebih dahulu
        if current_pasal and current_lines:
            content = "\n".join(current_lines).strip()
            metadata = {
                "jenis_peraturan": jenis_peraturan,
                "nomor_peraturan": nomor_peraturan,
                "tahun_peraturan": tahun_peraturan,
                "tipe_bagian": current_pasal,
                "bagian_dari": current_bab,
                "judul_peraturan": judul_peraturan,
            }
            pasal_docs.append(Document(page_content=content, metadata=metadata))

        # Mulai pasal baru
        current_pasal = f"Pasal {pasal_match.group(1)}"
        current_lines = []
        continue

    # Tambahkan baris ke konten pasal yang sedang diproses
    if current_pasal:
        # Singkirkan baris yang tidak relevan (Menimbang, Mengingat, dst.)
        if re.match(
            r"^(Menimbang|Mengingat|Memutuskan|Menetapkan)[:\s]", line, re.IGNORECASE
        ):
            continue
        current_lines.append(line)

# Simpan pasal terakhir
if current_pasal and current_lines:
    content = "\n".join(current_lines).strip()
    metadata = {
        "jenis_peraturan": jenis_peraturan,
        "nomor_peraturan": nomor_peraturan,
        "tahun_peraturan": tahun_peraturan,
        "tipe_bagian": current_pasal,
        "bagian_dari": current_bab,
        "judul_peraturan": judul_peraturan,
    }
    pasal_docs.append(Document(page_content=content, metadata=metadata))

# ------------------------------------------------------------------------------------
# 4. Simpan ke Vector Store
# ------------------------------------------------------------------------------------
docs = pasal_docs

# store chunks in vector store
vector_store = SupabaseVectorStore.from_documents(
    docs,
    embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
    chunk_size=500,
)

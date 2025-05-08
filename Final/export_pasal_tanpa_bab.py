import os
import re
import csv
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.schema import Document
import json

"""EXPORT PASAL TANPA BAB
Script ini membaca semua PDF di folder "data-fix", memecah konten per Pasal
untuk dokumen yang tidak memiliki struktur Bab yang jelas (seperti UU Darurat).
"""

# ---------------------------------------------------------------------------
# 1. Inisialisasi
# ---------------------------------------------------------------------------
load_dotenv()

# Sesuaikan jika folder PDF Anda berbeda
PDF_DIR = "documents"
OUTPUT_CSV = "pasal_output_tanpa_bab.csv"

# Placeholder, akan diisi otomatis setelah parsing header dokumen
JENIS_PERATURAN = "UNKNOWN"
NOMOR_PERATURAN = "-"
TAHUN_PERATURAN = "-"

# Add this at the beginning of the file, just after the imports
DEBUG_MODE = True  # Set ke True untuk debug atau False untuk produksi
AGGRESSIVE_MODE = True  # Mode agresif akan mendeteksi semua kemungkinan pasal
PRINT_RAW_TEXT = True  # Cetak teks mentah dari PDF untuk debugging

# ---------------------------------------------------------------------------
# 2. Muat dokumen PDF
# ---------------------------------------------------------------------------
print(f"Memuat dokumen dari {PDF_DIR}")
loader = PyPDFDirectoryLoader(PDF_DIR)
pages = loader.load()
full_text = "\n".join([page.page_content for page in pages])
print(f"Dokumen berhasil dimuat: {len(pages)} halaman")

# Print raw text untuk debugging jika diperlukan
if PRINT_RAW_TEXT:
    print("\n=== RAW TEXT (500 karakter pertama) ===")
    print(full_text[:500])
    print("...[terpotong]...")
    print("=== END RAW TEXT ===\n")

# ---------------------------------------------------------------------------
# 2.a. Ekstraksi Jenis, Nomor, Tahun Peraturan dari teks
# ---------------------------------------------------------------------------

# Coba deteksi jenis peraturan
if "UNDANG-UNDANG DARURAT" in full_text or "UU DARURAT" in full_text:
    JENIS_PERATURAN = "UU Darurat"
elif "PERATURAN PEMERINTAH" in full_text or "PP " in full_text:
    JENIS_PERATURAN = "PP"
elif "UNDANG-UNDANG" in full_text:
    JENIS_PERATURAN = "UU"

# Nomor Peraturan
nomor_match = re.search(r"NOMOR\s+([0-9A-Z]+)", full_text, re.IGNORECASE)
if nomor_match:
    NOMOR_PERATURAN = nomor_match.group(1)

# Tahun Peraturan
tahun_match = re.search(r"TAHUN\s+(\d{4})", full_text, re.IGNORECASE)
if tahun_match:
    TAHUN_PERATURAN = tahun_match.group(1)

# ---------------------------------------------------------------------------
# 3. Ekstraksi judul peraturan (setelah kata TENTANG)
# ---------------------------------------------------------------------------
judul_match = re.search(r"TENTANG\s+([A-Z0-9 ,\.\-()]+)", full_text, re.IGNORECASE)
if judul_match:
    JUDUL_PERATURAN = f"Tentang {judul_match.group(1).strip().title()}"
else:
    JUDUL_PERATURAN = "Tentang -"

print(f"Informasi dokumen:")
print(f"Jenis: {JENIS_PERATURAN}")
print(f"Nomor: {NOMOR_PERATURAN}")
print(f"Tahun: {TAHUN_PERATURAN}")
print(f"Judul: {JUDUL_PERATURAN}")


# ---------------------------------------------------------------------------
# Helper: Pembersihan Teks (Cleaning)
# ---------------------------------------------------------------------------
def clean_text(text):
    """
    Membersihkan teks dari kesalahan umum OCR pada dokumen perundangan
    """
    # Pembersihan ayat: "(21 " -> "(2)"
    cleaned = re.sub(r"\((\d)1\s", r"(\1)", text)

    # Pembersihan ayat: "(11 " -> "(1)"
    cleaned = re.sub(r"\(1{2,}\s", r"(1)", cleaned)

    # Perbaikan huruf 'l' yang terbaca sebagai angka '1' pada huruf kecil
    # contoh: "da1am" -> "dalam"
    cleaned = re.sub(r"([a-z])1([a-z])", r"\1l\2", cleaned)

    # Perbaikan angka 0 dan O
    cleaned = re.sub(r"([a-zA-Z])0([a-zA-Z])", r"\1O\2", cleaned)

    # Perbaikan spasi berlebih
    cleaned = re.sub(r"\s{2,}", " ", cleaned)

    # Perbaikan format ayat dengan spasi
    cleaned = re.sub(r"\(\s+(\d+)\s+\)", r"(\1)", cleaned)

    # Perbaikan jarak antara ayat dan konten
    cleaned = re.sub(r"\((\d+)\)(\S)", r"(\1) \2", cleaned)

    return cleaned


# Pembersihan khusus nomor pasal yang memiliki spasi di tengahnya
def clean_pasal_number(text):
    """
    Khusus menangani format "Pasal X Y" -> "Pasal XY"
    dan "Pasal X Y Z" -> "Pasal XYZ" karena spasi berlebih
    """
    # Normalisasi spasi antara "Pasal" dan nomor
    # PasalXXX -> Pasal XXX
    cleaned = re.sub(r"^(Pasal)(\d+)", r"\1 \2", text, flags=re.IGNORECASE)

    # Menangani format "Pasal 31 1" -> "Pasal 311"
    cleaned = re.sub(
        r"^(Pasal\s+)(\d+)(\s+)(\d+)$", r"\1\2\4", cleaned, flags=re.IGNORECASE
    )

    # Menangani format "Pasal 3 1 1" -> "Pasal 311"
    cleaned = re.sub(
        r"^(Pasal\s+)(\d+)(\s+)(\d+)(\s+)(\d+)$",
        r"\1\2\4\6",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Menangani format "Pasal 3 1 1 A" -> "Pasal 311A"
    cleaned = re.sub(
        r"^(Pasal\s+)(\d+)(\s+)(\d+)(\s+)(\d+)(\s*)([A-Z])?$",
        r"\1\2\4\6\8",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Perbaiki substitusi karakter yang sering tertukar pada nomor pasal
    # Ekstrak hanya bagian nomor saja
    pasal_num_match = re.search(r"^Pasal\s+([A-Za-z0-9]+)", cleaned, re.IGNORECASE)
    if pasal_num_match:
        pasal_num = pasal_num_match.group(1)
        fixed_num = fix_ocr_number(pasal_num)
        if pasal_num != fixed_num:
            cleaned = re.sub(
                r"^(Pasal\s+)[A-Za-z0-9]+",
                f"\\1{fixed_num}",
                cleaned,
                flags=re.IGNORECASE,
            )
            print(
                f"Memperbaiki substitusi karakter: Pasal {pasal_num} -> Pasal {fixed_num}"
            )

    return cleaned


# Fungsi untuk memperbaiki substitusi huruf-angka yang sering salah pada OCR
def fix_ocr_number(text):
    """
    Memperbaiki substitusi karakter pada nomor:
    - 'l' (huruf L kecil) -> '1' (angka satu)
    - 'L' (huruf L besar) -> '1' (angka satu)
    - 'I' (huruf I besar) -> '1' (angka satu)
    - 'O' (huruf O besar) -> '0' (angka nol)
    - 'T' (huruf T besar) -> '7' (angka tujuh) jika diikuti huruf
    """
    # Pendekatan berbasis karakter untuk menghindari masalah dengan look-behind regex
    result = ""

    # Perbaiki karakter per karakter
    for i, char in enumerate(text):
        # Kondisi untuk pemeriksaan jika digit sebelumnya
        prev_is_digit = i > 0 and (text[i - 1].isdigit() or text[i - 1] in "10")
        # Kondisi untuk pemeriksaan jika digit berikutnya
        next_is_digit = i < len(text) - 1 and (
            text[i + 1].isdigit() or text[i + 1] in "10"
        )

        # Aturan substitusi yang diperluas
        if char in "lL" and (
            i == 0 or prev_is_digit or next_is_digit or i == len(text) - 1
        ):
            # 'l' (huruf L kecil) atau 'L' (huruf L besar) -> '1' (angka satu)
            result += "1"
        elif char == "I" and (
            i == 0 or prev_is_digit or next_is_digit or i == len(text) - 1
        ):
            # 'I' (huruf I besar) -> '1' (angka satu)
            result += "1"
        elif char == "O" and (
            i == 0 or prev_is_digit or next_is_digit or i == len(text) - 1
        ):
            # 'O' (huruf O besar) -> '0' (angka nol)
            result += "0"
        elif char == "T" and (
            prev_is_digit or i == 0 or i == len(text) - 1 or next_is_digit
        ):
            # 'T' (huruf T besar) -> '7' (angka tujuh) - lebih permisif
            result += "7"
        else:
            # Karakter lain tidak diubah
            result += char

    return result


# ---------------------------------------------------------------------------
# 4. Parsing per Pasal tanpa mengandalkan struktur Bab
# ---------------------------------------------------------------------------
current_pasal = ""
current_lines: list[str] = []
last_pasal_number = 0  # menyimpan nomor pasal terakhir untuk validasi

pasal_docs: list[Document] = []


# Helper untuk mengambil nomor pasal sebagai integer dari string "Pasal X"
def extract_pasal_number(pasal_str: str) -> int:
    # Bersihkan nomor pasal dulu
    pasal_str = clean_pasal_number(pasal_str)
    match = re.search(r"Pasal\s+(\d+)[A-Z]?", pasal_str, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0


def is_valid_new_pasal(pasal_number: int, last_number: int) -> bool:
    """
    Memeriksa apakah nomor pasal valid sebagai pasal baru berdasarkan urutan
    Memberikan beberapa toleransi untuk pasal yang mungkin terlewat
    """
    # Tidak ada pasal sebelumnya, ini pasal pertama
    if last_number == 0:
        return True

    # Urutan pasal selanjutnya
    if pasal_number > last_number:
        # Toleransi untuk pasal yang terlewat, maksimum 10 pasal terlewat
        if pasal_number - last_number <= 10:
            return True
        else:
            print(
                f"Peringatan: Terjadi lompatan besar dari Pasal {last_number} ke {pasal_number}"
            )
            return True  # Tetap terima meskipun ada lompatan besar

    # Pasal sebelumnya atau referensi
    return False


# Flag untuk menandai kita sudah berada di bagian utama dokumen, bukan bagian preambule
in_main_content = False
debug_lines_after_memutuskan = []  # Untuk debugging

for raw_line in full_text.splitlines():
    line = raw_line.strip()

    # Bersihkan teks dari kesalahan OCR
    line = clean_text(line)

    # Perbaiki kasus "Pasa1" menjadi "Pasal" (huruf L terbaca sebagai angka 1)
    if re.search(r"Pasa1\s+\d+", line, re.IGNORECASE):
        original = line
        line = re.sub(r"Pasa1", "Pasal", line, flags=re.IGNORECASE)
        print(f"Memperbaiki OCR Pasa1: {original} -> {line}")

        # Jika baris hanya berisi "Pasa1 X", ini pasti awal pasal baru
        if re.match(r"^Pasal\s+\d+$", line, re.IGNORECASE):
            print(f"DETEKSI LANGSUNG: {line} sebagai awal pasal baru")
            # Simpan pasal lama jika ada
            if current_pasal and current_lines and in_main_content:
                content = "\n".join(current_lines).strip()
                metadata = {
                    "jenis_peraturan": JENIS_PERATURAN,
                    "nomor_peraturan": NOMOR_PERATURAN,
                    "tahun_peraturan": TAHUN_PERATURAN,
                    "tipe_bagian": current_pasal,
                    "bagian_dari": "",  # Kosongkan karena tidak ada bab
                    "judul_peraturan": JUDUL_PERATURAN,
                }
                pasal_docs.append(Document(page_content=content, metadata=metadata))

            # Ekstrak nomor pasal
            pasal_num_match = re.search(r"Pasal\s+(\d+)", line, re.IGNORECASE)
            if pasal_num_match:
                pasal_number_str = pasal_num_match.group(1)
                try:
                    pasal_number = int(pasal_number_str)
                    # Mulai pasal baru
                    current_pasal = f"Pasal {pasal_number_str}"
                    current_lines = []
                    last_pasal_number = pasal_number
                    print(f"DETEKSI BERHASIL: {current_pasal}")
                    in_main_content = True  # Pastikan kita di bagian utama dokumen
                    continue
                except ValueError:
                    print(f"Gagal konversi nomor pasal: {pasal_number_str}")

    # Perbaiki kasus spasi terpisah seperti "P asal", "Pa sal", dsb
    if re.match(r"^P\s*a\s*s\s*a\s*l\s+\d+", line, re.IGNORECASE):
        original = line
        line = re.sub(r"^P\s*a\s*s\s*a\s*l", "Pasal", line, flags=re.IGNORECASE)
        print(f"Memperbaiki spasi dalam Pasal: {original} -> {line}")

    # Perbaiki kasus typo umum seperti "Pasai", "Pasasl", "Pasrl", dll
    if re.match(r"^Pasa[irs]\s+\d+", line, re.IGNORECASE) or re.match(
        r"^Pasas?l\s+\d+", line, re.IGNORECASE
    ):
        original = line
        line = re.sub(r"^Pasa[irs]", "Pasal", line, flags=re.IGNORECASE)
        line = re.sub(r"^Pasas?l", "Pasal", line, flags=re.IGNORECASE)
        print(f"Memperbaiki typo dalam Pasal: {original} -> {line}")

    # Perbaiki kasus format "Artikel N" atau "Article N" (terjemahan dari dokumen asing)
    if re.match(r"^Ar[tl]i[ck][el][el]?\s+\d+", line, re.IGNORECASE):
        original = line
        line = re.sub(r"^Ar[tl]i[ck][el][el]?", "Pasal", line, flags=re.IGNORECASE)
        print(f"Memperbaiki Artikel/Article: {original} -> {line}")

    if not line or re.match(r"^\d+\s*$", line):
        continue

    # Mendeteksi bagian utama dokumen untuk memastikan kita tidak mengambil referensi di preambule
    if (
        re.match(r"^MEMUTUSKAN|^MENETAPKAN", line, re.IGNORECASE)
        and not in_main_content
    ):
        in_main_content = True
        print(f"Menemukan bagian utama dokumen: '{line}'")
        # Mulai mengumpulkan 30 baris setelah MEMUTUSKAN untuk debugging
        debug_lines_after_memutuskan = []
        continue

    # Dalam mode debug, kita selalu anggap dalam main content untuk mendeteksi semua pasal
    if DEBUG_MODE and not in_main_content:
        in_main_content = True
        print(f"DEBUG: Mengaktifkan mode main content untuk deteksi semua pasal")

    # Kumpulkan 30 baris pertama setelah MEMUTUSKAN untuk debugging
    if in_main_content and len(debug_lines_after_memutuskan) < 30:
        debug_lines_after_memutuskan.append(line)
        # Cetak setiap baris untuk debugging
        print(f"DEBUG [{len(debug_lines_after_memutuskan)}]: {line}")

    # Cek apakah ada format "PasalXXX" tanpa spasi
    if re.match(r"^Pasal[A-Za-z0-9]+\s*$", line, re.IGNORECASE) and not re.match(
        r"^Pasal\s+", line, re.IGNORECASE
    ):
        # Normalisasi format "PasalXXX" -> "Pasal XXX"
        nomor_asli = line
        line = re.sub(r"^(Pasal)([A-Za-z0-9]+)", r"\1 \2", line, flags=re.IGNORECASE)
        print(f"Memperbaiki format pasal tanpa spasi: {nomor_asli} -> {line}")

    # Cek kemungkinan pasal dengan karakter OCR yang salah ('I', 'O', 'l', 'L', 'T')
    ocr_pasal_match = re.match(r"^Pasal\s+([A-Za-z0-9]+)\s*$", line, re.IGNORECASE)
    if ocr_pasal_match and any(c in ocr_pasal_match.group(1) for c in "IOlLT"):
        # Nomor pasal berisi huruf, coba perbaiki
        nomor_asli = ocr_pasal_match.group(1)
        nomor_fixed = fix_ocr_number(nomor_asli)
        if nomor_fixed != nomor_asli and re.match(r"^\d+[A-Z]?$", nomor_fixed):
            line = f"Pasal {nomor_fixed}"
            print(f"Memperbaiki nomor pasal OCR: Pasal {nomor_asli} -> {line}")

    # Deteksi PASAL - pertama cek format dengan spasi dalam nomor, lalu cek format normal
    pasal_with_space_match = re.match(
        r"^Pasal\s+(\d+)\s+(\d+)(\s+\d+)?(\s*[A-Z]?)?\s*$", line, re.IGNORECASE
    )
    if pasal_with_space_match:
        # Format "Pasal 31 1" -> ubah jadi "Pasal 311" dulu
        line = clean_pasal_number(line)
        print(f"Memperbaiki nomor pasal dengan spasi: {raw_line.strip()} -> {line}")

    # Deteksi PASAL dengan format normal
    pasal_match = re.match(r"^Pasal\s+(\d+[A-Z]?)\s*$", line, re.IGNORECASE)

    # Deteksi KHUSUS "Pasa1 X" yang muncul sebagai baris terisolasi
    if not pasal_match and re.match(
        r"^[\s\.]*Pasa1\s+\d+[\s\.]*$", line, re.IGNORECASE
    ):
        original = line
        fixed_line = re.sub(r"Pasa1", "Pasal", line, flags=re.IGNORECASE)
        pasal_num = re.search(r"Pasa1\s+(\d+)", line, re.IGNORECASE)
        if pasal_num:
            pasal_match = re.match(
                r"^Pasal\s+(\d+)\s*$", f"Pasal {pasal_num.group(1)}", re.IGNORECASE
            )
            print(
                f"KASUS KHUSUS PASA1: {original} -> {pasal_match.group(0) if pasal_match else 'GAGAL'}"
            )

    # Mode agresif - jika ada "Pasal X" di mana saja dalam baris, anggap sebagai awal pasal baru
    if AGGRESSIVE_MODE and not pasal_match and not pasal_with_space_match:
        pasal_in_text = re.search(
            r"(?<!\w)(Pasa[l1]|P\.|Ps\.|Pasal|Pas\.)\s+(\d+)(?!\w)", line, re.IGNORECASE
        )
        if pasal_in_text:
            try:
                pasal_number = int(pasal_in_text.group(2))
                if is_valid_new_pasal(pasal_number, last_pasal_number):
                    # Simpan pasal lama jika ada
                    if current_pasal and current_lines and in_main_content:
                        content = "\n".join(current_lines).strip()
                        metadata = {
                            "jenis_peraturan": JENIS_PERATURAN,
                            "nomor_peraturan": NOMOR_PERATURAN,
                            "tahun_peraturan": TAHUN_PERATURAN,
                            "tipe_bagian": current_pasal,
                            "bagian_dari": "",  # Kosongkan karena tidak ada bab
                            "judul_peraturan": JUDUL_PERATURAN,
                        }
                        pasal_docs.append(
                            Document(page_content=content, metadata=metadata)
                        )

                    # Mulai pasal baru
                    current_pasal = f"Pasal {pasal_number}"
                    current_lines = []
                    last_pasal_number = pasal_number
                    print(f"MODE AGRESIF: Mendeteksi {current_pasal} dari '{line}'")

                    # Extract remaining content in the line after the pasal
                    content_after_pasal = re.sub(
                        r".*?(?<!\w)(Pasa[l1]|P\.|Ps\.|Pasal|Pas\.)\s+\d+(?!\w)",
                        "",
                        line,
                        re.IGNORECASE,
                    )
                    if content_after_pasal.strip():
                        current_lines.append(content_after_pasal.strip())

                    continue
            except ValueError:
                pass  # Ignore if not a valid number

    # Cek juga format "Pasal X" yang muncul di tengah dokumen tetapi di baris sendiri
    if (
        not pasal_match
        and line.strip()
        and re.match(r"^[\s\.]*(Pasal\s+\d+[A-Z]?)[\s\.]*$", line, re.IGNORECASE)
    ):
        pasal_match = re.search(r"(Pasal\s+\d+[A-Z]?)", line, re.IGNORECASE)
        if pasal_match:
            nomor_pasal = re.search(
                r"Pasal\s+(\d+[A-Z]?)", pasal_match.group(1), re.IGNORECASE
            )
            if nomor_pasal:
                pasal_match = re.match(
                    r"^Pasal\s+(\d+[A-Z]?)\s*$",
                    f"Pasal {nomor_pasal.group(1)}",
                    re.IGNORECASE,
                )
                print(f"DETEKSI PASAL TENGAH DOKUMEN: {line} -> {pasal_match.group(0)}")

    # Deteksi format pasal alternatif yang mungkin ada di UU Darurat
    alt_pasal_match = (
        re.match(r"^Ps\.\s*(\d+[A-Z]?)\s*$", line, re.IGNORECASE)
        or re.match(r"^Ps\s+(\d+[A-Z]?)\s*$", line, re.IGNORECASE)
        or re.match(r"^Pasal\s+ke-(\d+[A-Z]?)\s*$", line, re.IGNORECASE)
        # Tambahkan format alternatif baru
        or re.match(r"^Pasal\s+(\d+[A-Z]?)\.?$", line, re.IGNORECASE)
        or re.match(r"^Pasal\s+(\d+[A-Z]?)[,:;].*$", line, re.IGNORECASE)
        # Format khusus UU Darurat
        or re.match(
            r"^P\.?\s*(\d+[A-Z]?)\s*$", line, re.IGNORECASE
        )  # "P. 1" atau "P 1"
        or re.match(
            r"^P(asal|s)?\.?\s*(\d+[A-Z]?)[,:;].*$", line, re.IGNORECASE
        )  # "P. 1:" atau "P 1:"
        or re.match(r"^Pas[.,]\s*(\d+[A-Z]?)\s*$", line, re.IGNORECASE)  # "Pas. 1"
        or re.match(r"^Pas[.,]\s*(\d+[A-Z]?)[,:;].*$", line, re.IGNORECASE)  # "Pas. 1:"
        or re.match(
            r"^Psl\.?\s*(\d+[A-Z]?)\s*$", line, re.IGNORECASE
        )  # "Psl. 1" atau "Psl 1"
        or re.match(
            r"^Pasl\.?\s*(\d+[A-Z]?)\s*$", line, re.IGNORECASE
        )  # "Pasl. 1" atau "Pasl 1"
        or re.match(
            r"^Fasa[l1]\s+(\d+[A-Z]?)\s*$", line, re.IGNORECASE
        )  # "Fasal 1" (error OCR)
    )

    if pasal_match or alt_pasal_match:
        # Dapatkan format pasal yang benar
        pasal_number_str = None

        if pasal_match:
            pasal_number_str = pasal_match.group(1)
        elif alt_pasal_match:
            # Ambil grup hasil match yang sesuai
            for i in range(1, min(3, len(alt_pasal_match.groups()) + 1)):
                if (
                    alt_pasal_match.group(i) is not None
                    and alt_pasal_match.group(i).strip()
                ):
                    pasal_number_str = alt_pasal_match.group(i)
                    break

        if not pasal_number_str:
            continue

        try:
            pasal_number = int(re.sub(r"[A-Z]", "", pasal_number_str))

            # Hanya proses pasal jika kita sudah berada di bagian utama dokumen
            if in_main_content:
                # Validasi urutan pasal
                if is_valid_new_pasal(pasal_number, last_pasal_number):
                    # Simpan pasal lama
                    if current_pasal and current_lines:
                        content = "\n".join(current_lines).strip()
                        metadata = {
                            "jenis_peraturan": JENIS_PERATURAN,
                            "nomor_peraturan": NOMOR_PERATURAN,
                            "tahun_peraturan": TAHUN_PERATURAN,
                            "tipe_bagian": current_pasal,
                            "bagian_dari": "",  # Kosongkan karena tidak ada bab
                            "judul_peraturan": JUDUL_PERATURAN,
                        }
                        pasal_docs.append(
                            Document(page_content=content, metadata=metadata)
                        )

                    # Mulai pasal baru
                    if pasal_match:
                        current_pasal = f"Pasal {pasal_match.group(1)}"
                        print(f"Ditemukan format normal: {current_pasal}")
                    else:
                        current_pasal = f"Pasal {pasal_number_str}"
                        print(f"Ditemukan format alternatif: {line} -> {current_pasal}")
                    current_lines = []
                    last_pasal_number = pasal_number
                    print(f"Ditemukan {current_pasal}")

                    continue
                else:
                    print(f"Mengabaikan referensi pasal di baris: {line}")
            else:
                print(f"Mengabaikan pasal di bagian preambule: {line}")
        except ValueError:
            print(f"Gagal memproses nomor pasal: {pasal_number_str}")
            # Tetap tambahkan ke baris konten
            if current_pasal:
                current_lines.append(line)
            continue

    # Lewati header/top-page atau kata kunci yang tidak penting
    if re.match(r"^(PRESIDEN|REPUBLIK INDONESIA)$", line, re.IGNORECASE):
        continue
    if re.match(r"^-\s*\d+\s*-?$", line):  # contoh: "- 3 -"
        continue
    if not in_main_content and re.match(
        r"^(Menimbang|Mengingat|Memutuskan|Menetapkan)[:\s]", line, re.IGNORECASE
    ):
        continue

    # Deteksi footer, tapi lanjutkan proses jika belum ada pasal yang ditemukan
    if re.match(r"^(Disahkan|Ditetapkan)\s+di", line, re.IGNORECASE):
        print(f"Menemukan footer dokumen: '{line}'")
        # Jika belum ada pasal yang ditemukan, jangan hentikan proses
        if last_pasal_number > 0:
            # Selesai memproses dokumen, abaikan footer
            break
        else:
            print(
                "Footer dokumen ditemukan sebelum menemukan pasal. Melanjutkan proses..."
            )

    if current_pasal and in_main_content:
        current_lines.append(line)

# Cetak rangkuman debugging
print("\n=== DEBUGGING INFO ===")
print(
    f"Total baris setelah MEMUTUSKAN yang diambil: {len(debug_lines_after_memutuskan)}"
)
if debug_lines_after_memutuskan:
    print(
        "30 baris pertama setelah MEMUTUSKAN (lihat di sini format pasal yang digunakan):"
    )
    for i, line in enumerate(debug_lines_after_memutuskan[:30]):
        print(f"{i+1}. {line}")
else:
    print("PERINGATAN: Tidak ada konten yang terdeteksi setelah MEMUTUSKAN!")
print(f"Total pasal yang berhasil dideteksi: {len(pasal_docs)}")
print("=== END DEBUGGING INFO ===\n")

# Final cleanup - bersihkan semua konten pasal
clean_lines = []
for line in current_lines:
    clean_lines.append(clean_text(line))
current_lines = clean_lines

# Simpan pasal terakhir
if current_pasal and current_lines:
    content = "\n".join(current_lines).strip()
    metadata = {
        "jenis_peraturan": JENIS_PERATURAN,
        "nomor_peraturan": NOMOR_PERATURAN,
        "tahun_peraturan": TAHUN_PERATURAN,
        "tipe_bagian": current_pasal,
        "bagian_dari": "",  # Kosongkan karena tidak ada bab
        "judul_peraturan": JUDUL_PERATURAN,
    }
    pasal_docs.append(Document(page_content=content, metadata=metadata))

# ---------------------------------------------------------------------------
# 5. Tulis ke CSV (2 kolom saja: metadata & content)
# ---------------------------------------------------------------------------
fieldnames = ["metadata", "content"]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for doc in pasal_docs:
        # Bersihkan konten pasal sebelum disimpan ke CSV
        doc.page_content = clean_text(doc.page_content)
        writer.writerow(
            {
                "metadata": json.dumps(doc.metadata, ensure_ascii=False),
                "content": doc.page_content,
            }
        )

print(f"Berhasil mengekspor {len(pasal_docs)} pasal ke {OUTPUT_CSV}")

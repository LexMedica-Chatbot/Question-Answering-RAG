import os
import re
import csv
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.schema import Document
import json

"""EXPORT PASAL TANPA BAB
Script ini membaca semua PDF di folder "documents", memecah konten per Pasal
untuk dokumen yang tidak memiliki struktur Bab yang jelas (seperti UU Darurat).
"""

# ---------------------------------------------------------------------------
# 1. Inisialisasi
# ---------------------------------------------------------------------------
load_dotenv()

# Sesuaikan jika folder PDF Anda berbeda
PDF_DIR = "documents"
OUTPUT_CSV = "output/pasal_output_tanpa_bab.csv"

# Placeholder, akan diisi otomatis setelah parsing header dokumen
JENIS_PERATURAN = "UNKNOWN"
NOMOR_PERATURAN = "-"
TAHUN_PERATURAN = "-"

# Add this at the beginning of the file, just after the imports
DEBUG_MODE = True  # Set ke True untuk debug atau False untuk produksi
AGGRESSIVE_MODE = True  # Mode agresif akan mendeteksi semua kemungkinan pasal
PRINT_RAW_TEXT = True  # Cetak teks mentah dari PDF untuk debugging

# Nonaktifkan mode agresif ONLY di preambule
AGGRESSIVE_MODE_PREAMBULE = False  # Jangan deteksi agresif di preambule

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
    # Menangani kasus pasal dengan titik di akhir seperti "Pasal 1."
    cleaned = re.sub(
        r"^(Pasal\s+\d+)\.$",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )

    # Pembersihan ayat: "(21 " -> "(2)"
    cleaned = re.sub(r"\((\d)1\s", r"(\1)", cleaned)

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

    # Hapus footer dokumen
    # Hapus pola "PRESIDEN\nREPUBLIK INDONESIA\n- X -"
    cleaned = re.sub(
        r"PRESIDEN\s*\nREPUBLIK INDONESIA\s*\n-\s*\d+\s*-",
        "",
        cleaned,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    # Untuk case yang terpisah
    cleaned = re.sub(r"PRESIDEN\s*$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(
        r"REPUBLIK INDONESIA\s*$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE
    )
    cleaned = re.sub(
        r"^-\s*\d+\s*-\s*$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE
    )

    # Hapus tanda halaman berupa "..."
    cleaned = re.sub(r"\s*\.\.\.\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*\.\.\.\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\.\.\.\s*$", "", cleaned, flags=re.MULTILINE)

    # Hapus tanda yang diikuti elipsis "e. Dokter …"
    cleaned = re.sub(r"([a-z])\s*…\s*$", r"\1", cleaned, flags=re.MULTILINE)

    # Hapus tanda elipsis di awal baris
    cleaned = re.sub(r"^\s*…\s*", "", cleaned, flags=re.MULTILINE)

    # Hapus tanda "Agar …" di akhir pasal
    cleaned = re.sub(r"Agar\s*…\s*$", "", cleaned, flags=re.MULTILINE)

    # Hapus nomor halaman yang muncul di akhir teks (seperti angka 500, 501, dll)
    cleaned = re.sub(r"\s+\d{3}$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s+\d{3}\s+", " ", cleaned)

    # Menghapus penggandaan baris, misalnya:
    # "e. Dokter"
    # "e. Dokter Penguji..."
    # cari pola di mana baris berikutnya dimulai dengan teks yang sama
    lines = cleaned.splitlines()
    filtered_lines = []
    i = 0
    while i < len(lines):
        if i < len(lines) - 1:
            current_line = lines[i].strip()
            next_line = lines[i + 1].strip()

            # Jika baris saat ini adalah awalan dari baris berikutnya, skip baris saat ini
            if current_line and next_line.startswith(current_line):
                i += 1
                continue

        filtered_lines.append(lines[i])
        i += 1

    cleaned = "\n".join(filtered_lines)

    # Hapus spasi berlebih setelah penghapusan footer
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    cleaned = cleaned.strip()

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
current_lines = []
last_pasal_number = 0  # menyimpan nomor pasal terakhir untuk validasi

pasal_docs = []
processed_pasal_numbers = set()  # Melacak nomor pasal yang sudah diproses

# Flag untuk menandai kita sudah berada di bagian utama dokumen, bukan bagian preambule
in_main_content = False
in_preambule = True  # Kita anggap dokumen dimulai dengan preambule
debug_lines_after_memutuskan = []  # Untuk debugging

# Deteksi awal bagian utama dokumen (setelah preambule)
pasal1_index = -1
memutuskan_index = -1
menetapkan_index = -1

lines = full_text.splitlines()
for i, line in enumerate(lines):
    line = line.strip()
    if re.match(r"^(MEMUTUSKAN|Memutuskan)[\s:;]*$", line, re.IGNORECASE):
        memutuskan_index = i
    elif re.match(r"^(MENETAPKAN|Menetapkan)[\s:;]*$", line, re.IGNORECASE):
        menetapkan_index = i
    elif re.match(r"^Pasal\s+1\s*\.?\s*$", line, re.IGNORECASE):
        pasal1_index = i
        break

# Tentukan di mana bagian utama dokumen dimulai
if pasal1_index > 0:
    preambule_end_index = pasal1_index
    print(f"Dokumen mulai dari Pasal 1 pada baris {pasal1_index}")
elif menetapkan_index > 0:
    preambule_end_index = menetapkan_index + 1  # Baris setelah MENETAPKAN
    print(f"Dokumen mulai setelah MENETAPKAN pada baris {menetapkan_index}")
elif memutuskan_index > 0:
    preambule_end_index = memutuskan_index + 1  # Baris setelah MEMUTUSKAN
    print(f"Dokumen mulai setelah MEMUTUSKAN pada baris {memutuskan_index}")
else:
    preambule_end_index = 0  # Tidak ada preambule yang terdeteksi
    print("Tidak dapat mendeteksi akhir preambule, memproses seluruh dokumen")


# Helper untuk mengambil nomor pasal sebagai integer dari string "Pasal X"
def extract_pasal_number(pasal_str):
    # Bersihkan nomor pasal dulu
    pasal_str = clean_pasal_number(pasal_str)
    match = re.search(r"Pasal\s+(\d+)[A-Z]?", pasal_str, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0


def is_valid_new_pasal(pasal_number, last_number):
    """
    Memeriksa apakah nomor pasal valid sebagai pasal baru berdasarkan urutan
    Memberikan beberapa toleransi untuk pasal yang mungkin terlewat
    """
    # Tidak ada pasal sebelumnya, ini pasal pertama
    if last_number == 0:
        return True

    # Periksa jika nomor pasal sudah diproses sebelumnya
    if pasal_number in processed_pasal_numbers:
        print(f"Mengabaikan pasal {pasal_number} karena sudah diproses sebelumnya")
        return False

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


# Jika ingin selalu mendeteksi pasal meskipun dalam preambule
if DEBUG_MODE:
    in_main_content = True
    print("DEBUG: Mengaktifkan mode main content untuk deteksi semua pasal")

# Loop untuk parsing
line_number = 0
for raw_line in full_text.splitlines():
    line_number += 1
    line = raw_line.strip()

    # Bersihkan teks dari kesalahan OCR
    line = clean_text(line)

    if not line or re.match(r"^\d+\s*$", line):
        continue

    # Jika kita belum melewati preambule dan tidak dalam mode debug, lewati baris ini
    if not DEBUG_MODE and line_number < preambule_end_index:
        continue

    # Jika kita baru saja melewati preambule, aktifkan flag in_main_content
    if line_number == preambule_end_index:
        in_main_content = True
        print(f"Mulai memproses konten utama pada baris {line_number}: '{line}'")

    # Kumpulkan 30 baris pertama konten utama untuk debugging
    if in_main_content and len(debug_lines_after_memutuskan) < 30:
        debug_lines_after_memutuskan.append(line)
        print(f"DEBUG [{len(debug_lines_after_memutuskan)}]: {line}")

    # Perbaiki kasus "Pasa1" menjadi "Pasal" (huruf L terbaca sebagai angka 1)
    if re.search(r"Pasa1\s+\d+", line, re.IGNORECASE):
        original = line
        line = re.sub(r"Pasa1", "Pasal", line, flags=re.IGNORECASE)
        print(f"Memperbaiki OCR Pasa1: {original} -> {line}")

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

    # Deteksi format pasal alternatif yang mungkin ada di UU Darurat
    alt_pasal_match = (
        re.match(r"^Ps\.\s*(\d+[A-Z]?)\s*$", line, re.IGNORECASE)
        or re.match(r"^Ps\s+(\d+[A-Z]?)\s*$", line, re.IGNORECASE)
        or re.match(r"^Pasal\s+ke-(\d+[A-Z]?)\s*$", line, re.IGNORECASE)
        # Tambahkan format alternatif baru
        or re.match(r"^Pasal\s+(\d+[A-Z]?)\.?$", line, re.IGNORECASE)
        or re.match(r"^Pasal\s+(\d+[A-Z]?)[,:;].*$", line, re.IGNORECASE)
        # Format untuk pasal dengan ayat di baris yang sama
        or re.match(r"^Pasal\s+(\d+[A-Z]?)\s*\(\d+\).*$", line, re.IGNORECASE)
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

    # KHUSUS: Mode agresif untuk mencari "Pasal X" dalam baris
    if AGGRESSIVE_MODE and not pasal_match and not alt_pasal_match:
        # Jika dalam preambule dan AGGRESSIVE_MODE_PREAMBULE dinonaktifkan, skip
        is_preambule_line = (
            line_number < preambule_end_index and not AGGRESSIVE_MODE_PREAMBULE
        )

        # Jika dalam preambule dan deteksi agresif dinonaktifkan untuk preambule, lanjutkan
        if is_preambule_line:
            if current_pasal and in_main_content:
                current_lines.append(line)
            continue

        pasal_in_text = re.search(
            r"(?<!\w)(Pasa[l1]|P\.|Ps\.|Pasal|Pas\.)\s+(\d+)(?!\w)", line, re.IGNORECASE
        )
        if pasal_in_text:
            try:
                pasal_number = int(pasal_in_text.group(2))

                # Cek apakah ini adalah referensi pasal dan bukan pasal baru
                is_reference = re.search(
                    r"(dalam|tersebut|dimaksud) (pada |dalam )?pasal", line.lower()
                )
                if is_reference:
                    print(f"Mengabaikan referensi pasal dalam teks: '{line}'")
                    if current_pasal and in_main_content:
                        current_lines.append(line)
                    continue

                # Hanya tangani jika ini pasal 1-9, untuk mencegah pengenalan referensi pasal sebagai pasal baru
                if (
                    pasal_number <= 9
                    or is_valid_new_pasal(pasal_number, last_pasal_number)
                ) and pasal_number not in processed_pasal_numbers:
                    # Deteksi ini adalah pasalnya, bukan hanya referensi
                    if in_main_content:
                        # Simpan pasal lama jika ada
                        if current_pasal and current_lines:
                            content = "\n".join(current_lines).strip()
                            metadata = {
                                "jenis_peraturan": JENIS_PERATURAN,
                                "nomor_peraturan": NOMOR_PERATURAN,
                                "tahun_peraturan": TAHUN_PERATURAN,
                                "tipe_bagian": current_pasal,
                                "bagian_dari": "",  # Kosongkan karena tidak ada bab
                                "judul_peraturan": JUDUL_PERATURAN,
                                "status": "berlaku",
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

                        # Tambahkan ke pasal yang sudah diproses
                        processed_pasal_numbers.add(pasal_number)

                        continue
                    else:
                        print(f"Mengabaikan pasal di preambule: {line}")
                else:
                    print(f"Mengabaikan referensi pasal: {line}")
            except ValueError:
                pass  # Ignore if not a valid number

    # Deteksi pasal reguler
    if pasal_match or alt_pasal_match:
        pasal_number_str = None

        if pasal_match:
            pasal_number_str = pasal_match.group(1)
        elif alt_pasal_match:
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

            # Deteksi khusus untuk baris "Pasal 1" yang terpisah (awal dokumen)
            standalone_pasal = re.match(r"^Pasal\s+\d+\s*$", line, re.IGNORECASE)

            # Jika ini adalah baris dengan format "Pasal 1" sampai "Pasal 9" yang standalone,
            # maka ini pasti pasal utama bukan referensi
            if (
                standalone_pasal
                and 1 <= pasal_number <= 9
                and pasal_number not in processed_pasal_numbers
            ):
                in_main_content = True  # Pastikan ini diproses sebagai konten utama

                # Simpan pasal sebelumnya jika ada
                if current_pasal and current_lines:
                    content = "\n".join(current_lines).strip()
                    metadata = {
                        "jenis_peraturan": JENIS_PERATURAN,
                        "nomor_peraturan": NOMOR_PERATURAN,
                        "tahun_peraturan": TAHUN_PERATURAN,
                        "tipe_bagian": current_pasal,
                        "bagian_dari": "",  # Kosongkan karena tidak ada bab
                        "judul_peraturan": JUDUL_PERATURAN,
                        "status": "berlaku",
                    }
                    pasal_docs.append(Document(page_content=content, metadata=metadata))

                # Mulai pasal baru
                current_pasal = f"Pasal {pasal_number}"
                current_lines = []
                last_pasal_number = pasal_number
                print(f"Mendeteksi pasal mandiri: {current_pasal}")

                # Tambahkan ke pasal yang sudah diproses
                processed_pasal_numbers.add(pasal_number)

                continue

            # Validasi nomor pasal dan simpan pasal sebelumnya jika valid
            if in_main_content:
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
                            "status": "berlaku",
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

                    # Tambahkan ke pasal yang sudah diproses
                    processed_pasal_numbers.add(pasal_number)

                    continue
                else:
                    print(f"Mengabaikan referensi pasal di baris: {line}")
            else:
                print(f"Mengabaikan pasal di bagian preambule: {line}")
        except ValueError:
            print(f"Gagal memproses nomor pasal: {pasal_number_str}")
            if current_pasal:
                current_lines.append(line)
            continue

    # Deteksi footer, tapi lanjutkan proses jika belum ada pasal yang ditemukan
    if re.match(r"^(Disahkan|Ditetapkan)\s+di", line, re.IGNORECASE):
        print(f"Menemukan footer dokumen: '{line}'")
        if last_pasal_number > 0:
            # Tambahkan baris ini ke pasal sebelumnya sebagai konten terakhir
            if current_pasal and current_lines:
                current_lines.append(line)

            # Jika menemukan "Ditetapkan di Jakarta", langsung keluar dari loop
            if re.match(r"^Ditetapkan\s+di\s+Jakarta", line, re.IGNORECASE):
                print("Ditetapkan di Jakarta ditemukan. Proses chunking berhenti.")
                break

            # Jangan break langsung, teruskan ke baris berikutnya
            continue
        else:
            print(
                "Footer dokumen ditemukan sebelum menemukan pasal. Melanjutkan proses..."
            )

    # Tambahkan pendeteksian untuk pasal dengan format yang berbeda
    special_pasal_match = re.search(
        r"^(\s*)Pasal\s+(\d+[A-Z]?)(\s*)[,:]", line, re.IGNORECASE
    )
    if special_pasal_match and not pasal_match and not alt_pasal_match:
        pasal_number_str = special_pasal_match.group(2)
        try:
            pasal_number = int(re.sub(r"[A-Z]", "", pasal_number_str))
            if (
                is_valid_new_pasal(pasal_number, last_pasal_number)
                and pasal_number not in processed_pasal_numbers
            ):
                # Simpan pasal lama
                if current_pasal and current_lines and in_main_content:
                    content = "\n".join(current_lines).strip()
                    metadata = {
                        "jenis_peraturan": JENIS_PERATURAN,
                        "nomor_peraturan": NOMOR_PERATURAN,
                        "tahun_peraturan": TAHUN_PERATURAN,
                        "tipe_bagian": current_pasal,
                        "bagian_dari": "",  # Kosongkan karena tidak ada bab
                        "judul_peraturan": JUDUL_PERATURAN,
                        "status": "berlaku",
                    }
                    pasal_docs.append(Document(page_content=content, metadata=metadata))

                # Mulai pasal baru
                current_pasal = f"Pasal {pasal_number_str}"
                current_lines = [
                    line.replace(f"Pasal {pasal_number_str}:", "")
                    .replace(f"Pasal {pasal_number_str},", "")
                    .strip()
                ]
                last_pasal_number = pasal_number
                print(
                    f"Ditemukan format khusus dengan titik dua: {current_pasal} - {line}"
                )

                # Tambahkan ke pasal yang sudah diproses
                processed_pasal_numbers.add(pasal_number)

                continue
        except ValueError:
            print(f"Gagal memproses nomor pasal: {pasal_number_str}")
            if current_pasal:
                current_lines.append(line)

    # Tambahkan ke daftar pola format pasal alternatif
    if alt_pasal_match:
        for pattern in [
            # Format khusus lainnya bisa ditambahkan di sini
            r"^Pasa[l1]\s*ke\s*(\d+)[A-Z]?\s*$",
            r"^Pasa[l1]\s*(\d+)[A-Z]?\s*[:.]",
            r"^pasa[l1]\s*(\d+)[A-Z]?\s*$",
        ]:
            if re.match(pattern, line, re.IGNORECASE):
                match = re.match(pattern, line, re.IGNORECASE)
                pasal_number_str = match.group(1)
                print(
                    f"Ditemukan format khusus pasal: {line} -> Pasal {pasal_number_str}"
                )
                break

    # Tambahkan baris ke pasal saat ini jika kita sudah dalam konten utama
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

# Simpan pasal sebelumnya jika ada
if current_pasal and current_lines:
    content = "\n".join(current_lines).strip()

    # Cek apakah ada konten, jika kosong maka tidak perlu disimpan
    if content:
        # Jangan tambahkan konten pasal jika sudah ada metadata pasal yang sama
        if current_pasal in processed_pasal_numbers:
            print(f"Skip menyimpan {current_pasal} karena sudah ada")
        else:
            # Bersihkan konten dari bagian "Ditetapkan di Jakarta" dan setelahnya
            ditetapkan_match = re.search(
                r"Ditetapkan\s+di\s+Jakarta.*", content, re.DOTALL | re.IGNORECASE
            )
            if ditetapkan_match:
                content = content[: ditetapkan_match.start()].strip()
                print(
                    f"Bagian 'Ditetapkan di Jakarta' dan setelahnya tidak dimasukkan ke dalam CSV."
                )

            # Hanya simpan jika masih ada konten setelah pembersihan
            if content.strip():
                metadata = {
                    "jenis_peraturan": JENIS_PERATURAN,
                    "nomor_peraturan": NOMOR_PERATURAN,
                    "tahun_peraturan": TAHUN_PERATURAN,
                    "tipe_bagian": current_pasal,
                    "bagian_dari": "",  # Kosongkan karena tidak ada bab
                    "judul_peraturan": JUDUL_PERATURAN,
                    "status": "berlaku",
                }
                pasal_docs.append(Document(page_content=content, metadata=metadata))
                processed_pasal_numbers.add(pasal_number)
                print(f"Menyimpan konten {current_pasal}")

# Mulai pasal baru - hanya jika pasal belum diproses sebelumnya
if pasal_number not in processed_pasal_numbers:
    current_pasal = f"Pasal {pasal_number_str}"
    current_lines = [
        line.replace(f"Pasal {pasal_number_str}:", "")
        .replace(f"Pasal {pasal_number_str},", "")
        .strip()
    ]
    last_pasal_number = pasal_number
    processed_pasal_numbers.add(pasal_number)
    print(f"Mulai pasal baru: {current_pasal}")
else:
    print(f"Skip memulai pasal baru {pasal_number} karena sudah diproses")

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

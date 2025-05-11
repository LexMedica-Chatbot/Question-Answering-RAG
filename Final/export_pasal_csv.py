import os
import re
import csv
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.schema import Document
import json

"""EXPORT PASAL CSV
Script ini membaca semua PDF di folder "documents", memecah konten per Pasal,
menambah metadata yang relevan, lalu menulisnya ke file CSV. Gunakan untuk
verifikasi manual sebelum memasukkan data ke Supabase.
"""

# ---------------------------------------------------------------------------
# 1. Inisialisasi
# ---------------------------------------------------------------------------
load_dotenv()

# Sesuaikan jika folder PDF Anda berbeda
PDF_DIR = "documents"
OUTPUT_CSV = "pasal_output.csv"

# Placeholder, akan diisi otomatis setelah parsing header dokumen
JENIS_PERATURAN = "UNKNOWN"
NOMOR_PERATURAN = "-"
TAHUN_PERATURAN = "-"

# ---------------------------------------------------------------------------
# 2. Muat dokumen PDF
# ---------------------------------------------------------------------------
print(f"Memuat dokumen dari {PDF_DIR}")
loader = PyPDFDirectoryLoader(PDF_DIR)
pages = loader.load()
full_text = "\n".join([page.page_content for page in pages])
print(f"Dokumen berhasil dimuat: {len(pages)} halaman")

# ---------------------------------------------------------------------------
# 2.a. Ekstraksi Jenis, Nomor, Tahun Peraturan dari teks
# ---------------------------------------------------------------------------

# Jenis Peraturan (prioritaskan PP dahulu karena dokumen PP tetap memuat kata "Undang-Undang")
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
# 4. Parsing per Pasal
# ---------------------------------------------------------------------------
current_bab = ""
current_pasal = ""
current_lines: list[str] = []
current_bab_int = 0  # nilai integer bab terakhir yang valid
last_pasal_number = 0  # menyimpan nomor pasal terakhir untuk validasi

# Jika judul bab berada di baris berikutnya setelah "BAB X", gunakan flag
expect_bab_title = False

pasal_docs: list[Document] = []

# Helper mengonversi angka Romawi ke integer
ROMAN_MAP = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
    "XI": 11,
    "XII": 12,
    "XIII": 13,
    "XIV": 14,
    "XV": 15,
    "XVI": 16,
    "XVII": 17,
}


def roman_to_int(roman: str) -> int:
    return ROMAN_MAP.get(roman.upper(), 0)


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


for raw_line in full_text.splitlines():
    line = raw_line.strip()

    # Bersihkan teks dari kesalahan OCR
    line = clean_text(line)

    if not line or re.match(r"^\d+\s*$", line):
        continue

    # Jika kita sedang menunggu judul bab pada baris setelah 'BAB X'
    if expect_bab_title:
        # Jika baris ini bukan pasal atau header lainnya, anggap sebagai judul bab
        if not re.match(r"^Pasal\s+", line, re.IGNORECASE):
            title_line = line.title().strip()
            current_bab = current_bab + (f" - {title_line}" if title_line else "")
            expect_bab_title = False
            continue  # judul bab tidak dimasukkan ke konten pasal

    # Deteksi BAB (mungkin tanpa judul di baris yang sama)
    bab_match = re.match(
        r"^BAB\s+([IVXLCDM]+)(?:\s*[-\.]*\s*(.*))?", line, re.IGNORECASE
    )
    if bab_match:
        roman, title = bab_match.group(1), (bab_match.group(2) or "").strip()
        # Validasi urutan BAB
        roman_int = roman_to_int(roman)
        if current_bab_int == 0 and roman_int != 1:
            # Belum ada BAB I tapi sudah ketemu BAB lain, abaikan (kemungkinan header)
            continue
        if roman_int != 0 and roman_int < current_bab_int:
            # Kadang header menunjukkan BAB sebelumnya; abaikan regresi
            continue

        current_bab_int = roman_int

        title_formatted = title.title()
        current_bab = f"Bab {roman.strip()}"
        if title_formatted:
            current_bab += f" - {title_formatted}"
            expect_bab_title = False
        else:
            # judul ada di baris berikutnya
            expect_bab_title = True
        continue

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
    if pasal_match:
        # Validasi format: "Pasal X" harusnya berdiri sendiri di baris tersebut
        # Ini membantu menghindari referensi pasal dalam teks ayat

        pasal_number_str = pasal_match.group(1)
        try:
            pasal_number = int(re.sub(r"[A-Z]", "", pasal_number_str))

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
                        "bagian_dari": current_bab,
                        "judul_peraturan": JUDUL_PERATURAN,
                        "status": "berlaku",
                    }
                    pasal_docs.append(Document(page_content=content, metadata=metadata))

                # Mulai pasal baru
                current_pasal = f"Pasal {pasal_match.group(1)}"
                current_lines = []
                last_pasal_number = pasal_number
                print(f"Ditemukan {current_pasal}")

                continue
            else:
                print(f"Mengabaikan referensi pasal di baris: {line}")
        except ValueError:
            print(f"Gagal memproses nomor pasal: {pasal_match.group(1)}")
            # Tetap tambahkan ke baris konten
            if current_pasal:
                current_lines.append(line)
            continue

    # Lewati header/top-page atau kata kunci yang tidak penting
    if re.match(r"^(PRESIDEN|REPUBLIK INDONESIA)$", line, re.IGNORECASE):
        continue
    if re.match(r"^-\s*\d+\s*-?$", line):  # contoh: "- 3 -"
        continue
    if re.match(
        r"^(Menimbang|Mengingat|Memutuskan|Menetapkan)[:\s]", line, re.IGNORECASE
    ):
        continue

    if re.match(r"^(Disahkan|Ditetapkan)\s+di", line, re.IGNORECASE):
        print(f"Menemukan footer dokumen: '{line}'")
        # Selesai memproses dokumen, abaikan footer
        break

    if current_pasal:
        current_lines.append(line)

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
        "bagian_dari": current_bab,
        "judul_peraturan": JUDUL_PERATURAN,
        "status": "berlaku",
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

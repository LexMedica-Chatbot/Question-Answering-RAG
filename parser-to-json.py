import fitz  # PyMuPDF
import re
import json
from pathlib import Path

def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def final_clean_text(text):
    return re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()

def extract_header_info(raw_text):
    match = re.search(r"(?i)BAB\s+I", raw_text)
    header_text = raw_text[:match.start()] if match else raw_text
    lines = [line.strip() for line in header_text.splitlines() if line.strip()]

    title, nomor_tahun, tentang = "", "", ""

    for line in lines:
        if re.fullmatch(r"(?i)(UNDANG-UNDANG REPUBLIK INDONESIA|PERATURAN PEMERINTAH REPUBLIK INDONESIA|PERATURAN MENTERI.*)", line):
            title = line
            break

    for line in lines:
        m = re.search(r"(?i)^NOMOR\s+\d+[A-Z]*\s+TAHUN\s+\d{4}", line)
        if m:
            nomor_tahun = m.group(0)
            break

    for i, line in enumerate(lines):
        if re.fullmatch(r"(?i)TENTANG", line):
            if i + 1 < len(lines):
                tentang = lines[i + 1]
            break

    return {
        "judul": title,
        "nomor_tahun": nomor_tahun,
        "tentang": tentang
    }

def segment_by_chapter_and_pasals(text):
    chapters = re.split(r'(?m)(?=^BAB\s+[IVXLCDM]+\b)', text)
    chapters = [ch.strip() for ch in chapters if ch.strip()]
    chapter_segments = []

    for chapter in chapters:
        pasals = re.split(r'(?=(?i:Pasal\s+\d+[A-Za-z]*))', chapter)
        pasals = [p.strip() for p in pasals if p.strip()]
        chapter_header = re.split(r'(?i)\bPasal\b', pasals[0])[0].strip()
        pasal_list = pasals[1:] if len(pasals) > 1 else []

        chapter_segments.append({
            "chapter": final_clean_text(chapter_header),
            "pasals": [
                {
                    "pasal": re.search(r"(Pasal\s+\d+[A-Za-z]*)", p).group(0),
                    "isi": final_clean_text(p)
                }
                for p in pasal_list if re.search(r"(Pasal\s+\d+)", p)
            ]
        })

    return chapter_segments

def parse_pdf_to_json(pdf_path, output_folder):
    raw_text = extract_text_from_pdf(pdf_path)
    header_info = extract_header_info(raw_text)

    match = re.search(r"(?i)BAB\s+I\s*[\r\n]+\s*KETENTUAN\s+UMUM", raw_text)
    main_text = raw_text[match.start():] if match else raw_text
    clean_main = final_clean_text(main_text)

    segmented = segment_by_chapter_and_pasals(clean_main)

    # Ubah info header
    nomor_match = re.search(r"\d+[A-Z]*", header_info["nomor_tahun"])
    tahun_match = re.search(r"TAHUN\s+(\d{4})", header_info["nomor_tahun"])
    jenis_match = re.search(r"(UNDANG-UNDANG|PERATURAN PEMERINTAH|PERATURAN MENTERI.*)", header_info["judul"], re.I)

    output_json = {
        "jenis": jenis_match.group(1).title() if jenis_match else "Peraturan",
        "nomor": nomor_match.group(0) if nomor_match else "-",
        "tahun": int(tahun_match.group(1)) if tahun_match else 0,
        "judul": header_info["judul"],
        "tentang": header_info["tentang"],
        "pasals": []
    }

    for chapter in segmented:
        for pasal in chapter["pasals"]:
            output_json["pasals"].append({
                "chapter": chapter["chapter"],
                "pasal": pasal["pasal"],
                "isi": pasal["isi"]
            })

    # Simpan ke file
    nama_file = f"{output_json['jenis'].replace(' ', '_')}_{output_json['nomor']}_{output_json['tahun']}.json"
    save_path = Path(output_folder) / nama_file

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)

    print(f"✅ JSON saved: {save_path}")

parse_pdf_to_json("data/PP_Nomor_28_Tahun_2024.pdf", "data-json/")

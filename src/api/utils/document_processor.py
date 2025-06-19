"""
Document processing utilities untuk Multi-Step RAG system
"""

import re
import json
from typing import Dict, List, Any


def clean_control(text: str) -> str:
    """Bersihkan karakter kontrol dari teks."""
    return "".join(ch if ch >= " " else " " for ch in text)


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
        nomor_halaman = doc.metadata.get("nomor_halaman", "")  # Tambahkan nomor_halaman

        # Format dokumen dengan header yang lebih informatif
        if jenis_peraturan and nomor_peraturan and tahun_peraturan:
            doc_header = (
                f"{jenis_peraturan} No. {nomor_peraturan} Tahun {tahun_peraturan}"
            )
            if tipe_bagian:
                doc_header += f" {tipe_bagian}"
            doc_header += f" (Status: {status}"
            if nomor_halaman:
                doc_header += f", Hal: {nomor_halaman}"
            doc_header += ")"
        else:
            doc_header = f"Dokumen (Status: {status}"
            if nomor_halaman:
                doc_header += f", Hal: {nomor_halaman}"
            doc_header += ")"

        formatted_docs.append(f"{doc_header}:\n{doc.page_content}\n")

    return "\n\n".join(formatted_docs)


def extract_document_info(doc_content_str: str) -> Dict[str, str]:
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


def process_documents(formatted_docs_string: str) -> List[Dict[str, Any]]:
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


def format_reference(doc_info: dict) -> str:
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


def extract_legal_entities(docs) -> List[str]:
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


def find_document_links(doc_names, embedding_model="large") -> List[str]:
    """Find document links based on document names"""
    # Fungsi ini tetap sebagai placeholder
    # Tidak lagi menggunakan document_mapping hardcoded
    print(f"[DEBUG] Referensi dokumen: {doc_names}")

    # Return array kosong karena tidak ada mapping hardcoded
    return []

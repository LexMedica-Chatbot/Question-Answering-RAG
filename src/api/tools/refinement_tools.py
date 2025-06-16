"""
Query Refinement Tools untuk Multi-Step RAG system
"""

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from ..utils.config_manager import MODELS

# Variabel global untuk melacak jumlah penyempurnaan
refinement_count = 0

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
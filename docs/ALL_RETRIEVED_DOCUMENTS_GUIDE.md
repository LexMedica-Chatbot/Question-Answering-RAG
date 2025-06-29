# All Retrieved Documents Feature

## 📋 Overview

Field `all_retrieved_documents` telah ditambahkan ke response API untuk menyimpan **semua dokumen yang pernah diambil** dari vector store, termasuk dokumen yang statusnya "dicabut". Field ini berguna untuk **deteksi disharmony** oleh subsistem lain.

## 🎯 Purpose

-   **`referenced_documents`**: Dokumen aktif yang digunakan dalam jawaban
-   **`all_retrieved_documents`**: Semua dokumen (termasuk dicabut) untuk analisis disharmony

## 📊 Response Structure

```json
{
  "answer": "...",
  "referenced_documents": [
    // Hanya dokumen dengan status "berlaku"
  ],
  "all_retrieved_documents": [
    // SEMUA dokumen termasuk status "dicabut"
  ],
  "processing_steps": [...],
  "processing_time_ms": 1500,
  "model_info": {...}
}
```

## 🔍 Example Usage

### Input Query:

```
"Bagaimana ketentuan tenaga kesehatan menurut UU No. 36 Tahun 2009?"
```

### Response:

```json
{
    "referenced_documents": [
        // 4 dokumen aktif (PP 28/2024, PP 32/1996, UU 17/2023)
    ],
    "all_retrieved_documents": [
        // 7 dokumen total:
        // - 3 dokumen UU 36/2009 (dicabut) ⚠️
        // - 4 dokumen aktif
    ]
}
```

## ⚠️ Disharmony Detection

Dari contoh di atas, subsistem lain dapat mendeteksi:

-   **3 dokumen UU 36/2009 dicabut** tapi masih relevan dengan query
-   **Gap peraturan**: UU lama dicabut, perlu cek apakah diganti dengan peraturan baru
-   **Analisis ketidakselarasan**: Identifikasi inkonsistensi hukum

## 🛠️ Implementation Details

1. **Search Level**: Semua dokumen disimpan sebelum hard filtering
2. **API Level**: Field ditambahkan ke response model
3. **Cache Level**: Field di-cache untuk performance
4. **Deduplication**: Dokumen duplikat dihapus berdasarkan content hash

## 📈 Benefits for Subsystems

1. **Legal Compliance**: Identifikasi peraturan yang sudah tidak berlaku
2. **Gap Analysis**: Temukan area hukum yang perlu update
3. **Harmonization**: Deteksi inkonsistensi antar peraturan
4. **Research**: Data lengkap untuk analisis komprehensif

## 🔧 Usage Notes

-   Field selalu ada di response (bisa empty array)
-   Dokumen di-sort berdasarkan relevance score
-   Status dokumen tersedia di `metadata.status`
-   Deduplication berdasarkan content hash

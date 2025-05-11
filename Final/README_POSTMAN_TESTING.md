# Panduan Pengujian API RAG Dokumen Hukum dengan Postman

## Persiapan

### 1. Menjalankan Server API

Jalankan server API dengan perintah:

```bash
cd Final
python simple_api.py
```

Server akan berjalan di `http://localhost:8000`. Perhatikan output di terminal untuk mendapatkan API Key yang akan digunakan.

### 2. Menyiapkan Postman

1. Buka Postman
2. Buat koleksi baru (New Collection) dengan nama "RAG Dokumen Hukum"
3. Buat environment baru dengan variabel:
    - `base_url`: `http://localhost:8000`
    - `api_key`: [API Key dari output terminal]

## Endpoint yang Tersedia

### 1. Check Health - Memeriksa Kondisi Server

-   **Method**: GET
-   **URL**: `{{base_url}}/health`
-   **Headers**: Tidak perlu API Key

### 2. Get Available Models - Mendapatkan Informasi Model

-   **Method**: GET
-   **URL**: `{{base_url}}/api/models`
-   **Headers**:
    -   X-API-Key: `{{api_key}}`

### 3. Chat - Mengajukan Pertanyaan

-   **Method**: POST
-   **URL**: `{{base_url}}/api/chat`
-   **Headers**:
    -   X-API-Key: `{{api_key}}`
    -   Content-Type: `application/json`
-   **Body** (raw, JSON):

```json
{
    "query": "Jelaskan apa yang dimaksud dengan kesehatan reproduksi?",
    "embedding_model": "small",
    "previous_responses": []
}
```

## Contoh Pengujian untuk Status "Berlaku" dan "Dicabut"

### Pengujian 1: Informasi dari PP 61/2014 (Dicabut)

-   **Query**: `Bagaimana pengaturan tentang kesehatan reproduksi di Indonesia?`
-   **Ekspektasi**: Jawaban harus menyebutkan bahwa PP 61/2014 sudah dicabut dan memberikan informasi dari peraturan yang berlaku.

### Pengujian 2: Perbandingan Peraturan Lama dan Baru

-   **Query**: `Bandingkan pengaturan kesehatan reproduksi di PP 61/2014 dengan PP 28/2024`
-   **Ekspektasi**: Jawaban harus membandingkan kedua peraturan dengan jelas menyebutkan status masing-masing peraturan (yang satu dicabut, yang satu berlaku).

### Pengujian 3: Prioritas Peraturan Berlaku

-   **Query**: `Apa saja hak-hak kesehatan reproduksi?`
-   **Ekspektasi**: Jawaban harus memprioritaskan informasi dari peraturan yang masih berlaku, tetapi bisa mereferensikan peraturan yang dicabut untuk konteks historis.

## Tips Pengujian

1. **Gunakan model yang berbeda**: Ubah parameter `embedding_model` antara "small" dan "large" untuk melihat perbedaan hasil.

2. **Gunakan riwayat chat**: Untuk menguji konteks percakapan, tambahkan jawaban sebelumnya di array `previous_responses`.

3. **Periksa metadata dokumen**: Perhatikan detail dokumen yang direferensikan (termasuk status) di respons API. Format respons:

```json
{
  "answer": "Jawaban dari model",
  "document_links": [...],
  "model_info": {...},
  "referenced_documents": [
    {
      "name": "Dokumen #1",
      "description": "PP No. 28 Tahun 2024 Pasal 2 (Status: berlaku)",
      "source": "PP No. 28/2024 (Status: berlaku)",
      "content": "...",
      "metadata": {...}
    }
  ]
}
```

4. **Pertanyaan lanjutan**: Gunakan pertanyaan lanjutan untuk mengeksplorasi perbedaan antara peraturan yang dicabut dan yang berlaku.

## Troubleshooting

1. **Error 403 Forbidden**: Pastikan API Key yang digunakan sudah benar. Cek kembali terminal tempat menjalankan server.

2. **Tidak ada hasil yang relevan**: Pastikan query berkaitan dengan dokumen kesehatan yang ada di database. Lebih spesifik lebih baik.

3. **Jawaban terlalu umum**: Coba gunakan model "large" untuk hasil yang lebih akurat, atau tambahkan kata kunci spesifik seperti nomor peraturan.

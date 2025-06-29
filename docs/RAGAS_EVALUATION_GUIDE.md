# 📊 Panduan Evaluasi RAG dengan Ragas

Dokumentasi lengkap untuk mengevaluasi sistem Single RAG dan Multi-Agent RAG menggunakan framework Ragas.

## 🎯 Overview

Evaluasi ini akan membandingkan performa kedua sistem RAG Anda:

-   **Single RAG**: `https://lexmedica-chatbot-176465812210.asia-southeast2.run.app`
-   **Multi-Agent RAG**: `https://lexmedica-chatbot-multiagent-176465812210.asia-southeast2.run.app`

Menggunakan metrik evaluasi Ragas:

-   **Context Recall**: Seberapa baik sistem mengambil konteks yang relevan
-   **Faithfulness**: Seberapa akurat jawaban berdasarkan konteks yang diberikan
-   **Factual Correctness**: Kebenaran faktual dari jawaban
-   **Answer Relevancy**: Relevansi jawaban terhadap pertanyaan

## 🔧 Prerequisites

### 1. Environment Setup

```bash
# Aktivasi virtual environment
qa-system\Scripts\activate

# Verifikasi environment aktif
# Prompt harus menampilkan: (qa-system)
```

### 2. API Keys yang Diperlukan

-   **RAG API Key**: Untuk mengakses kedua endpoint RAG Anda
-   **OpenAI API Key**: Untuk evaluasi Ragas (menggunakan GPT-4o-mini)

### 3. Data Evaluasi

-   File `validasi_ta.csv` dengan format:
    ```csv
    question,answer,context,ground_truth
    "Pertanyaan 1",,,"Jawaban referensi 1"
    "Pertanyaan 2",,,"Jawaban referensi 2"
    ```

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
python install_evaluation_deps.py
```

### Step 2: Test Endpoints

```bash
python benchmarks/test_endpoints_updated.py
```

-   Masukkan API Key saat diminta
-   Verifikasi kedua endpoint bekerja dengan baik

### Step 3: Run Full Evaluation

```bash
python benchmarks/ragas_evaluation.py
```

-   Masukkan RAG API Key
-   Masukkan OpenAI API Key (jika belum di environment)
-   Tunggu proses evaluasi selesai

## 📁 File Structure

```
benchmarks/
├── ragas_evaluation.py          # Script utama evaluasi
├── test_endpoints_updated.py    # Test endpoint dengan format benar
├── test_endpoints.py           # Test endpoint basic (deprecated)
└── discover_endpoints.py       # Discovery endpoint (utility)

validasi_ta.csv                 # Data evaluasi
install_evaluation_deps.py      # Script install dependencies
```

## 🔧 Detail Teknis

### Format Request

#### Single RAG

```json
POST /api/chat
Headers: X-API-Key: <your_api_key>
{
  "query": "Pertanyaan Anda",
  "embedding_model": "large"
}
```

#### Multi-Agent RAG

```json
POST /api/chat
Headers: X-API-Key: <your_api_key>
{
  "query": "Pertanyaan Anda",
  "embedding_model": "large",
  "previous_responses": [],
  "use_parallel_execution": true
}
```

### Format Response

#### Single RAG Response

```json
{
  "answer": "string",
  "model_info": {},
  "referenced_documents": [],
  "processing_time_ms": int
}
```

#### Multi-Agent RAG Response

```json
{
  "answer": "string",
  "processing_steps": [
    {
      "tool": "string",
      "tool_input": {},
      "tool_output": "string"
    }
  ],
  "processing_time_ms": int,
  "model_info": {}
}
```

### Context Extraction Logic

**Single RAG**: Mengambil konteks dari field `referenced_documents`

**Multi-Agent RAG**: Mengambil konteks dari `processing_steps`, mencari tools yang berkaitan dengan search/retrieval

## 📊 Output Evaluasi

### 1. Console Output

```
RAG EVALUATION SUMMARY
================================================================================

SINGLE RAG:
--------------------------------------------------
Context Recall: 0.8500
Faithfulness: 0.9200
Factual Correctness: 0.7800
Answer Relevancy: 0.8900
Number of questions evaluated: 5

MULTI AGENT RAG:
--------------------------------------------------
Context Recall: 0.9100
Faithfulness: 0.9500
Factual Correctness: 0.8200
Answer Relevancy: 0.9300
Number of questions evaluated: 5
```

### 2. File Output

```
benchmark_results/
└── ragas_evaluation_20250116_143052_complete.json
```

Berisi:

-   Metrik evaluasi untuk kedua sistem
-   Raw data responses dari setiap pertanyaan
-   Konteks yang diambil
-   Ground truth comparisons

## 🎯 Metrik Evaluasi

### Context Recall (0.0 - 1.0)

-   **Tinggi (>0.8)**: Sistem berhasil mengambil sebagian besar konteks relevan
-   **Sedang (0.5-0.8)**: Sistem mengambil beberapa konteks relevan
-   **Rendah (<0.5)**: Sistem kesulitan mengambil konteks yang relevan

### Faithfulness (0.0 - 1.0)

-   **Tinggi (>0.8)**: Jawaban sangat konsisten dengan konteks
-   **Sedang (0.5-0.8)**: Jawaban cukup konsisten dengan konteks
-   **Rendah (<0.5)**: Jawaban tidak konsisten dengan konteks

### Factual Correctness (0.0 - 1.0)

-   **Tinggi (>0.8)**: Jawaban faktual sangat akurat
-   **Sedang (0.5-0.8)**: Jawaban faktual cukup akurat
-   **Rendah (<0.5)**: Jawaban faktual kurang akurat

### Answer Relevancy (0.0 - 1.0)

-   **Tinggi (>0.8)**: Jawaban sangat relevan dengan pertanyaan
-   **Sedang (0.5-0.8)**: Jawaban cukup relevan dengan pertanyaan
-   **Rendah (<0.5)**: Jawaban kurang relevan dengan pertanyaan

## 🐛 Troubleshooting

### Error 403 Forbidden

**Penyebab**: API Key tidak valid atau tidak disertakan
**Solusi**:

-   Verifikasi API Key benar
-   Pastikan header `X-API-Key` ter-set dengan benar

### Error 404 Not Found

**Penyebab**: Endpoint URL salah
**Solusi**:

-   Pastikan menggunakan `/api/chat` bukan `/query`
-   Verifikasi URL base correct

### Timeout Error

**Penyebab**: Request membutuhkan waktu terlalu lama
**Solusi**:

-   Periksa koneksi internet
-   Coba pertanyaan yang lebih sederhana
-   Tingkatkan timeout di script

### OpenAI API Error

**Penyebab**: API Key OpenAI tidak valid atau quota habis
**Solusi**:

-   Verifikasi OpenAI API Key
-   Periksa quota dan billing
-   Ganti ke model yang lebih murah (gpt-3.5-turbo)

### Import Error untuk Ragas

**Penyebab**: Dependencies tidak terinstall
**Solusi**:

```bash
pip install ragas==0.3.0 datasets langchain-openai
```

## 📈 Analisis Hasil

### Interpretasi Perbandingan

1. **Context Recall lebih tinggi** → Sistem lebih baik dalam retrieval
2. **Faithfulness lebih tinggi** → Sistem lebih konsisten dengan sumber
3. **Factual Correctness lebih tinggi** → Sistem lebih akurat secara faktual
4. **Answer Relevancy lebih tinggi** → Sistem lebih memahami intent pertanyaan

### Rekomendasi Berdasarkan Hasil

-   **Jika Single RAG lebih baik**: Arsitektur sederhana sudah optimal
-   **Jika Multi-Agent RAG lebih baik**: Multi-step processing memberikan value
-   **Jika hasil mixed**: Pertimbangkan hybrid approach atau fine-tuning

## 🔄 Workflow Lengkap

```bash
# 1. Setup environment
qa-system\Scripts\activate

# 2. Install dependencies
python install_evaluation_deps.py

# 3. Test endpoints
python benchmarks/test_endpoints_updated.py

# 4. Run evaluation (ini yang utama)
python benchmarks/ragas_evaluation.py

# 5. Analyze results
# Check console output dan file di benchmark_results/
```

## 📝 Custom Evaluation

Untuk menggunakan dataset evaluasi sendiri:

1. **Format CSV**:

    ```csv
    question,answer,context,ground_truth
    "Pertanyaan custom",,,"Ground truth custom"
    ```

2. **Run dengan file custom**:
    ```python
    evaluator = RAGEvaluator(api_key="your_key")
    results = evaluator.run_complete_evaluation("path/to/your/file.csv")
    ```

## 📞 Support

Jika mengalami masalah:

1. Periksa log error di console
2. Verifikasi API Keys
3. Test endpoint satu per satu
4. Periksa format data evaluasi

---

**Happy Evaluating! 🎉**

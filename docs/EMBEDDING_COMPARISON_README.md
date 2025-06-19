# 🔬 Embedding Comparison Benchmark System

Sistem benchmark untuk membandingkan performa Text Embedding Small vs Large pada Simple API dalam konteks sistem RAG hukum kesehatan Indonesia.

## 📋 Overview

Script ini membandingkan dua model embedding OpenAI:

-   **text-embedding-3-small**: Model embedding yang lebih cepat dan ekonomis
-   **text-embedding-3-large**: Model embedding yang lebih akurat dengan dimensi lebih tinggi

### Metrik yang Diukur

1. **Performance Metrics**

    - Response time (ms)
    - Error rate
    - Answer length & word count

2. **Quality Metrics**

    - Keyword matching score
    - RAGAs metrics (jika tersedia):
        - Faithfulness
        - Answer relevancy
        - Context precision/recall
        - Context relevancy

3. **Content Analysis**
    - Jumlah dokumen yang diambil (contexts)
    - Panjang total konteks

## 🚀 Quick Start

### 1. Menjalankan Benchmark dengan Mock Data

Untuk testing cepat tanpa perlu Simple API aktif:

```bash
python benchmarks/simple_embedding_comparison.py --dry-run
```

### 2. Menjalankan Benchmark dengan Real API

Pastikan Simple API berjalan di `http://localhost:8081`:

```bash
python benchmarks/simple_embedding_comparison.py
```

### 3. Melihat Hasil

```bash
# Melihat summary report
python view_embedding_results.py

# Melihat summary + chart
python view_embedding_results.py --chart

# Export ke markdown
python view_embedding_results.py --export

# Kombinasi
python view_embedding_results.py --chart --export
```

## 📊 Test Cases

Benchmark menggunakan 9 test cases yang mencakup berbagai kategori:

### Easy Level (3 cases)

-   Definisi tenaga kesehatan
-   Perizinan praktik dokter
-   STR (Surat Tanda Registrasi)

### Medium Level (4 cases)

-   Sanksi hukum praktik tanpa izin
-   Prosedur izin dokter spesialis
-   Informed consent
-   Ketentuan rekam medis

### Hard Level (2 cases)

-   Hubungan UU Praktik Kedokteran dengan UU Rumah Sakit
-   Perbedaan sanksi pidana vs perdata

## 📁 Output Structure

```
embedding_comparison_results/
├── charts/
│   └── embedding_comparison_YYYYMMDD_HHMMSS.png
├── data/
│   ├── embedding_comparison_YYYYMMDD_HHMMSS.csv
│   └── embedding_summary_YYYYMMDD_HHMMSS.json
└── embedding_report_YYYYMMDD_HHMMSS.md
```

## 🔧 Konfigurasi

Edit file `benchmarks/simple_embedding_comparison.py` bagian `BenchmarkConfig`:

```python
@dataclass
class BenchmarkConfig:
    api_base_url: str = "http://localhost:8081"  # URL Simple API
    api_key: str = "your_secure_api_key_here"    # API Key
    output_dir: str = "embedding_comparison_results"  # Output directory
    timeout: int = 60                            # Timeout per request
    dry_run: bool = False                        # Mock data mode
```

## 📈 Interpretasi Hasil

### Response Time

-   **Perbedaan <10%**: Pilih berdasarkan faktor lain (cost, quality)
-   **Small lebih cepat**: Cocok untuk high-volume applications
-   **Large lebih cepat**: Unusual, tapi bisa terjadi tergantung load

### Quality Metrics

-   **Keyword Score**: Seberapa banyak kata kunci yang terdapat dalam jawaban
-   **Faithfulness**: Seberapa akurat jawaban terhadap konteks (RAGAs)
-   **Answer Relevancy**: Seberapa relevan jawaban terhadap pertanyaan (RAGAs)

### Error Analysis

-   **0% error**: API berjalan normal
-   **>0% error**: Periksa koneksi API atau konfigurasi

## 🎯 Use Cases

### Kapan Menggunakan Small Embedding

-   ✅ Real-time applications
-   ✅ High-volume queries
-   ✅ Budget constraints
-   ✅ Response time critical

### Kapan Menggunakan Large Embedding

-   ✅ Accuracy critical applications
-   ✅ Complex legal analysis
-   ✅ Quality over speed
-   ✅ Detailed document retrieval

## 🛠️ Troubleshooting

### Error: "Connection refused"

```bash
# Pastikan Simple API berjalan
curl http://localhost:8081/health

# Atau gunakan dry-run mode
python benchmarks/simple_embedding_comparison.py --dry-run
```

### Error: "No module named 'ragas'"

```bash
# Install RAGAs (optional)
pip install ragas

# Atau lanjutkan tanpa RAGAs (metrics akan 0)
```

### Error: "Charts not displaying"

```bash
# Install PIL untuk display charts
pip install Pillow

# Atau buka file PNG secara manual
```

## 📊 Sample Output

```
📊 EMBEDDING COMPARISON SUMMARY REPORT
============================================================
📅 Generated: 20250617_155607
🧪 Total Tests: 18
🔍 Models Compared: Small vs Large Embedding

📈 PERFORMANCE METRICS
----------------------------------------
⚡ Response Time:
   Small Embedding: 1928 ms
   Large Embedding: 1822 ms
   Difference: 107 ms (5.9%)
   Winner: Large ⭐

🎯 QUALITY METRICS
----------------------------------------
Keyword Matching:
   Small: 0.389
   Large: 0.389
   Diff: +0.000

💡 RECOMMENDATIONS
----------------------------------------
🟡 Performance difference is minimal (<10%)
   → Choose based on quality metrics or cost considerations
```

## 🔄 Extending the Benchmark

### Menambah Test Cases

Edit fungsi `create_test_cases()` di `benchmarks/simple_embedding_comparison.py`:

```python
TestCase(
    question="Pertanyaan baru Anda",
    category="kategori_baru",
    difficulty="easy|medium|hard",
    expected_keywords=["kata", "kunci", "yang", "diharapkan"]
)
```

### Menambah Metrik Custom

Edit fungsi `calculate_keyword_score()` atau tambahkan fungsi evaluasi baru.

### Mengubah Visualisasi

Edit fungsi `create_visualizations()` untuk menyesuaikan grafik.

## 📝 Dependencies

```
pandas
matplotlib
seaborn
requests
pathlib
dataclasses
numpy
```

Optional:

```
ragas  # Untuk advanced quality metrics
PIL    # Untuk display charts
```

## 🤝 Contributing

1. Fork repository
2. Buat feature branch
3. Commit changes
4. Push ke branch
5. Create Pull Request

## 📄 License

MIT License - see LICENSE file for details.

---

**Generated by:** Embedding Comparison Benchmark System  
**Last Updated:** 2025-06-17

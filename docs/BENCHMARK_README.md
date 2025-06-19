# 🔬 RAG Benchmark System untuk Laporan Skripsi

Sistem benchmark komprehensif untuk membandingkan **Simple API** vs **Multi-Step API** dengan **Small Embedding** vs **Large Embedding** untuk penelitian skripsi.

## 🎯 Tujuan Benchmark

Menghasilkan data empiris untuk laporan skripsi yang membandingkan **6 kombinasi**:

1. **Simple API + Small Embedding** - Basic RAG dengan embedding cepat
2. **Simple API + Large Embedding** - Basic RAG dengan embedding komprehensif
3. **Multi API (Parallel) + Small Embedding** - Multi-step RAG paralel dengan embedding cepat
4. **Multi API (Parallel) + Large Embedding** - Multi-step RAG paralel dengan embedding komprehensif
5. **Multi API (Sequential) + Small Embedding** - Multi-step RAG berurutan dengan embedding cepat
6. **Multi API (Sequential) + Large Embedding** - Multi-step RAG berurutan dengan embedding komprehensif

**Perbedaan Execution Mode**:

-   **Parallel**: Multiple processing steps dijalankan bersamaan (faster, less thorough)
-   **Sequential**: Processing steps dijalankan berurutan (slower, more thorough)

## 📊 Metrics yang Diukur

### Performance Metrics

-   **Response Time** (ms) - Waktu respons API
-   **Success Rate** (%) - Tingkat keberhasilan request
-   **Throughput** - Jumlah request per detik

### Quality Metrics

-   **Answer Length** - Panjang jawaban (characters/words)
-   **Document Retrieval** - Jumlah dokumen yang diambil
-   **Keyword Matching** - Kesesuaian dengan kata kunci yang diharapkan
-   **Structure Score** - Kualitas struktur jawaban
-   **Overall Quality** - Skor kualitas keseluruhan

### Technical Metrics

-   **Processing Steps** - Jumlah langkah pemrosesan (Multi API)
-   **Memory Usage** - Penggunaan memori
-   **Error Analysis** - Analisis error yang terjadi

## 🛠️ Setup dan Instalasi

### Prerequisites

```bash
# Install Docker dan Docker Compose
# Install Python 3.8+
# Install dependencies
pip install requests pandas pathlib logging
```

### Environment Variables

Pastikan file `.env` sudah dikonfigurasi dengan:

```env
API_KEY=your_secure_api_key_here
OPENAI_API_KEY=your_openai_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_supabase_key
REDIS_URL=your_redis_url
```

## 🚀 Cara Menjalankan Benchmark

### Opsi 1: Automatic Script (Recommended)

```bash
# Linux/Mac
./run_benchmark.sh

# Windows (PowerShell)
bash run_benchmark.sh
```

### Opsi 2: Manual Step-by-Step

#### Step 1: Start Multi API

```bash
docker-compose up -d
# Wait for health check: http://localhost:8080/health
```

#### Step 2: Start Simple API

```bash
docker-compose -f docker-compose-simple.yml up -d
```

#### Step 3: Run Benchmark

```bash
python benchmarks/complete_benchmark.py
```

### Opsi 3: RAGAs Academic Evaluation (6 Combinations)

```bash
# Install RAGAs framework
pip install ragas

# Run comprehensive RAGAs benchmark
python run_ragas_benchmark.py
```

### Opsi 4: Quick Test (Limited Questions)

```bash
python benchmarks/simple_benchmark.py
```

## 📝 Test Cases

Benchmark menggunakan 12 pertanyaan hukum kesehatan dengan tingkat kesulitan:

### Easy (3 questions)

-   Definisi tenaga kesehatan
-   Kewenangan penerbitan STR
-   Kepanjangan STR dan SIP

### Medium (4 questions)

-   Sanksi praktik tanpa izin
-   Prosedur izin praktik spesialis
-   Kewajiban informed consent
-   Ketentuan rekam medis

### Hard (3 questions)

-   Hubungan UU Praktik Kedokteran dengan UU Rumah Sakit
-   Perbedaan sanksi pidana dan perdata
-   Implementasi telemedicine

### Very Hard (2 questions)

-   Analisis komprehensif perlindungan hukum
-   Koordinasi pengawasan di era otonomi daerah

## 📊 Output yang Dihasilkan

### File untuk Laporan Skripsi

1. **comprehensive_report_TIMESTAMP.md**

    - Laporan lengkap dalam format Markdown
    - Executive summary
    - Detailed performance analysis
    - Recommendations

2. **latex_table_TIMESTAMP.tex**

    - Tabel LaTeX siap pakai untuk thesis
    - Format sesuai standar akademik

3. **benchmark_data_TIMESTAMP.csv**

    - Raw data untuk analisis statistik
    - Compatible dengan SPSS, R, Python

4. **benchmark_summary_TIMESTAMP.json**
    - Summary statistics
    - Key performance indicators

### Directory Structure

```
benchmark_results/
├── comprehensive_report_20241206_143022.md
├── latex_table_20241206_143022.tex
├── benchmark_data_20241206_143022.csv
├── benchmark_summary_20241206_143022.json
└── detailed/
    └── raw_responses_20241206_143022.json
```

## 📈 Analisis yang Disediakan

### Comparative Analysis

-   **Best Overall Performance**: Kombinasi terbaik secara keseluruhan
-   **Fastest Response**: Kombinasi tercepat
-   **Highest Quality**: Kualitas jawaban terbaik
-   **Most Comprehensive**: Paling komprehensif dalam retrieval

### Performance by Difficulty

-   Success rate per tingkat kesulitan
-   Response time breakdown
-   Quality score analysis

### Statistical Analysis

-   Mean, median, min, max untuk semua metrics
-   Standard deviation
-   Confidence intervals

## 🎓 Untuk Laporan Skripsi

### Methodology Section

```latex
\subsection{Evaluasi Sistem}

Evaluasi dilakukan menggunakan benchmark yang membandingkan dua pendekatan:
\begin{enumerate}
    \item Simple API - Pipeline RAG dasar
    \item Multi-Step API - Enhanced Multi-Step RAG dengan pendekatan agent-based
\end{enumerate}

Setiap pendekatan diuji dengan dua model embedding:
\begin{itemize}
    \item Small Embedding (text-embedding-3-small)
    \item Large Embedding (text-embedding-3-large)
\end{itemize}

Dataset evaluasi terdiri dari 12 pertanyaan hukum kesehatan dengan 4 tingkat kesulitan.
```

### Results Section

Gunakan tabel LaTeX yang dihasilkan dan data dari comprehensive report.

### Discussion Section

Analisis trade-offs antara:

-   Speed vs Quality
-   Simple vs Complex approach
-   Small vs Large embeddings

## 🐛 Troubleshooting

### Port Already in Use

```bash
# Stop existing containers
docker-compose down
docker-compose -f docker-compose-simple.yml down

# Check ports
netstat -an | grep :8080
netstat -an | grep :8081
```

### API Not Responding

```bash
# Check container logs
docker-compose logs
docker-compose -f docker-compose-simple.yml logs

# Check health endpoints
curl http://localhost:8080/health
curl http://localhost:8081/health
```

### Benchmark Failures

```bash
# Check Python dependencies
pip install -r requirements.txt

# Run with verbose logging
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
python benchmarks/complete_benchmark.py
```

## ⚡ Quick Start Commands

```bash
# Complete benchmark run
docker-compose up -d && docker-compose -f docker-compose-simple.yml up -d
sleep 30
python benchmarks/complete_benchmark.py

# Stop all services
docker-compose down && docker-compose -f docker-compose-simple.yml down
```

## 📋 Checklist untuk Laporan Skripsi

-   [ ] ✅ Redis cache berfungsi (Multi API)
-   [ ] ✅ Kedua API responding correctly
-   [ ] ✅ Benchmark completed successfully
-   [ ] ✅ All 6 combinations tested (Simple×2 + Multi-Parallel×2 + Multi-Sequential×2)
-   [ ] ✅ Results saved with timestamp
-   [ ] ✅ LaTeX table generated
-   [ ] ✅ Comprehensive report available
-   [ ] ✅ CSV data ready for statistical analysis

## 🎯 Expected Results for Thesis

Berdasarkan implementasi, ekspektasi hasil:

-   **Multi API** akan menunjukkan kualitas jawaban lebih tinggi
-   **Large Embedding** akan memberikan akurasi lebih baik
-   **Simple API** akan lebih cepat dalam response time
-   **Small Embedding** cocok untuk aplikasi real-time

Data ini akan mendukung kesimpulan research question dalam skripsi.

---

**Timestamp Generation**: Semua file hasil akan memiliki timestamp yang sama untuk konsistensi referensi dalam laporan skripsi.

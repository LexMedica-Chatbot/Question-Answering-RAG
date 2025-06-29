# 📊 Datasets

Folder ini berisi file dataset dalam format CSV yang digunakan untuk evaluasi dan validasi sistem RAG.

## 📁 File yang Tersedia

### Evaluation Datasets

-   **`evaluation_dataset_health_law.csv`** - Dataset utama untuk evaluasi sistem RAG pada domain hukum kesehatan
-   **`validasi_ta.csv`** - Dataset validasi untuk tugas akhir/thesis

## 📋 Struktur Dataset

### evaluation_dataset_health_law.csv

Dataset ini berisi:

-   **Query/Pertanyaan** - Pertanyaan dalam bahasa Indonesia terkait hukum kesehatan
-   **Expected Answer** - Jawaban yang diharapkan berdasarkan peraturan
-   **Ground Truth** - Referensi dokumen yang seharusnya digunakan
-   **Metadata** - Informasi tambahan untuk evaluasi

### validasi_ta.csv

Dataset validasi untuk:

-   Verifikasi akurasi sistem
-   Testing edge cases
-   Validasi hasil penelitian

## 🎯 Penggunaan

Dataset ini digunakan oleh:

-   `analysis/run_fixed_evaluation.py` - Evaluasi lengkap sistem
-   `analysis/debug_evaluation.py` - Debug dan troubleshooting
-   Script evaluasi lainnya di folder `benchmarks/`

## 📝 Format

```csv
query,expected_answer,ground_truth_docs,category
"Apa itu fasyankes?","Fasilitas pelayanan kesehatan...","UU 17/2023","definisi"
```

## 🚀 Cara Menjalankan Evaluasi

```bash
# Menggunakan dataset untuk evaluasi
python analysis/run_fixed_evaluation.py

# Evaluasi dengan dataset khusus
python benchmarks/ragas_benchmark.py
```

## 📝 Catatan

-   File CSV tidak dimasukkan ke dalam Docker container
-   Dataset terus diperbarui untuk meningkatkan akurasi evaluasi
-   Digunakan untuk development dan testing, bukan production

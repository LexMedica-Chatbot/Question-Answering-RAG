# 📊 Analysis Scripts

Folder ini berisi script-script Python untuk analisis dan evaluasi sistem RAG.

## 📁 File yang Tersedia

### Evaluation Scripts

-   **`run_fixed_evaluation.py`** - Script utama untuk menjalankan evaluasi lengkap sistem
-   **`debug_evaluation.py`** - Script debugging untuk evaluasi dengan logging detail

### Visualization Scripts

-   **`create_specific_charts.py`** - Membuat chart khusus untuk analisis performa
-   **`quick_visualization.py`** - Visualisasi cepat untuk hasil evaluasi
-   **`evaluate_and_visualize.py`** - Kombinasi evaluasi dan visualisasi dalam satu script

## 🚀 Penggunaan

```bash
# Menjalankan evaluasi lengkap
python analysis/run_fixed_evaluation.py

# Debug evaluasi dengan logging detail
python analysis/debug_evaluation.py

# Membuat visualisasi cepat
python analysis/quick_visualization.py
```

## 📝 Catatan

-   Script ini tidak dimasukkan ke dalam Docker container
-   Digunakan untuk development dan analisis performa
-   Output biasanya disimpan di folder `benchmark_results/` atau `visualizations/`

#!/usr/bin/env python3
"""
SCRIPT DEBUG EVALUASI RAG
Untuk memahami bagaimana angka-angka evaluasi dihasilkan
"""

import json
import pandas as pd
from datetime import datetime


def load_latest_evaluation_results():
    """Load hasil evaluasi terbaru"""
    print("=" * 60)
    print("🔍 DEBUGGING EVALUATION RESULTS")
    print("=" * 60)

    # Load hasil evaluasi terbaru
    import glob
    import os

    results_files = glob.glob("benchmark_results/ragas_evaluation_*_complete.json")
    if not results_files:
        print("❌ Tidak ada file hasil evaluasi. Jalankan evaluasi dulu!")
        return None

    latest_file = max(results_files, key=os.path.getctime)
    print(f"📁 Loading file: {latest_file}")

    with open(latest_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    return results, latest_file


def analyze_evaluation_process(results):
    """Analisis proses evaluasi step by step"""

    print("\n" + "=" * 60)
    print("📊 ANALISIS PROSES EVALUASI")
    print("=" * 60)

    for system_name, system_data in results.items():
        print(f"\n🏷️ SISTEM: {system_name.upper()}")
        print("-" * 40)

        # Raw data (input untuk evaluasi)
        raw_data = system_data.get("raw_data", [])
        print(f"📝 Questions evaluated: {len(raw_data)}")

        # Evaluation metrics (output dari Ragas)
        metrics = system_data.get("evaluation_metrics", {})

        print("\n📈 EVALUATION METRICS:")
        for metric_name, score in metrics.items():
            if isinstance(score, (int, float)):
                percentage = score * 100
                print(f"   {metric_name}: {score:.4f} ({percentage:.1f}%)")

        print("\n🔍 SAMPLE DATA EVALUATION:")
        if raw_data:
            sample = raw_data[0]  # Ambil sample pertama
            print(f"   Question: {sample.get('user_input', '')[:100]}...")
            print(f"   Answer length: {len(sample.get('response', ''))} chars")
            print(f"   Context count: {len(sample.get('retrieved_contexts', []))}")
            print(f"   Ground truth length: {len(sample.get('reference', ''))} chars")


def show_detailed_calculation():
    """Tunjukkan detail perhitungan untuk 1 sample"""

    print("\n" + "=" * 60)
    print("🧮 DETAIL PERHITUNGAN EVALUASI")
    print("=" * 60)

    print(
        """
🔬 BAGAIMANA RAGAS MENGHITUNG ANGKA-ANGKA:

1️⃣ CONTEXT RECALL (Seberapa lengkap context vs ground truth):
   - GPT-4o bandingkan context yang diambil vs ground truth
   - Hitung: berapa % info dari ground truth ada di context
   - Formula: (Info ground truth yang ada di context) / (Total info ground truth)

2️⃣ FAITHFULNESS (Seberapa faithful answer vs context):
   - GPT-4o cek: apakah semua statement di answer didukung context?
   - Hitung: berapa % statement yang didukung vs total statement
   - Formula: (Statement didukung context) / (Total statement)

3️⃣ FACTUAL CORRECTNESS (Seberapa akurat faktual answer vs ground truth):
   - GPT-4o bandingkan fakta-fakta di answer vs ground truth
   - Hitung F1-score: precision + recall fakta yang benar
   - Formula: 2 * (Precision * Recall) / (Precision + Recall)

4️⃣ ANSWER RELEVANCY (Seberapa relevan answer vs question):
   - GPT-4o generate pertanyaan dari answer, bandingkan dengan question asli
   - Hitung semantic similarity antara questions
   - Formula: Cosine similarity average dari multiple generated questions

🤖 SEMUA DIHITUNG OLEH GPT-4o SEBAGAI EVALUATOR LLM!
    """
    )


def compare_systems_detail():
    """Bandingkan detail kedua sistem"""

    results, _ = load_latest_evaluation_results()
    if not results:
        return

    print("\n" + "=" * 60)
    print("⚖️ COMPARISON ANALYSIS")
    print("=" * 60)

    single_metrics = results["single_rag"]["evaluation_metrics"]
    multi_metrics = results["multi_agent"]["evaluation_metrics"]

    print("\n📊 DETAILED COMPARISON:")
    print("Metric                | Single RAG | Multi-Agent | Winner    | Gap")
    print("-" * 70)

    metrics_map = {
        "context_recall": "Context Recall     ",
        "faithfulness": "Faithfulness       ",
        "factual_correctness(mode=f1)": "Factual Correctness",
        "answer_relevancy": "Answer Relevancy   ",
    }

    for metric_key, metric_label in metrics_map.items():
        single_val = single_metrics.get(metric_key, 0)
        multi_val = multi_metrics.get(metric_key, 0)

        if multi_val > single_val:
            winner = "Multi-Agent"
            gap = f"+{((multi_val - single_val) / single_val * 100):.1f}%"
        elif single_val > multi_val:
            winner = "Single RAG "
            gap = f"+{((single_val - multi_val) / multi_val * 100):.1f}%"
        else:
            winner = "Tie        "
            gap = "0.0%"

        print(
            f"{metric_label} | {single_val:.3f}      | {multi_val:.3f}       | {winner} | {gap}"
        )

    # Overall comparison
    single_avg = sum([single_metrics.get(k, 0) for k in metrics_map.keys()]) / len(
        metrics_map
    )
    multi_avg = sum([multi_metrics.get(k, 0) for k in metrics_map.keys()]) / len(
        metrics_map
    )

    print("-" * 70)
    print(
        f"OVERALL AVERAGE      | {single_avg:.3f}      | {multi_avg:.3f}       | ",
        end="",
    )

    if multi_avg > single_avg:
        overall_gap = (multi_avg - single_avg) / single_avg * 100
        print(f"Multi-Agent | +{overall_gap:.1f}%")
    else:
        overall_gap = (single_avg - multi_avg) / multi_avg * 100
        print(f"Single RAG  | +{overall_gap:.1f}%")


def main():
    """Main function untuk debugging"""

    print("🐛 RAG EVALUATION DEBUGGER")
    print("=" * 60)
    print("Script ini membantu memahami bagaimana angka evaluasi dihasilkan")

    # Load dan analisis hasil
    results = load_latest_evaluation_results()
    if not results:
        print("\n💡 CARA MENJALANKAN EVALUASI:")
        print("1. python benchmarks/ragas_evaluation.py")
        print("2. python evaluate_and_visualize.py")
        print("3. python quick_visualization.py (untuk visualisasi existing)")
        return

    # Analisis proses
    analyze_evaluation_process(results[0])

    # Detail perhitungan
    show_detailed_calculation()

    # Comparison
    compare_systems_detail()

    print("\n" + "=" * 60)
    print("✅ DEBUGGING COMPLETED!")
    print("=" * 60)

    print("\n💡 UNTUK TESTING LEBIH LANJUT:")
    print("1. Edit validasi_ta.csv untuk tambah pertanyaan")
    print("2. Jalankan python benchmarks/test_endpoints_updated.py untuk test koneksi")
    print("3. Jalankan python benchmarks/ragas_evaluation.py untuk evaluasi baru")
    print("4. Jalankan python debug_evaluation.py untuk analisis hasil")


if __name__ == "__main__":
    main()

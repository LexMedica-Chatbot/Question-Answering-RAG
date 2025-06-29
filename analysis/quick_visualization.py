"""
🎯 QUICK VISUALIZATION UNTUK RAG COMPARISON
Script sederhana untuk membuat visualisasi menggunakan hasil evaluasi existing
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from glob import glob

# Set style untuk grafik
plt.style.use("default")
sns.set_palette("husl")


def load_latest_ragas_results():
    """Load hasil evaluasi Ragas terbaru"""
    results_dir = "benchmark_results"

    if not os.path.exists(results_dir):
        print("❌ Folder benchmark_results tidak ditemukan")
        return None

    # Cari file evaluasi terbaru
    pattern = os.path.join(results_dir, "ragas_evaluation_*_complete.json")
    json_files = glob(pattern)

    if not json_files:
        print("❌ Tidak ada file hasil evaluasi Ragas")
        return None

    # Ambil file terbaru
    latest_file = max(json_files, key=os.path.getctime)
    print(f"📁 Loading results dari: {os.path.basename(latest_file)}")

    try:
        with open(latest_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None


def extract_metrics_data(results):
    """Extract data metrik untuk visualisasi"""
    if not results:
        return None

    metrics_data = {}

    for system_name, system_data in results.items():
        if (
            "evaluation_metrics" in system_data
            and "error" not in system_data["evaluation_metrics"]
        ):
            metrics = system_data["evaluation_metrics"]
            system_display_name = (
                "Single RAG" if system_name == "single_rag" else "Multi-Agent RAG"
            )

            metrics_data[system_display_name] = {
                "Context Recall": metrics.get("context_recall", 0),
                "Faithfulness": metrics.get("faithfulness", 0),
                "Factual Correctness": metrics.get("factual_correctness(mode=f1)", 0),
                "Answer Relevancy": metrics.get("answer_relevancy", 0),
            }

            # Hitung average score
            avg_score = sum(metrics_data[system_display_name].values()) / len(
                metrics_data[system_display_name]
            )
            metrics_data[system_display_name]["Average Score"] = avg_score

    return metrics_data


def create_comparison_charts(metrics_data):
    """Buat grafik perbandingan"""
    if not metrics_data:
        print("❌ Tidak ada data untuk divisualisasikan")
        return

    # Create figure dengan subplots
    fig = plt.figure(figsize=(16, 12))

    # 1. Bar Chart Comparison
    ax1 = plt.subplot(2, 2, 1)

    # Prepare data untuk bar chart
    systems = list(metrics_data.keys())
    metrics = [
        "Context Recall",
        "Faithfulness",
        "Factual Correctness",
        "Answer Relevancy",
    ]

    x = np.arange(len(metrics))
    width = 0.35

    single_scores = [metrics_data[systems[0]][metric] for metric in metrics]
    multi_scores = [metrics_data[systems[1]][metric] for metric in metrics]

    bars1 = ax1.bar(
        x - width / 2,
        single_scores,
        width,
        label=systems[0],
        color="#FF6B6B",
        alpha=0.8,
    )
    bars2 = ax1.bar(
        x + width / 2, multi_scores, width, label=systems[1], color="#4ECDC4", alpha=0.8
    )

    ax1.set_xlabel("Metrik Evaluasi", fontweight="bold")
    ax1.set_ylabel("Score", fontweight="bold")
    ax1.set_title("📊 Perbandingan Metrik RAG Systems", fontweight="bold", pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, 1.1)

    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    add_value_labels(bars1)
    add_value_labels(bars2)

    # 2. Overall Score Comparison
    ax2 = plt.subplot(2, 2, 2)

    overall_scores = [metrics_data[system]["Average Score"] for system in systems]
    colors = ["#FF6B6B", "#4ECDC4"]

    bars = ax2.bar(systems, overall_scores, color=colors, alpha=0.8)
    ax2.set_title("🏆 Overall Performance Comparison", fontweight="bold", pad=20)
    ax2.set_ylabel("Average Score", fontweight="bold")
    ax2.set_ylim(0, 1)
    ax2.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar, score in zip(bars, overall_scores):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 3. Heatmap
    ax3 = plt.subplot(2, 2, 3)

    # Prepare data matrix untuk heatmap
    heatmap_data = []
    for system in systems:
        row = [metrics_data[system][metric] for metric in metrics + ["Average Score"]]
        heatmap_data.append(row)

    df_heatmap = pd.DataFrame(
        heatmap_data, index=systems, columns=metrics + ["Average Score"]
    )

    sns.heatmap(
        df_heatmap,
        annot=True,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        fmt=".3f",
        cbar_kws={"label": "Score"},
        ax=ax3,
    )
    ax3.set_title("🔥 Performance Heatmap", fontweight="bold", pad=20)

    # 4. Detailed Summary Table
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis("tight")
    ax4.axis("off")

    # Create summary table
    summary_data = []
    for system in systems:
        row = [system]
        for metric in metrics:
            row.append(f"{metrics_data[system][metric]:.1%}")
        row.append(f"{metrics_data[system]['Average Score']:.1%}")
        summary_data.append(row)

    table = ax4.table(
        cellText=summary_data,
        colLabels=["System"] + metrics + ["Average"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Style the table
    for i in range(len(systems) + 1):
        for j in range(len(metrics) + 2):
            if i == 0:  # Header row
                table[(i, j)].set_facecolor("#40466e")
                table[(i, j)].set_text_props(weight="bold", color="white")
            else:
                if j == 0:  # System names
                    table[(i, j)].set_facecolor("#f1f1f2")
                    table[(i, j)].set_text_props(weight="bold")
                else:
                    # Color based on performance
                    try:
                        value = float(summary_data[i - 1][j].rstrip("%")) / 100
                        if value >= 0.8:
                            table[(i, j)].set_facecolor("#90EE90")  # Light green
                        elif value >= 0.6:
                            table[(i, j)].set_facecolor("#FFFFE0")  # Light yellow
                        else:
                            table[(i, j)].set_facecolor("#FFB6C1")  # Light red
                    except:
                        pass

    ax4.set_title("📋 Detailed Performance Summary", fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig("rag_comparison_visualization.png", dpi=300, bbox_inches="tight")
    plt.show()

    return fig


def print_analysis(metrics_data):
    """Print analisis detail"""
    if not metrics_data:
        return

    print("\n" + "=" * 80)
    print("🔍 ANALISIS DETAIL PERBANDINGAN RAG SYSTEMS")
    print("=" * 80)

    systems = list(metrics_data.keys())
    metrics = [
        "Context Recall",
        "Faithfulness",
        "Factual Correctness",
        "Answer Relevancy",
    ]

    # Winner per metrik
    print("\n🏆 WINNER PER METRIK:")
    print("-" * 50)
    for metric in metrics:
        scores = {system: metrics_data[system][metric] for system in systems}
        winner = max(scores, key=scores.get)
        winner_score = scores[winner]
        print(f"{metric:20}: {winner:15} ({winner_score:.1%})")

    # Overall comparison
    print("\n📊 OVERALL COMPARISON:")
    print("-" * 50)
    for system in systems:
        avg_score = metrics_data[system]["Average Score"]
        print(f"{system:15}: {avg_score:.1%}")

    # Calculate advantage
    overall_scores = [metrics_data[system]["Average Score"] for system in systems]
    winner_idx = np.argmax(overall_scores)
    loser_idx = 1 - winner_idx

    winner_system = systems[winner_idx]
    winner_score = overall_scores[winner_idx]
    loser_score = overall_scores[loser_idx]

    advantage = ((winner_score - loser_score) / loser_score) * 100

    print(f"\n🎯 FINAL VERDICT:")
    print("-" * 50)
    print(f"Winner: {winner_system}")
    print(f"Performance advantage: +{advantage:.1f}%")

    # Insights
    print(f"\n💡 KEY INSIGHTS:")
    print("-" * 50)

    single_metrics = metrics_data.get("Single RAG", {})
    multi_metrics = metrics_data.get("Multi-Agent RAG", {})

    if single_metrics and multi_metrics:
        context_diff = (
            multi_metrics["Context Recall"] - single_metrics["Context Recall"]
        )
        factual_diff = (
            multi_metrics["Factual Correctness"] - single_metrics["Factual Correctness"]
        )

        print(
            f"- Context Recall: Multi-Agent {'unggul' if context_diff > 0 else 'tertinggal'} {abs(context_diff):.1%}"
        )
        print(
            f"- Factual Correctness: Multi-Agent {'unggul' if factual_diff > 0 else 'tertinggal'} {abs(factual_diff):.1%}"
        )
        print(f"- Multi-Agent RAG cocok untuk: Research, analysis, detailed inquiries")
        print(f"- Single RAG cocok untuk: Quick responses, user-facing applications")


def main():
    """Main function"""
    print("🎯 QUICK RAG VISUALIZATION SUITE")
    print("=" * 60)

    # Load hasil evaluasi
    results = load_latest_ragas_results()
    if not results:
        print("❌ Tidak dapat memuat hasil evaluasi")
        return

    # Extract metrics data
    metrics_data = extract_metrics_data(results)
    if not metrics_data:
        print("❌ Tidak dapat mengekstrak data metrik")
        return

    print("✅ Data loaded successfully")
    print(f"📊 Systems found: {list(metrics_data.keys())}")

    # Create visualizations
    print("\n🎨 Creating visualizations...")
    fig = create_comparison_charts(metrics_data)

    if fig:
        print(
            "✅ Grafik berhasil dibuat dan disimpan sebagai: rag_comparison_visualization.png"
        )

        # Print analysis
        print_analysis(metrics_data)

        print(f"\n🎉 VISUALIZATION COMPLETED!")
        print(f"📁 Check file: rag_comparison_visualization.png")
    else:
        print("❌ Gagal membuat visualisasi")


if __name__ == "__main__":
    main()

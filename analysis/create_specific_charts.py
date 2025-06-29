"""
🎯 VISUALISASI SPESIFIK UNTUK ANALISIS RAG
Script untuk membuat grafik khusus yang fokus pada aspek tertentu
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from glob import glob

# Set style
plt.style.use("default")
sns.set_palette("Set2")


def load_latest_results():
    """Load hasil evaluasi terbaru"""
    results_dir = "benchmark_results"
    pattern = os.path.join(results_dir, "ragas_evaluation_*_complete.json")
    json_files = glob(pattern)

    if not json_files:
        return None

    latest_file = max(json_files, key=os.path.getctime)

    with open(latest_file, "r", encoding="utf-8") as f:
        return json.load(f)


def create_radar_chart():
    """Buat radar chart yang fokus pada 4 metrik utama"""
    results = load_latest_results()
    if not results:
        return

    # Extract metrics
    metrics_data = {}
    for system_name, system_data in results.items():
        if "evaluation_metrics" in system_data:
            metrics = system_data["evaluation_metrics"]
            display_name = (
                "Single RAG" if system_name == "single_rag" else "Multi-Agent RAG"
            )

            metrics_data[display_name] = [
                metrics.get("context_recall", 0),
                metrics.get("faithfulness", 0),
                metrics.get("factual_correctness(mode=f1)", 0),
                metrics.get("answer_relevancy", 0),
            ]

    # Setup radar chart
    categories = [
        "Context Recall",
        "Faithfulness",
        "Factual Correctness",
        "Answer Relevancy",
    ]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

    # Calculate angle for each metric
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Colors
    colors = ["#FF6B6B", "#4ECDC4"]

    # Plot each system
    for i, (system_name, scores) in enumerate(metrics_data.items()):
        scores += scores[:1]  # Complete the circle
        ax.plot(angles, scores, "o-", linewidth=3, label=system_name, color=colors[i])
        ax.fill(angles, scores, alpha=0.25, color=colors[i])

    # Customize
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.title(
        "Radar Chart: Performance Comparison RAG Systems\n4 Key Metrics Evaluation",
        fontsize=16,
        fontweight="bold",
        pad=30,
    )
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=12)

    plt.tight_layout()
    plt.savefig("radar_chart_rag_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("✅ Radar chart disimpan: radar_chart_rag_comparison.png")


def create_performance_gaps_chart():
    """Buat chart yang menunjukkan gap performance antar sistem"""
    results = load_latest_results()
    if not results:
        return

    # Extract data
    single_metrics = results["single_rag"]["evaluation_metrics"]
    multi_metrics = results["multi_agent"]["evaluation_metrics"]

    metrics = [
        "context_recall",
        "faithfulness",
        "factual_correctness(mode=f1)",
        "answer_relevancy",
    ]
    metric_labels = [
        "Context Recall",
        "Faithfulness",
        "Factual Correctness",
        "Answer Relevancy",
    ]

    # Calculate gaps (Multi-Agent - Single)
    gaps = []
    single_scores = []
    multi_scores = []

    for metric in metrics:
        single_score = single_metrics.get(metric, 0)
        multi_score = multi_metrics.get(metric, 0)
        gap = multi_score - single_score

        gaps.append(gap)
        single_scores.append(single_score)
        multi_scores.append(multi_score)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # 1. Performance Gap Chart
    colors = ["green" if gap > 0 else "red" for gap in gaps]
    bars = ax1.barh(metric_labels, gaps, color=colors, alpha=0.7)

    ax1.axvline(x=0, color="black", linestyle="-", linewidth=1)
    ax1.set_xlabel("Performance Gap (Multi-Agent - Single)", fontweight="bold")
    ax1.set_title(
        "Performance Gap Analysis\n(Positive = Multi-Agent Better)",
        fontweight="bold",
        pad=20,
    )
    ax1.grid(axis="x", alpha=0.3)

    # Add value labels
    for i, (bar, gap) in enumerate(zip(bars, gaps)):
        width = bar.get_width()
        ax1.text(
            width + 0.01 if width >= 0 else width - 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{gap:+.1%}",
            ha="left" if width >= 0 else "right",
            va="center",
            fontweight="bold",
        )

    # 2. Side-by-side comparison
    x = np.arange(len(metric_labels))
    width = 0.35

    bars1 = ax2.bar(
        x - width / 2,
        single_scores,
        width,
        label="Single RAG",
        color="#FF6B6B",
        alpha=0.8,
    )
    bars2 = ax2.bar(
        x + width / 2,
        multi_scores,
        width,
        label="Multi-Agent RAG",
        color="#4ECDC4",
        alpha=0.8,
    )

    ax2.set_xlabel("Metrics", fontweight="bold")
    ax2.set_ylabel("Score", fontweight="bold")
    ax2.set_title("Side-by-Side Performance Comparison", fontweight="bold", pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metric_labels, rotation=45, ha="right")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, 1.1)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.1%}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig("performance_gaps_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("✅ Performance gaps chart disimpan: performance_gaps_analysis.png")


def create_strength_weakness_analysis():
    """Analisis kekuatan dan kelemahan masing-masing sistem"""
    results = load_latest_results()
    if not results:
        return

    single_metrics = results["single_rag"]["evaluation_metrics"]
    multi_metrics = results["multi_agent"]["evaluation_metrics"]

    metrics = [
        "context_recall",
        "faithfulness",
        "factual_correctness(mode=f1)",
        "answer_relevancy",
    ]
    metric_labels = [
        "Context\nRecall",
        "Faithfulness",
        "Factual\nCorrectness",
        "Answer\nRelevancy",
    ]

    single_scores = [single_metrics.get(metric, 0) for metric in metrics]
    multi_scores = [multi_metrics.get(metric, 0) for metric in metrics]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(metric_labels))
    width = 0.35

    # Create bars
    bars1 = ax.bar(
        x - width / 2,
        single_scores,
        width,
        label="Single RAG",
        color="#FF6B6B",
        alpha=0.8,
        edgecolor="darkred",
        linewidth=1,
    )
    bars2 = ax.bar(
        x + width / 2,
        multi_scores,
        width,
        label="Multi-Agent RAG",
        color="#4ECDC4",
        alpha=0.8,
        edgecolor="darkblue",
        linewidth=1,
    )

    # Customize
    ax.set_xlabel("Evaluation Metrics", fontsize=14, fontweight="bold")
    ax.set_ylabel("Performance Score", fontsize=14, fontweight="bold")
    ax.set_title(
        "Strengths & Weaknesses Analysis\nRAG Systems Performance Breakdown",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12, fontweight="bold")
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.1)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{height:.1%}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

    # Add performance threshold line
    ax.axhline(
        y=0.8,
        color="green",
        linestyle="--",
        alpha=0.7,
        label="Excellence Threshold (80%)",
    )
    ax.axhline(
        y=0.6, color="orange", linestyle="--", alpha=0.7, label="Good Threshold (60%)"
    )

    # Add annotations for best performer in each metric
    for i, (single, multi) in enumerate(zip(single_scores, multi_scores)):
        if single > multi:
            ax.annotate(
                "WINNER",
                xy=(i - width / 2, single + 0.05),
                ha="center",
                va="bottom",
                fontweight="bold",
                color="darkred",
                fontsize=10,
            )
        elif multi > single:
            ax.annotate(
                "WINNER",
                xy=(i + width / 2, multi + 0.05),
                ha="center",
                va="bottom",
                fontweight="bold",
                color="darkblue",
                fontsize=10,
            )

    plt.tight_layout()
    plt.savefig("strengths_weaknesses_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("✅ Strengths & weaknesses chart disimpan: strengths_weaknesses_analysis.png")


def create_summary_infographic():
    """Buat infographic summary hasil evaluasi"""
    results = load_latest_results()
    if not results:
        return

    single_metrics = results["single_rag"]["evaluation_metrics"]
    multi_metrics = results["multi_agent"]["evaluation_metrics"]

    # Calculate overall scores
    single_avg = (
        single_metrics.get("context_recall", 0)
        + single_metrics.get("faithfulness", 0)
        + single_metrics.get("factual_correctness(mode=f1)", 0)
        + single_metrics.get("answer_relevancy", 0)
    ) / 4

    multi_avg = (
        multi_metrics.get("context_recall", 0)
        + multi_metrics.get("faithfulness", 0)
        + multi_metrics.get("factual_correctness(mode=f1)", 0)
        + multi_metrics.get("answer_relevancy", 0)
    ) / 4

    # Determine winner
    winner = "Multi-Agent RAG" if multi_avg > single_avg else "Single RAG"
    advantage = abs(multi_avg - single_avg) / min(single_avg, multi_avg) * 100

    # Create infographic
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Overall Winner
    ax1.pie(
        [single_avg, multi_avg],
        labels=[f"Single RAG\n{single_avg:.1%}", f"Multi-Agent RAG\n{multi_avg:.1%}"],
        colors=["#FF6B6B", "#4ECDC4"],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 12, "fontweight": "bold"},
    )
    ax1.set_title(
        f"Overall Performance\nWinner: {winner}", fontsize=14, fontweight="bold", pad=20
    )

    # 2. Best metrics for each system
    single_best = max(
        [
            "context_recall",
            "faithfulness",
            "factual_correctness(mode=f1)",
            "answer_relevancy",
        ],
        key=lambda x: single_metrics.get(x, 0),
    )
    multi_best = max(
        [
            "context_recall",
            "faithfulness",
            "factual_correctness(mode=f1)",
            "answer_relevancy",
        ],
        key=lambda x: multi_metrics.get(x, 0),
    )

    labels = ["Single RAG\nBest Metric", "Multi-Agent RAG\nBest Metric"]
    values = [single_metrics.get(single_best, 0), multi_metrics.get(multi_best, 0)]

    bars = ax2.bar(labels, values, color=["#FF6B6B", "#4ECDC4"], alpha=0.8)
    ax2.set_title("Best Performance per System", fontsize=14, fontweight="bold", pad=20)
    ax2.set_ylabel("Score", fontweight="bold")
    ax2.set_ylim(0, 1.1)

    # Add metric names
    ax2.text(
        0,
        values[0] + 0.05,
        single_best.replace("_", " ").title(),
        ha="center",
        va="bottom",
        fontweight="bold",
    )
    ax2.text(
        1,
        values[1] + 0.05,
        multi_best.replace("_", " ").title(),
        ha="center",
        va="bottom",
        fontweight="bold",
    )

    for bar, value in zip(bars, values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            value / 2,
            f"{value:.1%}",
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=12,
            color="white",
        )

    # 3. Performance Matrix
    metrics_matrix = [
        [
            "Context Recall",
            single_metrics.get("context_recall", 0),
            multi_metrics.get("context_recall", 0),
        ],
        [
            "Faithfulness",
            single_metrics.get("faithfulness", 0),
            multi_metrics.get("faithfulness", 0),
        ],
        [
            "Factual Correctness",
            single_metrics.get("factual_correctness(mode=f1)", 0),
            multi_metrics.get("factual_correctness(mode=f1)", 0),
        ],
        [
            "Answer Relevancy",
            single_metrics.get("answer_relevancy", 0),
            multi_metrics.get("answer_relevancy", 0),
        ],
    ]

    ax3.axis("tight")
    ax3.axis("off")

    table_data = []
    for metric, single, multi in metrics_matrix:
        winner_mark = "🏆" if multi > single else "👑" if single > multi else "🤝"
        table_data.append([metric, f"{single:.1%}", f"{multi:.1%}", winner_mark])

    table = ax3.table(
        cellText=table_data,
        colLabels=["Metric", "Single RAG", "Multi-Agent RAG", "Winner"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)

    # Style table
    for i in range(len(metrics_matrix) + 1):
        for j in range(4):
            if i == 0:  # Header
                table[(i, j)].set_facecolor("#40466e")
                table[(i, j)].set_text_props(weight="bold", color="white")
            else:
                if j == 0:  # Metric names
                    table[(i, j)].set_facecolor("#f1f1f2")
                    table[(i, j)].set_text_props(weight="bold")
                elif j == 3:  # Winner column
                    table[(i, j)].set_facecolor("#fff2cc")

    ax3.set_title("Detailed Performance Matrix", fontsize=14, fontweight="bold", pad=20)

    # 4. Key Insights
    ax4.axis("off")
    insights = [
        f"🎯 Overall Winner: {winner}",
        f"📈 Performance Advantage: +{advantage:.1f}%",
        f"🏆 Single RAG Best At: {single_best.replace('_', ' ').title()}",
        f"🚀 Multi-Agent Best At: {multi_best.replace('_', ' ').title()}",
        "",
        "💡 Recommendations:",
        "• Single RAG: Quick responses, user apps",
        "• Multi-Agent: Research, detailed analysis",
        "",
        "🔍 Data based on 2 health law questions",
        "📊 Evaluated using Ragas framework",
    ]

    y_pos = 0.9
    for insight in insights:
        if insight == "":
            y_pos -= 0.06
            continue

        font_size = 12 if insight.startswith("🎯") or insight.startswith("📈") else 10
        font_weight = "bold" if insight.startswith(("🎯", "📈", "💡")) else "normal"

        ax4.text(
            0.05,
            y_pos,
            insight,
            fontsize=font_size,
            fontweight=font_weight,
            transform=ax4.transAxes,
            verticalalignment="top",
        )
        y_pos -= 0.08

    ax4.set_title(
        "Key Insights & Recommendations", fontsize=14, fontweight="bold", pad=20
    )

    plt.tight_layout()
    plt.savefig("rag_evaluation_summary_infographic.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("✅ Summary infographic disimpan: rag_evaluation_summary_infographic.png")


def main():
    """Main function untuk membuat semua grafik spesifik"""
    print("🎨 CREATING SPECIFIC RAG ANALYSIS CHARTS")
    print("=" * 60)

    print("\n1. 🎯 Creating Radar Chart...")
    create_radar_chart()

    print("\n2. 📊 Creating Performance Gaps Analysis...")
    create_performance_gaps_chart()

    print("\n3. 💪 Creating Strengths & Weaknesses Analysis...")
    create_strength_weakness_analysis()

    print("\n4. 📋 Creating Summary Infographic...")
    create_summary_infographic()

    print("\n🎉 ALL SPECIFIC CHARTS COMPLETED!")
    print("📁 Generated files:")
    print("   - radar_chart_rag_comparison.png")
    print("   - performance_gaps_analysis.png")
    print("   - strengths_weaknesses_analysis.png")
    print("   - rag_evaluation_summary_infographic.png")


if __name__ == "__main__":
    main()

"""
🎯 EVALUASI DAN VISUALISASI RAG SYSTEM
Script untuk mengevaluasi kedua RAG system dan membuat visualisasi perbandingan
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from benchmarks.ragas_evaluation import RAGEvaluator
import warnings

warnings.filterwarnings("ignore")

# Set style untuk grafik
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class RAGVisualizationEvaluator:
    def __init__(self, api_key: str = None, openai_api_key: str = None):
        """Initialize evaluator dengan API keys"""
        self.evaluator = RAGEvaluator(api_key=api_key, openai_api_key=openai_api_key)
        self.results = None

    def run_evaluation(self, csv_file: str = "validasi_ta.csv"):
        """Jalankan evaluasi lengkap"""
        print("🚀 Memulai evaluasi RAG systems...")
        print("=" * 60)

        try:
            self.results = self.evaluator.run_complete_evaluation(csv_file)
            print("✅ Evaluasi berhasil!")
            return self.results
        except Exception as e:
            print(f"❌ Error saat evaluasi: {e}")
            return None

    def load_latest_results(self):
        """Load hasil evaluasi terbaru jika ada"""
        try:
            results_dir = "benchmark_results"
            if not os.path.exists(results_dir):
                return None

            # Cari file evaluasi terbaru
            json_files = [
                f
                for f in os.listdir(results_dir)
                if f.startswith("ragas_evaluation_") and f.endswith("_complete.json")
            ]
            if not json_files:
                return None

            latest_file = max(
                json_files, key=lambda x: os.path.getctime(os.path.join(results_dir, x))
            )
            file_path = os.path.join(results_dir, latest_file)

            with open(file_path, "r", encoding="utf-8") as f:
                self.results = json.load(f)

            print(f"📁 Loaded results dari: {latest_file}")
            return self.results

        except Exception as e:
            print(f"❌ Error loading results: {e}")
            return None

    def extract_metrics_data(self):
        """Extract data metrik untuk visualisasi"""
        if not self.results:
            print("❌ Tidak ada hasil evaluasi. Jalankan evaluasi terlebih dahulu.")
            return None

        metrics_data = {}

        for system_name, system_data in self.results.items():
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
                    "Factual Correctness": metrics.get(
                        "factual_correctness(mode=f1)", 0
                    ),
                    "Answer Relevancy": metrics.get("answer_relevancy", 0),
                }

                # Hitung average score
                avg_score = sum(metrics_data[system_display_name].values()) / len(
                    metrics_data[system_display_name]
                )
                metrics_data[system_display_name]["Average Score"] = avg_score

        return metrics_data

    def create_radar_chart(self, metrics_data, save_path=None):
        """Buat radar chart untuk perbandingan multi-dimensi"""
        if not metrics_data:
            return None

        # Setup radar chart
        metrics = [
            "Context Recall",
            "Faithfulness",
            "Factual Correctness",
            "Answer Relevancy",
        ]

        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection="polar"))

        # Posisi sudut untuk setiap metrik
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the circle

        colors = ["#FF6B6B", "#4ECDC4"]

        for i, (system_name, system_metrics) in enumerate(metrics_data.items()):
            if system_name.endswith("Score"):  # Skip average score
                continue

            values = [system_metrics[metric] for metric in metrics]
            values += values[:1]  # Close the circle

            ax.plot(
                angles, values, "o-", linewidth=2, label=system_name, color=colors[i]
            )
            ax.fill(angles, values, alpha=0.25, color=colors[i])

        # Customize chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"])
        ax.grid(True)

        plt.title(
            "🎯 Perbandingan Performance RAG Systems\n(Radar Chart)",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"📊 Radar chart disimpan: {save_path}")

        plt.show()
        return fig

    def create_bar_comparison(self, metrics_data, save_path=None):
        """Buat bar chart untuk perbandingan per metrik"""
        if not metrics_data:
            return None

        # Prepare data
        df_data = []
        for system_name, system_metrics in metrics_data.items():
            if system_name.endswith("Score"):  # Skip average score for now
                continue
            for metric_name, score in system_metrics.items():
                if metric_name != "Average Score":
                    df_data.append(
                        {"System": system_name, "Metric": metric_name, "Score": score}
                    )

        df = pd.DataFrame(df_data)

        # Create bar chart
        fig, ax = plt.subplots(figsize=(14, 8))

        bar_plot = sns.barplot(data=df, x="Metric", y="Score", hue="System", ax=ax)

        # Customize chart
        ax.set_title(
            "📊 Perbandingan Metrik Evaluasi RAG Systems",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Metrik Evaluasi", fontsize=12, fontweight="bold")
        ax.set_ylabel("Score (0-1)", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.1)

        # Add value labels on bars
        for container in bar_plot.containers:
            bar_plot.bar_label(container, fmt="%.3f", fontsize=10)

        # Rotate x labels untuk readability
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="RAG System", title_fontsize=12, fontsize=11)
        plt.grid(axis="y", alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"📊 Bar chart disimpan: {save_path}")

        plt.show()
        return fig

    def create_heatmap(self, metrics_data, save_path=None):
        """Buat heatmap untuk visualisasi performance"""
        if not metrics_data:
            return None

        # Prepare data matrix
        systems = []
        metrics = [
            "Context Recall",
            "Faithfulness",
            "Factual Correctness",
            "Answer Relevancy",
            "Average Score",
        ]
        data_matrix = []

        for system_name, system_metrics in metrics_data.items():
            if system_name.endswith("Score"):
                continue
            systems.append(system_name)
            row = [system_metrics[metric] for metric in metrics]
            data_matrix.append(row)

        df_heatmap = pd.DataFrame(data_matrix, index=systems, columns=metrics)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 6))

        sns.heatmap(
            df_heatmap,
            annot=True,
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            fmt=".3f",
            cbar_kws={"label": "Score"},
            ax=ax,
        )

        ax.set_title(
            "🔥 Heatmap Performance RAG Systems", fontsize=16, fontweight="bold", pad=20
        )
        ax.set_xlabel("Metrik Evaluasi", fontsize=12, fontweight="bold")
        ax.set_ylabel("RAG System", fontsize=12, fontweight="bold")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"📊 Heatmap disimpan: {save_path}")

        plt.show()
        return fig

    def create_overall_comparison(self, metrics_data, save_path=None):
        """Buat grafik perbandingan overall score"""
        if not metrics_data:
            return None

        # Extract overall scores
        systems = []
        scores = []

        for system_name, system_metrics in metrics_data.items():
            if system_name.endswith("Score"):
                continue
            systems.append(system_name)
            scores.append(system_metrics["Average Score"])

        # Create comparison chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Bar chart untuk overall score
        colors = ["#FF6B6B", "#4ECDC4"]
        bars = ax1.bar(systems, scores, color=colors, alpha=0.8)

        ax1.set_title(
            "🏆 Overall Performance Comparison", fontsize=14, fontweight="bold"
        )
        ax1.set_ylabel("Average Score", fontsize=12)
        ax1.set_ylim(0, 1)
        ax1.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar, score in zip(bars, scores):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Pie chart untuk proporsi
        ax2.pie(
            scores,
            labels=systems,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 11},
        )
        ax2.set_title("📈 Performance Distribution", fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"📊 Overall comparison disimpan: {save_path}")

        plt.show()
        return fig

    def create_detailed_summary_table(self, metrics_data):
        """Buat tabel summary detail"""
        if not metrics_data:
            return None

        print("\n" + "=" * 80)
        print("📋 DETAILED EVALUATION SUMMARY")
        print("=" * 80)

        df_summary = pd.DataFrame(metrics_data).T
        df_summary = df_summary.round(4)

        # Add percentage format
        df_summary_pct = df_summary.copy()
        for col in df_summary_pct.columns:
            df_summary_pct[col] = df_summary_pct[col].apply(lambda x: f"{x:.1%}")

        print(df_summary_pct.to_string())

        # Analisis winner per metrik
        print("\n" + "=" * 50)
        print("🏆 WINNER PER METRIK:")
        print("=" * 50)

        for metric in [
            "Context Recall",
            "Faithfulness",
            "Factual Correctness",
            "Answer Relevancy",
        ]:
            if metric in df_summary.columns:
                winner = df_summary[metric].idxmax()
                score = df_summary.loc[winner, metric]
                print(f"{metric:20}: {winner:15} ({score:.1%})")

        # Overall winner
        winner = df_summary["Average Score"].idxmax()
        winner_score = df_summary.loc[winner, "Average Score"]
        loser = df_summary["Average Score"].idxmin()
        loser_score = df_summary.loc[loser, "Average Score"]
        improvement = ((winner_score - loser_score) / loser_score) * 100

        print("\n" + "=" * 50)
        print("🎯 OVERALL WINNER:")
        print("=" * 50)
        print(f"Winner: {winner} ({winner_score:.1%})")
        print(f"Performance advantage: +{improvement:.1f}%")

        return df_summary

    def generate_all_visualizations(self, run_new_evaluation=False):
        """Generate semua visualisasi sekaligus"""
        print("🎨 GENERATING RAG VISUALIZATION SUITE")
        print("=" * 60)

        # Load atau run evaluation
        if run_new_evaluation or not self.load_latest_results():
            if not self.run_evaluation():
                print("❌ Gagal menjalankan evaluasi")
                return

        # Extract metrics data
        metrics_data = self.extract_metrics_data()
        if not metrics_data:
            print("❌ Gagal mengekstrak data metrik")
            return

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"visualization_results_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        print(f"📁 Output directory: {output_dir}")

        # Generate semua visualisasi
        print("\n🎯 Creating Radar Chart...")
        self.create_radar_chart(metrics_data, f"{output_dir}/radar_chart.png")

        print("\n📊 Creating Bar Comparison...")
        self.create_bar_comparison(metrics_data, f"{output_dir}/bar_comparison.png")

        print("\n🔥 Creating Heatmap...")
        self.create_heatmap(metrics_data, f"{output_dir}/heatmap.png")

        print("\n🏆 Creating Overall Comparison...")
        self.create_overall_comparison(
            metrics_data, f"{output_dir}/overall_comparison.png"
        )

        print("\n📋 Creating Summary Table...")
        summary_df = self.create_detailed_summary_table(metrics_data)

        # Save summary table
        if summary_df is not None:
            summary_df.to_csv(f"{output_dir}/evaluation_summary.csv")
            print(f"📊 Summary table disimpan: {output_dir}/evaluation_summary.csv")

        print(f"\n✅ Semua visualisasi berhasil dibuat di: {output_dir}")
        return output_dir


def main():
    """Main function untuk menjalankan evaluasi dan visualisasi"""
    print("🎯 RAG EVALUATION & VISUALIZATION SUITE")
    print("=" * 60)

    # Get API keys
    api_key = input("Enter API key untuk RAG systems (X-API-Key): ").strip()
    if not api_key:
        print("❌ API key diperlukan untuk evaluasi RAG.")
        return

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        openai_key = input("Enter OpenAI API key (untuk Ragas): ").strip()
        if not openai_key:
            print("❌ OpenAI API key diperlukan untuk evaluasi Ragas.")
            return

    # Initialize evaluator
    visualizer = RAGVisualizationEvaluator(api_key=api_key, openai_api_key=openai_key)

    # Ask user preference
    choice = input("\nRun new evaluation? (y/n, default=n): ").strip().lower()
    run_new = choice == "y"

    # Generate all visualizations
    output_dir = visualizer.generate_all_visualizations(run_new_evaluation=run_new)

    if output_dir:
        print(f"\n🎉 Visualization suite selesai!")
        print(f"📁 Check folder: {output_dir}")
        print("📊 Grafik yang dibuat:")
        print("   - Radar Chart (multi-dimensi comparison)")
        print("   - Bar Chart (per-metric comparison)")
        print("   - Heatmap (performance overview)")
        print("   - Overall Comparison (winner analysis)")
        print("   - Summary Table (detailed metrics)")


if __name__ == "__main__":
    main()

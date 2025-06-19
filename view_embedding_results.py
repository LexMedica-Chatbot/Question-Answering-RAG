#!/usr/bin/env python3
"""
📊 Embedding Comparison Results Viewer
Script untuk menampilkan hasil perbandingan embedding dan grafik
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from datetime import datetime


def load_latest_results(results_dir="embedding_comparison_results"):
    """Load the latest embedding comparison results"""
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"❌ Results directory '{results_dir}' not found!")
        return None, None

    # Find latest files
    data_dir = results_path / "data"
    json_files = list(data_dir.glob("embedding_summary_*.json"))
    csv_files = list(data_dir.glob("embedding_comparison_*.csv"))

    if not json_files or not csv_files:
        print("❌ No results files found!")
        return None, None

    # Get latest files
    latest_json = max(json_files, key=lambda x: x.stem.split("_")[-1])
    latest_csv = max(csv_files, key=lambda x: x.stem.split("_")[-1])

    print(f"📁 Loading results from {latest_json.stem.split('_')[-1]}")

    # Load data
    with open(latest_json, "r", encoding="utf-8") as f:
        summary = json.load(f)

    df = pd.read_csv(latest_csv)

    return summary, df


def print_summary_report(summary, df):
    """Print detailed summary report"""
    print("\n" + "=" * 60)
    print("📊 EMBEDDING COMPARISON SUMMARY REPORT")
    print("=" * 60)

    # Basic info
    timestamp = df["timestamp"].iloc[0] if "timestamp" in df.columns else "Unknown"
    total_tests = len(df)

    print(f"📅 Generated: {timestamp}")
    print(f"🧪 Total Tests: {total_tests}")
    print(f"🔍 Models Compared: Small vs Large Embedding")

    print("\n📈 PERFORMANCE METRICS")
    print("-" * 40)

    # Response Time Comparison
    small_time = summary["small"]["avg_response_time_ms"]
    large_time = summary["large"]["avg_response_time_ms"]
    time_diff = abs(small_time - large_time)
    time_diff_pct = (time_diff / min(small_time, large_time)) * 100

    print(f"⚡ Response Time:")
    print(f"   Small Embedding: {small_time:.0f} ms")
    print(f"   Large Embedding: {large_time:.0f} ms")
    print(f"   Difference: {time_diff:.0f} ms ({time_diff_pct:.1f}%)")
    print(f"   Winner: {summary['comparison']['faster_model'].title()} ⭐")

    # Quality Metrics
    print(f"\n🎯 QUALITY METRICS")
    print("-" * 40)

    metrics = ["avg_keyword_score", "avg_faithfulness", "avg_answer_relevancy"]
    metric_names = ["Keyword Matching", "Faithfulness", "Answer Relevancy"]

    for metric, name in zip(metrics, metric_names):
        small_score = summary["small"][metric]
        large_score = summary["large"][metric]
        diff = large_score - small_score

        print(f"{name}:")
        print(f"   Small: {small_score:.3f}")
        print(f"   Large: {large_score:.3f}")
        print(f"   Diff: {diff:+.3f}")

    # Error Analysis
    print(f"\n🚨 ERROR ANALYSIS")
    print("-" * 40)

    small_errors = summary["small"]["error_rate"]
    large_errors = summary["large"]["error_rate"]

    print(f"Small Embedding Error Rate: {small_errors:.1%}")
    print(f"Large Embedding Error Rate: {large_errors:.1%}")

    if small_errors > 0 or large_errors > 0:
        print("⚠️  Some API calls failed - check if Simple API is running")
    else:
        print("✅ All API calls successful")

    # Content Analysis
    print(f"\n📝 CONTENT ANALYSIS")
    print("-" * 40)

    small_length = summary["small"]["avg_answer_length"]
    large_length = summary["large"]["avg_answer_length"]
    small_words = summary["small"]["avg_word_count"]
    large_words = summary["large"]["avg_word_count"]

    print(f"Average Answer Length:")
    print(f"   Small: {small_length:.0f} characters ({small_words:.1f} words)")
    print(f"   Large: {large_length:.0f} characters ({large_words:.1f} words)")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 40)

    if small_errors == 0 and large_errors == 0:
        if time_diff_pct < 10:
            print("🟡 Performance difference is minimal (<10%)")
            print("   → Choose based on quality metrics or cost considerations")
        elif summary["comparison"]["faster_model"] == "small":
            print("🟢 Small embedding is significantly faster")
            print("   → Recommended for high-volume, real-time applications")
        else:
            print("🟢 Large embedding is faster (unusual but possible)")
            print("   → Consider using large for both speed and quality")

        if summary["comparison"]["higher_quality_model"] == "large":
            print("🔵 Large embedding shows better quality metrics")
            print("   → Recommended for accuracy-critical applications")
    else:
        print("🔴 API connectivity issues detected")
        print("   → Ensure Simple API is running on http://localhost:8081")
        print("   → Check API key configuration")


def show_charts(results_dir="embedding_comparison_results"):
    """Display the generated charts"""
    charts_dir = Path(results_dir) / "charts"

    if not charts_dir.exists():
        print("❌ Charts directory not found!")
        return

    chart_files = list(charts_dir.glob("embedding_comparison_*.png"))

    if not chart_files:
        print("❌ No chart files found!")
        return

    latest_chart = max(chart_files, key=lambda x: x.stem.split("_")[-1])

    print(f"\n📊 Displaying chart: {latest_chart.name}")

    try:
        from PIL import Image

        img = Image.open(latest_chart)
        img.show()
        print("✅ Chart displayed successfully!")
    except ImportError:
        print("⚠️  PIL not available. Chart saved at:")
        print(f"   {latest_chart.absolute()}")
    except Exception as e:
        print(f"❌ Error displaying chart: {e}")
        print(f"Chart available at: {latest_chart.absolute()}")


def export_markdown_report(summary, df, results_dir="embedding_comparison_results"):
    """Export results to markdown report"""
    timestamp = (
        df["timestamp"].iloc[0]
        if "timestamp" in df.columns
        else datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    report = f"""# Embedding Comparison Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Data Timestamp:** {timestamp}

## Executive Summary

Perbandingan performa antara Text Embedding Small vs Large pada Simple API untuk sistem RAG hukum kesehatan Indonesia.

## Performance Results

### Response Time
- **Small Embedding:** {summary['small']['avg_response_time_ms']:.0f} ms
- **Large Embedding:** {summary['large']['avg_response_time_ms']:.0f} ms
- **Winner:** {summary['comparison']['faster_model'].title()} Embedding

### Quality Metrics
- **Keyword Matching:**
  - Small: {summary['small']['avg_keyword_score']:.3f}
  - Large: {summary['large']['avg_keyword_score']:.3f}
- **Faithfulness:**
  - Small: {summary['small']['avg_faithfulness']:.3f}
  - Large: {summary['large']['avg_faithfulness']:.3f}
- **Answer Relevancy:**
  - Small: {summary['small']['avg_answer_relevancy']:.3f}
  - Large: {summary['large']['avg_answer_relevancy']:.3f}

### Error Rates
- **Small Embedding:** {summary['small']['error_rate']:.1%}
- **Large Embedding:** {summary['large']['error_rate']:.1%}

## Recommendations

Based on the results:
- **Speed Priority:** Use {summary['comparison']['faster_model'].title()} Embedding
- **Quality Priority:** Use {summary['comparison']['higher_quality_model'].title()} Embedding
- **Balanced Use:** Consider your specific use case requirements

## Technical Details

- Total test cases: {len(df)}
- API endpoint: Simple API (Single-step RAG)
- Embedding models: text-embedding-3-small vs text-embedding-3-large
- Test categories: Definisi, Perizinan, Sanksi, Prosedur, Etika, Analisis

---
*Generated by Embedding Comparison Benchmark System*
"""

    report_path = Path(results_dir) / f"embedding_report_{timestamp}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"📝 Markdown report exported to: {report_path}")


def main():
    """Main function"""
    print("📊 Embedding Comparison Results Viewer")
    print("=" * 50)

    # Load results
    summary, df = load_latest_results()

    if summary is None or df is None:
        print("❌ Failed to load results. Run the benchmark first:")
        print("   python benchmarks/simple_embedding_comparison.py --dry-run")
        return

    # Print summary report
    print_summary_report(summary, df)

    # Check command line arguments
    if len(sys.argv) > 1:
        if "--chart" in sys.argv:
            show_charts()
        if "--export" in sys.argv:
            export_markdown_report(summary, df)
    else:
        print(f"\n🔧 Additional Options:")
        print(f"   python {sys.argv[0]} --chart    # Display charts")
        print(f"   python {sys.argv[0]} --export   # Export markdown report")
        print(f"   python {sys.argv[0]} --chart --export  # Both")


if __name__ == "__main__":
    main()

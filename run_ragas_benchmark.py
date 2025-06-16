#!/usr/bin/env python3
"""
🔬 RAGAs Benchmark Runner
Script untuk menjalankan comprehensive RAGAs evaluation dengan 6 kombinasi:

1. Simple API + Small Embedding
2. Simple API + Large Embedding
3. Multi API (Parallel) + Small Embedding
4. Multi API (Parallel) + Large Embedding
5. Multi API (Sequential) + Small Embedding
6. Multi API (Sequential) + Large Embedding

Menggunakan framework RAGAs untuk evaluasi akademik yang rigorous.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add benchmarks directory to path
sys.path.append(str(Path(__file__).parent / "benchmarks"))

from ragas_benchmark import RAGAsBenchmark, BenchmarkConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function"""
    logger.info("🚀 Starting RAGAs Benchmark with 6 Combinations")
    logger.info("=" * 80)

    # Configuration
    config = BenchmarkConfig(
        api_base_url="http://localhost:8080",
        api_key="your_secure_api_key_here",
        output_dir="ragas_benchmark_results",
        timeout=120,  # 2 minutes timeout
        max_retries=3,
    )

    logger.info("📊 Test Combinations:")
    logger.info("1. Simple API + Small Embedding")
    logger.info("2. Simple API + Large Embedding")
    logger.info("3. Multi API (Parallel) + Small Embedding")
    logger.info("4. Multi API (Parallel) + Large Embedding")
    logger.info("5. Multi API (Sequential) + Small Embedding")
    logger.info("6. Multi API (Sequential) + Large Embedding")
    logger.info("=" * 80)

    # Initialize benchmark
    benchmark = RAGAsBenchmark(config)

    # Check if RAGAs is available
    try:
        from ragas import evaluate

        logger.info("✅ RAGAs framework is available")
    except ImportError:
        logger.error("❌ RAGAs not installed. Please install with: pip install ragas")
        logger.info(
            "💡 Alternative: Use complete_benchmark.py for custom quality scoring"
        )
        return

    # Run benchmark
    try:
        logger.info("🔄 Starting comprehensive RAGAs evaluation...")
        logger.info("⏱️  Estimated time: 20-30 minutes for all combinations")

        # Run async benchmark
        results = asyncio.run(benchmark.run_full_benchmark())

        # Generate analysis
        logger.info("📊 Analyzing results...")
        summary = benchmark.generate_summary_report(results)

        # Save results
        timestamp = benchmark.save_results(results, summary)

        # Generate LaTeX report
        benchmark.generate_latex_report(summary, timestamp)

        # Print summary
        logger.info("=" * 80)
        logger.info("✅ RAGAs BENCHMARK COMPLETED SUCCESSFULLY!")
        logger.info(f"📁 Results saved with timestamp: {timestamp}")
        logger.info("=" * 80)

        print("\n📊 QUICK SUMMARY:")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Combinations: 6 (Simple×2 + Multi-Parallel×2 + Multi-Sequential×2)")

        if summary["summary_by_combination"]:
            print("\n🏆 TOP PERFORMERS:")

            # Best RAGAs score
            best_combo = summary["comparison"]["best_overall_ragas"]
            print(
                f"Best RAGAs Score: {best_combo['combination']} ({best_combo['score']:.3f})"
            )

            # Fastest response
            fastest_combo = summary["comparison"]["fastest_response"]
            print(
                f"Fastest Response: {fastest_combo['combination']} ({fastest_combo['avg_time_ms']:.0f}ms)"
            )

            # Most comprehensive
            comprehensive_combo = summary["comparison"]["most_comprehensive"]
            print(
                f"Most Comprehensive: {comprehensive_combo['combination']} ({comprehensive_combo['avg_contexts']:.1f} docs)"
            )

        print(f"\n📄 Files generated:")
        print(
            f"- RAGAs Summary: ragas_benchmark_results/reports/benchmark_summary_{timestamp}.json"
        )
        print(
            f"- Raw Data: ragas_benchmark_results/raw_data/benchmark_raw_{timestamp}.json"
        )
        print(
            f"- CSV Data: ragas_benchmark_results/reports/benchmark_data_{timestamp}.csv"
        )
        print(
            f"- LaTeX Report: ragas_benchmark_results/reports/benchmark_report_{timestamp}.tex"
        )

        logger.info("🎓 Ready for thesis analysis!")

    except KeyboardInterrupt:
        logger.info("⚠️ Benchmark interrupted by user")
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        logger.info("💡 Check API availability and configuration")


if __name__ == "__main__":
    main()

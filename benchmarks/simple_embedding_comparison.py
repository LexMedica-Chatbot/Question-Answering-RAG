#!/usr/bin/env python3
"""
🔬 Simple API Embedding Comparison Benchmark
Membandingkan performa Text Embedding Small vs Large pada Simple API saja
Dengan visualisasi grafik untuk analisis
"""

import asyncio
import json
import time
import pandas as pd
import httpx  # async HTTP client
import matplotlib.pyplot as plt

# seaborn opsional – aktifkan jika ingin theme
# import seaborn as sns
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
from pathlib import Path
import logging
import os
from dotenv import load_dotenv
import concurrent.futures
from threading import Lock

# Load environment variables from .env file
load_dotenv()

# RAGAs imports
try:
    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.types import Score

    RAGAS_AVAILABLE = True
    print("✅ RAGAs library available")
except ImportError as e:
    print(f"⚠️ RAGAs not available: {e}")
    RAGAS_AVAILABLE = False

# Setup matplotlib
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Konfigurasi benchmark"""

    api_base_url: str = "http://localhost:8080"  # Simple API port (Docker)
    api_key: str = "your_secure_api_key_here"
    output_dir: str = "embedding_comparison_results"
    timeout: int = 60
    dry_run: bool = False  # If True, use mock data instead of real API calls


@dataclass
class TestCase:
    """Test case untuk benchmark"""

    question: str
    category: str = "general"
    difficulty: str = "medium"
    expected_keywords: Optional[List[str]] = None


@dataclass
class APIResponse:
    """Response dari API"""

    answer: str
    contexts: List[str]
    response_time_ms: int
    model_info: Dict[str, Any]
    embedding_model: str
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Hasil benchmark untuk satu test case"""

    test_case: TestCase
    api_response: APIResponse
    ragas_scores: Dict[str, float]
    performance_metrics: Dict[str, Any]


class EmbeddingComparisonBenchmark:
    """Main benchmark class for embedding comparison"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.setup_output_directory()

    def setup_output_directory(self):
        """Setup direktori output"""
        self.output_path = Path(self.config.output_dir)
        self.output_path.mkdir(exist_ok=True)

        # Create subdirectories
        (self.output_path / "charts").mkdir(exist_ok=True)
        (self.output_path / "data").mkdir(exist_ok=True)

    def create_test_cases(self) -> List[TestCase]:
        """Buat test cases yang fokus pada hukum kesehatan Indonesia"""
        return [
            # Easy Cases - Definisi Basic
            TestCase(
                question="Apa definisi tenaga kesehatan menurut undang-undang?",
                category="definisi",
                difficulty="easy",
                expected_keywords=["tenaga kesehatan", "undang-undang", "definisi"],
            ),
            TestCase(
                question="Siapa yang berwenang mengeluarkan izin praktik dokter?",
                category="perizinan",
                difficulty="easy",
                expected_keywords=["izin praktik", "dokter", "berwenang"],
            ),
            TestCase(
                question="Apa kepanjangan dari STR dalam konteks praktik kedokteran?",
                category="perizinan",
                difficulty="easy",
                expected_keywords=[
                    "STR",
                    "surat tanda registrasi",
                    "praktik kedokteran",
                ],
            ),
            # Medium Cases - Prosedur dan Sanksi
            TestCase(
                question="Apa sanksi hukum bagi praktik kedokteran tanpa izin?",
                category="sanksi",
                difficulty="medium",
                expected_keywords=[
                    "sanksi",
                    "praktik kedokteran",
                    "tanpa izin",
                    "hukum",
                ],
            ),
            TestCase(
                question="Bagaimana prosedur pengajuan izin praktik dokter spesialis?",
                category="prosedur",
                difficulty="medium",
                expected_keywords=["prosedur", "izin praktik", "dokter spesialis"],
            ),
            TestCase(
                question="Apa kewajiban dokter dalam memberikan informed consent?",
                category="etika",
                difficulty="medium",
                expected_keywords=["kewajiban", "dokter", "informed consent"],
            ),
            TestCase(
                question="Bagaimana ketentuan rekam medis menurut peraturan yang berlaku?",
                category="administrasi",
                difficulty="medium",
                expected_keywords=["rekam medis", "ketentuan", "peraturan"],
            ),
            # Hard Cases - Analisis Kompleks
            TestCase(
                question="Bagaimana hubungan antara UU Praktik Kedokteran dengan UU Rumah Sakit dalam hal tanggung jawab medis?",
                category="analisis",
                difficulty="hard",
                expected_keywords=[
                    "UU Praktik Kedokteran",
                    "UU Rumah Sakit",
                    "tanggung jawab medis",
                ],
            ),
            TestCase(
                question="Apa perbedaan sanksi pidana dan perdata dalam kasus malpraktik medis?",
                category="analisis",
                difficulty="hard",
                expected_keywords=[
                    "sanksi pidana",
                    "sanksi perdata",
                    "malpraktik medis",
                ],
            ),
        ]

    async def call_simple_api(self, question: str, embedding_model: str) -> APIResponse:
        """Call Simple API endpoint"""

        # For dry run, return mock data
        if self.config.dry_run:
            import random

            return APIResponse(
                answer=f"This is a mock answer for '{question[:30]}...' using {embedding_model} embedding.",
                contexts=[
                    f"Mock context 1 for {embedding_model}",
                    f"Mock context 2 for {embedding_model}",
                ],
                response_time_ms=random.randint(1000, 3000),
                model_info={"model": f"mock-{embedding_model}"},
                embedding_model=embedding_model,
            )

        url = f"{self.config.api_base_url}/api/chat"
        headers = {"X-API-Key": self.config.api_key, "Content-Type": "application/json"}

        payload = {
            "query": question,
            "embedding_model": embedding_model,
            "previous_responses": [],
        }

        start_time = time.perf_counter()

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                resp = await client.post(url, json=payload, headers=headers)

            response_time_ms = int((time.perf_counter() - start_time) * 1000)

            if resp.status_code == 200:
                data = resp.json()

                # Extract contexts from referenced_documents
                contexts = []
                if "referenced_documents" in data:
                    for doc in data["referenced_documents"]:
                        if isinstance(doc, dict) and "content" in doc:
                            contexts.append(doc["content"])
                        elif isinstance(doc, str):
                            contexts.append(doc)

                return APIResponse(
                    answer=data.get("answer", ""),
                    contexts=contexts,
                    response_time_ms=response_time_ms,
                    model_info=data.get("model_info", {}),
                    embedding_model=embedding_model,
                )
            else:
                logger.warning(
                    f"API call failed with status {resp.status_code}: {resp.text}"
                )
                return APIResponse(
                    answer="",
                    contexts=[],
                    response_time_ms=response_time_ms,
                    model_info={},
                    embedding_model=embedding_model,
                    error=f"HTTP {resp.status_code}: {resp.text}",
                )

        except Exception as e:
            response_time_ms = int((time.perf_counter() - start_time) * 1000)
            logger.error(f"API call exception: {e}")
            return APIResponse(
                answer="",
                contexts=[],
                response_time_ms=response_time_ms,
                model_info={},
                embedding_model=embedding_model,
                error=str(e),
            )

    def calculate_ragas_scores(
        self, question: str, answer: str, contexts: List[str]
    ) -> Dict[str, float]:
        """Calculate RAGAs scores using new API"""
        if not RAGAS_AVAILABLE or not contexts or not answer.strip():
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "context_relevancy": 0.0,
            }

        # Check if OpenAI API key is available
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning(
                "OPENAI_API_KEY not found in environment variables. Skipping RAGAs evaluation."
            )
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "context_relevancy": 0.0,
            }

        try:
            # Initialize LLM and embeddings for RAGAs
            evaluator_llm = LangchainLLMWrapper(
                ChatOpenAI(model="gpt-4.1", temperature=0, api_key=openai_api_key)
            )

            evaluator_embeddings = LangchainEmbeddingsWrapper(
                OpenAIEmbeddings(api_key=openai_api_key)
            )

            # Initialize metrics with LLM only (embeddings are handled automatically)
            faithfulness = Faithfulness(llm=evaluator_llm)
            answer_relevancy = AnswerRelevancy(
                llm=evaluator_llm, embeddings=evaluator_embeddings
            )
            context_precision = ContextPrecision(llm=evaluator_llm)
            context_recall = ContextRecall(llm=evaluator_llm)

            # Create SingleTurnSample
            sample = SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
                reference=answer,  # Using answer as ground truth for now
            )

            # Calculate scores individually
            faithfulness_score = faithfulness.single_turn_score(sample)
            answer_relevancy_score = answer_relevancy.single_turn_score(sample)
            context_precision_score = context_precision.single_turn_score(sample)
            context_recall_score = context_recall.single_turn_score(sample)

            def _to_float(s):
                return s.value if isinstance(s, Score) else float(s)

            return {
                "faithfulness": _to_float(faithfulness_score),
                "answer_relevancy": _to_float(answer_relevancy_score),
                "context_precision": _to_float(context_precision_score),
                "context_recall": _to_float(context_recall_score),
                "context_relevancy": 0.0,  # Not available in current RAGAs version
            }

        except Exception as e:
            logger.error(f"RAGAs evaluation failed: {e}")
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "context_relevancy": 0.0,
            }

    def calculate_keyword_score(
        self, answer: str, expected_keywords: List[str]
    ) -> float:
        """Calculate keyword matching score"""
        if not expected_keywords:
            return 1.0

        answer_lower = answer.lower()
        matched_keywords = sum(
            1 for keyword in expected_keywords if keyword.lower() in answer_lower
        )
        return matched_keywords / len(expected_keywords)

    async def run_single_benchmark(
        self, test_case: TestCase, embedding_model: str
    ) -> BenchmarkResult:
        """Run benchmark untuk satu test case"""
        logger.info(f"Testing: Simple API, {embedding_model} embedding")
        logger.info(f"Question: {test_case.question[:60]}...")

        # Call API
        api_response = await self.call_simple_api(test_case.question, embedding_model)

        # Calculate RAGAs scores
        ragas_scores = self.calculate_ragas_scores(
            test_case.question, api_response.answer, api_response.contexts
        )

        # Calculate additional performance metrics
        keyword_score = self.calculate_keyword_score(
            api_response.answer, test_case.expected_keywords or []
        )

        performance_metrics = {
            "response_time_ms": api_response.response_time_ms,
            "answer_length": len(api_response.answer),
            "answer_word_count": len(api_response.answer.split()),
            "num_contexts": len(api_response.contexts),
            "contexts_total_length": sum(len(ctx) for ctx in api_response.contexts),
            "has_error": api_response.error is not None,
            "keyword_score": keyword_score,
        }

        return BenchmarkResult(
            test_case=test_case,
            api_response=api_response,
            ragas_scores=ragas_scores,
            performance_metrics=performance_metrics,
        )

    async def run_full_benchmark(self) -> List[BenchmarkResult]:
        """Run full benchmark for both embedding models with multi-threading"""
        test_cases = self.create_test_cases()
        embedding_models = ["small", "large"]

        # Create all combinations of test cases and models
        tasks = []
        for test_case in test_cases:
            for embedding_model in embedding_models:
                tasks.append((test_case, embedding_model))

        total_tests = len(tasks)
        logger.info(f"🚀 Starting embedding comparison with {total_tests} total tests")

        # Use ThreadPoolExecutor for concurrent API calls
        results = []
        completed = 0
        progress_lock = Lock()

        def run_benchmark_sync(task_data):
            test_case, embedding_model = task_data

            # Run the async function in a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.run_single_benchmark(test_case, embedding_model)
                )

                # Thread-safe progress tracking
                nonlocal completed
                with progress_lock:
                    completed += 1
                    logger.info(f"Progress: {completed}/{total_tests}")

                return result
            finally:
                loop.close()

        # Use ThreadPoolExecutor with max 4 workers to avoid overwhelming the API
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_task = {
                executor.submit(run_benchmark_sync, task): task for task in tasks
            }

            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    task = future_to_task[future]
                    logger.error(f"Task {task} failed with error: {e}")

        return results

    def generate_summary_report(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate summary report"""
        df_data = []

        for result in results:
            row = {
                "embedding_model": result.api_response.embedding_model,
                "category": result.test_case.category,
                "difficulty": result.test_case.difficulty,
                "response_time_ms": result.performance_metrics["response_time_ms"],
                "answer_length": result.performance_metrics["answer_length"],
                "answer_word_count": result.performance_metrics["answer_word_count"],
                "num_contexts": result.performance_metrics["num_contexts"],
                "has_error": result.performance_metrics["has_error"],
                "keyword_score": result.performance_metrics["keyword_score"],
                **result.ragas_scores,
            }
            df_data.append(row)

        df = pd.DataFrame(df_data)

        # Calculate summary statistics
        summary = {}

        for embedding_model in df["embedding_model"].unique():
            subset = df[df["embedding_model"] == embedding_model]

            if len(subset) > 0:
                summary[embedding_model] = {
                    "avg_response_time_ms": float(subset["response_time_ms"].mean()),
                    "median_response_time_ms": float(
                        subset["response_time_ms"].median()
                    ),
                    "avg_answer_length": float(subset["answer_length"].mean()),
                    "avg_word_count": float(subset["answer_word_count"].mean()),
                    "avg_num_contexts": float(subset["num_contexts"].mean()),
                    "error_rate": float(subset["has_error"].mean()),
                    "avg_keyword_score": float(subset["keyword_score"].mean()),
                    "avg_faithfulness": float(subset["faithfulness"].mean()),
                    "avg_answer_relevancy": float(subset["answer_relevancy"].mean()),
                    "avg_context_precision": float(subset["context_precision"].mean()),
                    "avg_context_recall": float(subset["context_recall"].mean()),
                    "avg_context_relevancy": float(subset["context_relevancy"].mean()),
                    "total_tests": len(subset),
                }

        # Overall comparison
        summary["comparison"] = {
            "faster_model": min(
                summary.keys(), key=lambda x: summary[x]["avg_response_time_ms"]
            ),
            "higher_quality_model": max(
                summary.keys(), key=lambda x: summary[x]["avg_faithfulness"]
            ),
            "better_relevancy_model": max(
                summary.keys(), key=lambda x: summary[x]["avg_answer_relevancy"]
            ),
        }

        return {"summary": summary, "detailed_data": df}

    def create_visualizations(self, summary_data: Dict[str, Any], timestamp: str):
        """Create comparison charts"""
        df = summary_data["detailed_data"]
        summary = summary_data["summary"]

        # 1. Response Time Comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Response time by embedding model
        response_times = [
            summary["small"]["avg_response_time_ms"],
            summary["large"]["avg_response_time_ms"],
        ]
        models = ["Small Embedding", "Large Embedding"]
        colors = ["#3498db", "#e74c3c"]

        bars1 = ax1.bar(models, response_times, color=colors, alpha=0.7)
        ax1.set_title("Rata-rata Response Time (ms)", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Response Time (ms)")

        # Add value labels on bars
        for bar, value in zip(bars1, response_times):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 100,
                f"{value:.0f}ms",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 2. RAGAs Metrics Comparison
        metrics = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "context_relevancy",
        ]
        small_scores = [summary["small"][f"avg_{metric}"] for metric in metrics]
        large_scores = [summary["large"][f"avg_{metric}"] for metric in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        bars2 = ax2.bar(
            x - width / 2,
            small_scores,
            width,
            label="Small Embedding",
            color="#3498db",
            alpha=0.7,
        )
        bars3 = ax2.bar(
            x + width / 2,
            large_scores,
            width,
            label="Large Embedding",
            color="#e74c3c",
            alpha=0.7,
        )

        ax2.set_title("Perbandingan RAGAs Metrics", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Score")
        ax2.set_xticks(x)
        ax2.set_xticklabels(
            [m.replace("_", "\n") for m in metrics], rotation=45, ha="right"
        )
        ax2.legend()
        ax2.set_ylim(0, 1)

        # 3. Performance by Difficulty
        difficulty_order = ["easy", "medium", "hard"]
        small_by_diff = (
            df[df["embedding_model"] == "small"]
            .groupby("difficulty")["response_time_ms"]
            .mean()
        )
        large_by_diff = (
            df[df["embedding_model"] == "large"]
            .groupby("difficulty")["response_time_ms"]
            .mean()
        )

        x_diff = np.arange(len(difficulty_order))
        small_times = [small_by_diff.get(diff, 0) for diff in difficulty_order]
        large_times = [large_by_diff.get(diff, 0) for diff in difficulty_order]

        ax3.plot(
            x_diff,
            small_times,
            "o-",
            label="Small Embedding",
            color="#3498db",
            linewidth=2,
            markersize=8,
        )
        ax3.plot(
            x_diff,
            large_times,
            "s-",
            label="Large Embedding",
            color="#e74c3c",
            linewidth=2,
            markersize=8,
        )

        ax3.set_title(
            "Response Time berdasarkan Tingkat Kesulitan",
            fontsize=14,
            fontweight="bold",
        )
        ax3.set_ylabel("Response Time (ms)")
        ax3.set_xlabel("Tingkat Kesulitan")
        ax3.set_xticks(x_diff)
        ax3.set_xticklabels([d.replace("_", " ").title() for d in difficulty_order])
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Answer Quality Metrics
        quality_metrics = ["keyword_score", "faithfulness", "answer_relevancy"]
        small_quality = [
            summary["small"][f"avg_{metric}"] for metric in quality_metrics
        ]
        large_quality = [
            summary["large"][f"avg_{metric}"] for metric in quality_metrics
        ]

        x_qual = np.arange(len(quality_metrics))

        bars4 = ax4.bar(
            x_qual - width / 2,
            small_quality,
            width,
            label="Small Embedding",
            color="#3498db",
            alpha=0.7,
        )
        bars5 = ax4.bar(
            x_qual + width / 2,
            large_quality,
            width,
            label="Large Embedding",
            color="#e74c3c",
            alpha=0.7,
        )

        ax4.set_title("Perbandingan Kualitas Jawaban", fontsize=14, fontweight="bold")
        ax4.set_ylabel("Score")
        ax4.set_xticks(x_qual)
        ax4.set_xticklabels(["Keyword\nMatching", "Faithfulness", "Answer\nRelevancy"])
        ax4.legend()
        ax4.set_ylim(0, 1)

        # Add value labels on quality bars
        for bars in [bars4, bars5]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.01,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        plt.tight_layout()
        plt.savefig(
            self.output_path / "charts" / f"embedding_comparison_{timestamp}.png",
            dpi=300,
            bbox_inches="tight",
        )

        logger.info(f"📊 Charts saved to {self.output_path}/charts/")

        plt.close("all")  # Close all figures to free memory

    def save_results(
        self, results: List[BenchmarkResult], summary: Dict[str, Any], timestamp: str
    ):
        """Save all results to files"""

        # Save detailed results to CSV
        df_data = []
        for result in results:
            row = {
                "timestamp": timestamp,
                "embedding_model": result.api_response.embedding_model,
                "question": result.test_case.question,
                "category": result.test_case.category,
                "difficulty": result.test_case.difficulty,
                "answer": result.api_response.answer,
                "response_time_ms": result.performance_metrics["response_time_ms"],
                "answer_length": result.performance_metrics["answer_length"],
                "answer_word_count": result.performance_metrics["answer_word_count"],
                "num_contexts": result.performance_metrics["num_contexts"],
                "has_error": result.performance_metrics["has_error"],
                "keyword_score": result.performance_metrics["keyword_score"],
                **result.ragas_scores,
            }
            df_data.append(row)

        df = pd.DataFrame(df_data)
        csv_path = self.output_path / "data" / f"embedding_comparison_{timestamp}.csv"
        df.to_csv(csv_path, index=False)

        # Save summary to JSON (exclude DataFrame)
        json_path = self.output_path / "data" / f"embedding_summary_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary["summary"], f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Results saved to {self.output_path}/data/")


async def main():
    """Main function"""
    print("🚀 Starting Embedding Comparison Benchmark")
    print("=" * 60)

    # Check if we want to run in dry-run mode
    import sys

    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("🧪 Running in DRY-RUN mode with mock data")

    # Initialize benchmark
    config = BenchmarkConfig(dry_run=dry_run)
    benchmark = EmbeddingComparisonBenchmark(config)

    # Run benchmark
    print("📊 Running benchmark tests...")
    results = await benchmark.run_full_benchmark()

    # Generate summary
    print("📈 Generating analysis...")
    summary_data = benchmark.generate_summary_report(results)

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save results
    print("💾 Saving results...")
    benchmark.save_results(results, summary_data, timestamp)

    # Generate visualizations
    print("📊 Creating charts...")
    benchmark.create_visualizations(summary_data, timestamp)

    print("✅ Benchmark completed!")
    print(f"📁 Results saved in: {benchmark.output_path}")
    print(f"🔍 Check the charts in: {benchmark.output_path}/charts/")
    print(f"📊 Data files in: {benchmark.output_path}/data/")

    # Print quick summary
    summary = summary_data["summary"]
    print("\n📋 Quick Summary:")
    print(
        f"Small Embedding - Avg Response Time: {summary['small']['avg_response_time_ms']:.0f}ms"
    )
    print(
        f"Large Embedding - Avg Response Time: {summary['large']['avg_response_time_ms']:.0f}ms"
    )
    print(f"Faster Model: {summary['comparison']['faster_model'].title()}")

    if not dry_run:
        print(f"Error Rate - Small: {summary['small']['error_rate']:.1%}")
        print(f"Error Rate - Large: {summary['large']['error_rate']:.1%}")


if __name__ == "__main__":
    asyncio.run(main())

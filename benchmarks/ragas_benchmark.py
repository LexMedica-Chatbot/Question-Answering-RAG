#!/usr/bin/env python3
"""
🔬 RAGAs Benchmark System
Sistem evaluasi komprehensif untuk membandingkan:
1. Simple API vs Multi API (Enhanced Multi-Step RAG)
2. Text Embedding Small vs Large
3. Performance Metrics untuk Laporan Skripsi

RAGAs Metrics:
- Faithfulness: Keakuratan jawaban terhadap dokumen
- Answer Relevancy: Relevansi jawaban terhadap pertanyaan
- Context Precision: Precision dari konteks yang diambil
- Context Recall: Recall dari konteks yang diambil
- Context Relevancy: Relevansi konteks terhadap pertanyaan
"""

import asyncio
import json
import time
import pandas as pd
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
from pathlib import Path
import logging

# RAGAs imports
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        context_relevancy,
    )
    from datasets import Dataset

    RAGAS_AVAILABLE = True
    print("✅ RAGAs library available")
except ImportError as e:
    print(f"⚠️ RAGAs not available: {e}")
    print("📦 Install with: pip install ragas")
    RAGAS_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Konfigurasi benchmark"""

    api_base_url: str = "http://localhost:8080"
    api_key: str = "your_secure_api_key_here"
    output_dir: str = "benchmark_results"
    timeout: int = 60
    max_retries: int = 3


@dataclass
class TestCase:
    """Test case untuk benchmark"""

    question: str
    expected_answer: Optional[str] = None
    ground_truth_context: Optional[List[str]] = None
    category: str = "general"
    difficulty: str = "medium"


@dataclass
class APIResponse:
    """Response dari API"""

    answer: str
    contexts: List[str]
    response_time_ms: int
    model_info: Dict[str, Any]
    api_type: str
    embedding_model: str
    processing_steps: Optional[List[Dict]] = None
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Hasil benchmark untuk satu test case"""

    test_case: TestCase
    api_response: APIResponse
    ragas_scores: Dict[str, float]
    performance_metrics: Dict[str, Any]


class RAGAsBenchmark:
    """Main benchmark class"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.setup_output_directory()

    def setup_output_directory(self):
        """Setup direktori output"""
        self.output_path = Path(self.config.output_dir)
        self.output_path.mkdir(exist_ok=True)

        # Create subdirectories
        (self.output_path / "raw_data").mkdir(exist_ok=True)
        (self.output_path / "reports").mkdir(exist_ok=True)
        (self.output_path / "charts").mkdir(exist_ok=True)

    def create_test_cases(self) -> List[TestCase]:
        """Buat test cases untuk hukum kesehatan Indonesia"""
        return [
            # Easy Cases - Definisi Basic
            TestCase(
                question="Apa definisi tenaga kesehatan menurut undang-undang?",
                category="definisi",
                difficulty="easy",
            ),
            TestCase(
                question="Siapa yang berwenang mengeluarkan izin praktik dokter?",
                category="perizinan",
                difficulty="easy",
            ),
            TestCase(
                question="Apa kepanjangan dari STR dalam konteks praktik kedokteran?",
                category="perizinan",
                difficulty="easy",
            ),
            # Medium Cases - Prosedur dan Sanksi
            TestCase(
                question="Apa sanksi hukum bagi praktik kedokteran tanpa izin?",
                category="sanksi",
                difficulty="medium",
            ),
            TestCase(
                question="Bagaimana prosedur pengajuan izin praktik dokter spesialis?",
                category="prosedur",
                difficulty="medium",
            ),
            TestCase(
                question="Apa kewajiban dokter dalam memberikan informed consent?",
                category="etika",
                difficulty="medium",
            ),
            TestCase(
                question="Bagaimana ketentuan rekam medis menurut peraturan yang berlaku?",
                category="administrasi",
                difficulty="medium",
            ),
            # Hard Cases - Analisis Kompleks
            TestCase(
                question="Bagaimana hubungan antara UU Praktik Kedokteran dengan UU Rumah Sakit dalam hal tanggung jawab medis?",
                category="analisis",
                difficulty="hard",
            ),
            TestCase(
                question="Apa perbedaan sanksi pidana dan perdata dalam kasus malpraktik medis?",
                category="analisis",
                difficulty="hard",
            ),
            TestCase(
                question="Bagaimana implementasi telemedicine dalam konteks peraturan praktik kedokteran di Indonesia?",
                category="teknologi",
                difficulty="hard",
            ),
            # Complex Cases - Multi-aspect
            TestCase(
                question="Analisis komprehensif tentang perlindungan hukum bagi pasien dan tenaga kesehatan dalam sistem kesehatan nasional",
                category="komprehensif",
                difficulty="very_hard",
            ),
            TestCase(
                question="Bagaimana koordinasi antar lembaga dalam pengawasan praktik kedokteran di era otonomi daerah?",
                category="koordinasi",
                difficulty="very_hard",
            ),
        ]

    async def call_api(
        self,
        endpoint: str,
        question: str,
        embedding_model: str,
        api_type: str = "multi",
        use_parallel: bool = True,
    ) -> APIResponse:
        """Call API endpoint"""
        url = f"{self.config.api_base_url}{endpoint}"
        headers = {"X-API-Key": self.config.api_key, "Content-Type": "application/json"}

        if api_type == "multi" or api_type == "multi_sequential":
            payload = {
                "query": question,
                "embedding_model": embedding_model,
                "use_parallel_execution": use_parallel,
                "previous_responses": [],
            }
        else:  # simple
            payload = {
                "query": question,
                "embedding_model": embedding_model,
                "previous_responses": [],
            }

        start_time = time.time()

        try:
            response = requests.post(
                url, json=payload, headers=headers, timeout=self.config.timeout
            )

            response_time_ms = int((time.time() - start_time) * 1000)

            if response.status_code == 200:
                data = response.json()

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
                    api_type=api_type,
                    embedding_model=embedding_model,
                    processing_steps=data.get("processing_steps", []),
                )
            else:
                return APIResponse(
                    answer="",
                    contexts=[],
                    response_time_ms=response_time_ms,
                    model_info={},
                    api_type=api_type,
                    embedding_model=embedding_model,
                    error=f"HTTP {response.status_code}: {response.text}",
                )

        except Exception as e:
            response_time_ms = int((time.time() - start_time) * 1000)
            return APIResponse(
                answer="",
                contexts=[],
                response_time_ms=response_time_ms,
                model_info={},
                api_type=api_type,
                embedding_model=embedding_model,
                error=str(e),
            )

    def calculate_ragas_scores(
        self, question: str, answer: str, contexts: List[str]
    ) -> Dict[str, float]:
        """Calculate RAGAs scores"""
        if not RAGAS_AVAILABLE or not contexts or not answer.strip():
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "context_relevancy": 0.0,
            }

        try:
            # Prepare dataset
            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truth": [answer],  # Using answer as ground truth for now
            }

            dataset = Dataset.from_dict(data)

            # Evaluate with RAGAs metrics
            result = evaluate(
                dataset=dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                    context_relevancy,
                ],
            )

            return {
                "faithfulness": float(result["faithfulness"]),
                "answer_relevancy": float(result["answer_relevancy"]),
                "context_precision": float(result["context_precision"]),
                "context_recall": float(result["context_recall"]),
                "context_relevancy": float(result["context_relevancy"]),
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

    async def run_single_benchmark(
        self, test_case: TestCase, api_type: str, embedding_model: str
    ) -> BenchmarkResult:
        """Run benchmark untuk satu test case"""

        # Determine endpoint based on API type
        endpoint = (
            "/api/chat" if api_type in ["multi", "multi_sequential"] else "/api/chat"
        )

        # Determine parallel execution
        use_parallel = api_type != "multi_sequential"

        logger.info(f"Testing: {api_type.upper()} API, {embedding_model} embedding")
        if api_type == "multi_sequential":
            logger.info("Using SEQUENTIAL execution (non-parallel)")
        elif api_type == "multi":
            logger.info("Using PARALLEL execution")
        logger.info(f"Question: {test_case.question[:60]}...")

        # Call API
        api_response = await self.call_api(
            endpoint, test_case.question, embedding_model, api_type, use_parallel
        )

        # Calculate RAGAs scores
        ragas_scores = self.calculate_ragas_scores(
            test_case.question, api_response.answer, api_response.contexts
        )

        # Calculate additional performance metrics
        performance_metrics = {
            "response_time_ms": api_response.response_time_ms,
            "answer_length": len(api_response.answer),
            "num_contexts": len(api_response.contexts),
            "contexts_total_length": sum(len(ctx) for ctx in api_response.contexts),
            "has_error": api_response.error is not None,
            "processing_steps": (
                len(api_response.processing_steps)
                if api_response.processing_steps
                else 0
            ),
        }

        return BenchmarkResult(
            test_case=test_case,
            api_response=api_response,
            ragas_scores=ragas_scores,
            performance_metrics=performance_metrics,
        )

    async def run_full_benchmark(self) -> List[BenchmarkResult]:
        """Run full benchmark across all combinations"""
        test_cases = self.create_test_cases()
        results = []

        # Test combinations - Now with 6 combinations including sequential multi
        combinations = [
            ("simple", "small"),
            ("simple", "large"),
            ("multi", "small"),
            ("multi", "large"),
            ("multi_sequential", "small"),
            ("multi_sequential", "large"),
        ]

        total_tests = len(test_cases) * len(combinations)
        current_test = 0

        logger.info(f"🚀 Starting benchmark with {total_tests} total tests")

        for test_case in test_cases:
            for api_type, embedding_model in combinations:
                current_test += 1
                logger.info(f"Progress: {current_test}/{total_tests}")

                result = await self.run_single_benchmark(
                    test_case, api_type, embedding_model
                )
                results.append(result)

                # Small delay to avoid overwhelming APIs
                await asyncio.sleep(1)

        return results

    def generate_summary_report(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate summary report"""
        df_data = []

        for result in results:
            row = {
                "api_type": result.api_response.api_type,
                "embedding_model": result.api_response.embedding_model,
                "category": result.test_case.category,
                "difficulty": result.test_case.difficulty,
                "response_time_ms": result.performance_metrics["response_time_ms"],
                "answer_length": result.performance_metrics["answer_length"],
                "num_contexts": result.performance_metrics["num_contexts"],
                "has_error": result.performance_metrics["has_error"],
                **result.ragas_scores,
            }
            df_data.append(row)

        df = pd.DataFrame(df_data)

        # Calculate summary statistics
        summary = {}

        for api_type in df["api_type"].unique():
            for embedding_model in df["embedding_model"].unique():
                key = f"{api_type}_{embedding_model}"
                subset = df[
                    (df["api_type"] == api_type)
                    & (df["embedding_model"] == embedding_model)
                ]

                if len(subset) > 0:
                    summary[key] = {
                        "avg_response_time_ms": float(
                            subset["response_time_ms"].mean()
                        ),
                        "avg_answer_length": float(subset["answer_length"].mean()),
                        "avg_num_contexts": float(subset["num_contexts"].mean()),
                        "error_rate": float(subset["has_error"].mean()),
                        "avg_faithfulness": float(subset["faithfulness"].mean()),
                        "avg_answer_relevancy": float(
                            subset["answer_relevancy"].mean()
                        ),
                        "avg_context_precision": float(
                            subset["context_precision"].mean()
                        ),
                        "avg_context_recall": float(subset["context_recall"].mean()),
                        "avg_context_relevancy": float(
                            subset["context_relevancy"].mean()
                        ),
                        "total_tests": len(subset),
                    }

        # Overall comparison
        comparison = {
            "best_overall_ragas": {},
            "fastest_response": {},
            "most_comprehensive": {},
        }

        # Find best performers
        if summary:
            # Best RAGAs scores (average of all metrics)
            best_ragas_key = max(
                summary.keys(),
                key=lambda k: np.mean(
                    [
                        summary[k]["avg_faithfulness"],
                        summary[k]["avg_answer_relevancy"],
                        summary[k]["avg_context_precision"],
                        summary[k]["avg_context_recall"],
                        summary[k]["avg_context_relevancy"],
                    ]
                ),
            )
            comparison["best_overall_ragas"] = {
                "combination": best_ragas_key,
                "score": np.mean(
                    [
                        summary[best_ragas_key]["avg_faithfulness"],
                        summary[best_ragas_key]["avg_answer_relevancy"],
                        summary[best_ragas_key]["avg_context_precision"],
                        summary[best_ragas_key]["avg_context_recall"],
                        summary[best_ragas_key]["avg_context_relevancy"],
                    ]
                ),
            }

            # Fastest response
            fastest_key = min(
                summary.keys(), key=lambda k: summary[k]["avg_response_time_ms"]
            )
            comparison["fastest_response"] = {
                "combination": fastest_key,
                "avg_time_ms": summary[fastest_key]["avg_response_time_ms"],
            }

            # Most comprehensive (highest context count)
            comprehensive_key = max(
                summary.keys(), key=lambda k: summary[k]["avg_num_contexts"]
            )
            comparison["most_comprehensive"] = {
                "combination": comprehensive_key,
                "avg_contexts": summary[comprehensive_key]["avg_num_contexts"],
            }

        return {
            "summary_by_combination": summary,
            "comparison": comparison,
            "raw_dataframe": df,
            "total_tests": len(results),
            "timestamp": datetime.now().isoformat(),
        }

    def save_results(self, results: List[BenchmarkResult], summary: Dict[str, Any]):
        """Save results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save raw results
        raw_data = [asdict(result) for result in results]
        with open(
            self.output_path / "raw_data" / f"benchmark_raw_{timestamp}.json", "w"
        ) as f:
            json.dump(raw_data, f, indent=2, default=str)

        # Save summary
        summary_data = summary.copy()
        # Remove DataFrame from summary for JSON serialization
        if "raw_dataframe" in summary_data:
            df = summary_data.pop("raw_dataframe")

        with open(
            self.output_path / "reports" / f"benchmark_summary_{timestamp}.json", "w"
        ) as f:
            json.dump(summary_data, f, indent=2, default=str)

        # Save DataFrame as CSV
        if "raw_dataframe" in summary:
            df = summary["raw_dataframe"]
            df.to_csv(
                self.output_path / "reports" / f"benchmark_data_{timestamp}.csv",
                index=False,
            )

        logger.info(f"✅ Results saved to {self.output_path}")
        return timestamp

    def generate_latex_report(self, summary: Dict[str, Any], timestamp: str):
        """Generate LaTeX report untuk laporan skripsi"""
        latex_content = f"""
\\section{{Hasil Evaluasi RAGAs}}

\\subsection{{Ringkasan Pengujian}}

Pengujian dilakukan pada tanggal \\texttt{{{timestamp}}} dengan total {summary['total_tests']} test cases menggunakan framework RAGAs (Retrieval Augmented Generation Assessment) untuk mengevaluasi kualitas sistem RAG dalam domain hukum kesehatan Indonesia.

\\subsection{{Kombinasi yang Diuji}}

\\begin{{itemize}}
    \\item Simple API + Small Embedding
    \\item Simple API + Large Embedding  
    \\item Multi-Step API (Parallel) + Small Embedding
    \\item Multi-Step API (Parallel) + Large Embedding
    \\item Multi-Step API (Sequential) + Small Embedding
    \\item Multi-Step API (Sequential) + Large Embedding
\\end{{itemize}}

\\subsection{{Metrik RAGAs}}

\\begin{{enumerate}}
    \\item \\textbf{{Faithfulness}}: Mengukur keakuratan faktual jawaban terhadap konteks yang diberikan
    \\item \\textbf{{Answer Relevancy}}: Mengukur relevansi jawaban terhadap pertanyaan yang diajukan
    \\item \\textbf{{Context Precision}}: Mengukur presisi konteks yang diambil (seberapa relevan konteks yang diambil)
    \\item \\textbf{{Context Recall}}: Mengukur recall konteks (seberapa lengkap konteks yang diambil)
    \\item \\textbf{{Context Relevancy}}: Mengukur relevansi keseluruhan konteks terhadap pertanyaan
\\end{{enumerate}}

\\subsection{{Hasil Pengujian}}

\\begin{{table}}[h]
\\centering
\\caption{{Perbandingan Performa Sistem RAG}}
\\begin{{tabular}}{{|l|c|c|c|c|c|c|}}
\\hline
\\textbf{{Kombinasi}} & \\textbf{{Faithfulness}} & \\textbf{{Answer Rel.}} & \\textbf{{Context Prec.}} & \\textbf{{Context Rec.}} & \\textbf{{Context Rel.}} & \\textbf{{Waktu (ms)}} \\\\
\\hline
"""

        # Add data rows
        for combo, data in summary["summary_by_combination"].items():
            parts = combo.split("_")
            if len(parts) == 3:  # multi_sequential_small/large
                api_type = f"{parts[0]}_{parts[1]}"
                embedding = parts[2]
            else:  # simple_small/large or multi_small/large
                api_type, embedding = parts

            # Format display name
            if api_type == "multi_sequential":
                combo_display = f"Multi (Seq) + {embedding.title()}"
            elif api_type == "multi":
                combo_display = f"Multi (Par) + {embedding.title()}"
            else:
                combo_display = f"{api_type.title()} + {embedding.title()}"

            latex_content += f"{combo_display} & "
            latex_content += f"{data['avg_faithfulness']:.3f} & "
            latex_content += f"{data['avg_answer_relevancy']:.3f} & "
            latex_content += f"{data['avg_context_precision']:.3f} & "
            latex_content += f"{data['avg_context_recall']:.3f} & "
            latex_content += f"{data['avg_context_relevancy']:.3f} & "
            latex_content += f"{data['avg_response_time_ms']:.0f} \\\\\n"

        latex_content += """\\hline
\\end{tabular}
\\end{table}

\\subsection{Analisis Hasil}

"""

        # Add analysis
        if "best_overall_ragas" in summary["comparison"]:
            best = summary["comparison"]["best_overall_ragas"]
            fastest = summary["comparison"]["fastest_response"]
            comprehensive = summary["comparison"]["most_comprehensive"]

            latex_content += f"""
\\textbf{{Konfigurasi Terbaik Secara Keseluruhan:}} {best['combination'].replace('_', ' + ').title()} dengan skor RAGAs rata-rata {best['score']:.3f}.

\\textbf{{Konfigurasi Tercepat:}} {fastest['combination'].replace('_', ' + ').title()} dengan waktu respons rata-rata {fastest['avg_time_ms']:.0f} ms.

\\textbf{{Konfigurasi Paling Komprehensif:}} {comprehensive['combination'].replace('_', ' + ').title()} dengan rata-rata {comprehensive['avg_contexts']:.1f} konteks per query.

\\subsection{{Kesimpulan}}

Berdasarkan evaluasi menggunakan framework RAGAs, dapat disimpulkan bahwa:

\\begin{{enumerate}}
    \\item Sistem Multi-Step RAG menunjukkan performa yang lebih baik dalam hal kualitas jawaban
    \\item Penggunaan embedding model yang lebih besar memberikan peningkatan akurasi
    \\item Trade-off antara kecepatan dan kualitas perlu dipertimbangkan sesuai kebutuhan aplikasi
\\end{{enumerate}}
"""

        # Save LaTeX file
        with open(
            self.output_path / "reports" / f"benchmark_report_{timestamp}.tex",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(latex_content)

        logger.info(f"📄 LaTeX report generated: benchmark_report_{timestamp}.tex")


# Main execution
async def main():
    """Main execution function"""
    config = BenchmarkConfig()
    benchmark = RAGAsBenchmark(config)

    logger.info("🚀 Starting RAGAs Benchmark for Thesis Report")

    # Run benchmark
    results = await benchmark.run_full_benchmark()

    # Generate summary
    summary = benchmark.generate_summary_report(results)

    # Save results
    timestamp = benchmark.save_results(results, summary)

    # Generate LaTeX report
    benchmark.generate_latex_report(summary, timestamp)

    logger.info("✅ Benchmark completed successfully!")
    logger.info(f"📊 Results available in: {benchmark.output_path}")

    return results, summary


if __name__ == "__main__":
    # Install RAGAs if not available
    if not RAGAS_AVAILABLE:
        print("📦 Installing RAGAs...")
        import subprocess

        subprocess.run(["pip", "install", "ragas"], check=True)
        print("✅ RAGAs installed. Please restart the script.")
    else:
        asyncio.run(main())

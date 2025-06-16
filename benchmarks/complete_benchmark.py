#!/usr/bin/env python3
"""
🔬 Complete RAG Benchmark System
Benchmark komprehensif untuk membandingkan:
1. Simple API (Port 8081) vs Multi API (Port 8080)
2. Text Embedding Small vs Large
3. Performance Metrics untuk Laporan Skripsi

Metrics yang diukur:
- Response Time (ms)
- Answer Quality (length, completeness)
- Document Retrieval (number of docs, relevance)
- Processing Steps (for Multi API)
- Success Rate
- Error Analysis
"""

import asyncio
import json
import time
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path
import logging
import statistics
from concurrent.futures import ThreadPoolExecutor
import threading

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CompleteBenchmark:
    def __init__(self):
        self.simple_api_base = "http://localhost:8081"
        self.multi_api_base = "http://localhost:8080"
        self.api_key = "your_secure_api_key_here"
        self.headers = {"X-API-Key": self.api_key, "Content-Type": "application/json"}
        self.setup_output_dir()
        self.results = []

    def setup_output_dir(self):
        """Setup output directory"""
        self.output_dir = Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "detailed").mkdir(exist_ok=True)

    def get_comprehensive_test_questions(self) -> List[Dict]:
        """Comprehensive test questions untuk evaluasi hukum kesehatan"""
        return [
            # Easy Level - Basic Definitions
            {
                "question": "Apa definisi tenaga kesehatan menurut undang-undang?",
                "category": "definisi",
                "difficulty": "easy",
                "expected_keywords": ["tenaga kesehatan", "setiap orang", "pendidikan"],
            },
            {
                "question": "Siapa yang berwenang mengeluarkan STR dokter?",
                "category": "perizinan",
                "difficulty": "easy",
                "expected_keywords": [
                    "konsil kedokteran",
                    "STR",
                    "surat tanda registrasi",
                ],
            },
            {
                "question": "Apa kepanjangan dari STR dan SIP?",
                "category": "akronim",
                "difficulty": "easy",
                "expected_keywords": ["surat tanda registrasi", "surat izin praktik"],
            },
            # Medium Level - Procedures and Requirements
            {
                "question": "Apa sanksi hukum bagi praktik kedokteran tanpa izin?",
                "category": "sanksi",
                "difficulty": "medium",
                "expected_keywords": ["sanksi", "pidana", "denda", "penjara"],
            },
            {
                "question": "Bagaimana prosedur pengajuan izin praktik dokter spesialis?",
                "category": "prosedur",
                "difficulty": "medium",
                "expected_keywords": ["persyaratan", "dokumen", "pengajuan"],
            },
            {
                "question": "Apa kewajiban dokter dalam memberikan informed consent?",
                "category": "etika",
                "difficulty": "medium",
                "expected_keywords": ["informed consent", "persetujuan", "informasi"],
            },
            {
                "question": "Bagaimana ketentuan rekam medis menurut peraturan yang berlaku?",
                "category": "administrasi",
                "difficulty": "medium",
                "expected_keywords": ["rekam medis", "penyimpanan", "kerahasiaan"],
            },
            # Hard Level - Complex Analysis
            {
                "question": "Bagaimana hubungan antara UU Praktik Kedokteran dengan UU Rumah Sakit dalam hal tanggung jawab medis?",
                "category": "analisis",
                "difficulty": "hard",
                "expected_keywords": [
                    "tanggung jawab",
                    "rumah sakit",
                    "praktik kedokteran",
                ],
            },
            {
                "question": "Apa perbedaan sanksi pidana dan perdata dalam kasus malpraktik medis?",
                "category": "hukum",
                "difficulty": "hard",
                "expected_keywords": ["pidana", "perdata", "malpraktik", "perbedaan"],
            },
            {
                "question": "Bagaimana implementasi telemedicine dalam konteks peraturan praktik kedokteran di Indonesia?",
                "category": "teknologi",
                "difficulty": "hard",
                "expected_keywords": ["telemedicine", "teknologi", "peraturan"],
            },
            # Very Hard Level - Comprehensive Analysis
            {
                "question": "Analisis komprehensif tentang perlindungan hukum bagi pasien dan tenaga kesehatan dalam sistem kesehatan nasional",
                "category": "komprehensif",
                "difficulty": "very_hard",
                "expected_keywords": [
                    "perlindungan hukum",
                    "pasien",
                    "tenaga kesehatan",
                    "sistem kesehatan",
                ],
            },
            {
                "question": "Bagaimana koordinasi antar lembaga dalam pengawasan praktik kedokteran di era otonomi daerah?",
                "category": "koordinasi",
                "difficulty": "very_hard",
                "expected_keywords": ["koordinasi", "pengawasan", "otonomi daerah"],
            },
        ]

    def call_simple_api(self, question: str, embedding_model: str) -> Dict:
        """Call Simple API"""
        url = f"{self.simple_api_base}/api/chat"
        payload = {
            "query": question,
            "embedding_model": embedding_model,
            "previous_responses": [],
        }

        return self._make_api_call(url, payload, "simple", embedding_model)

    def call_multi_api(self, question: str, embedding_model: str) -> Dict:
        """Call Multi-Step API"""
        url = f"{self.multi_api_base}/api/chat"
        payload = {
            "query": question,
            "embedding_model": embedding_model,
            "use_parallel_execution": True,
            "previous_responses": [],
        }

        return self._make_api_call(url, payload, "multi", embedding_model)

    def _make_api_call(
        self, url: str, payload: Dict, api_type: str, embedding_model: str
    ) -> Dict:
        """Generic API call method"""
        start_time = time.time()

        try:
            response = requests.post(
                url, json=payload, headers=self.headers, timeout=120
            )
            response_time = int((time.time() - start_time) * 1000)

            if response.status_code == 200:
                data = response.json()

                # Extract detailed metrics
                result = {
                    "success": True,
                    "answer": data.get("answer", ""),
                    "response_time_ms": response_time,
                    "api_type": api_type,
                    "embedding_model": embedding_model,
                    "answer_length": len(data.get("answer", "")),
                    "answer_word_count": len(data.get("answer", "").split()),
                }

                # API-specific metrics
                if api_type == "simple":
                    result.update(
                        {
                            "num_documents": len(data.get("referenced_documents", [])),
                            "processing_steps": 1,  # Simple API has 1 step
                            "model_info": data.get("model_info", {}),
                        }
                    )
                else:  # multi
                    result.update(
                        {
                            "num_documents": len(data.get("referenced_documents", [])),
                            "processing_steps": len(data.get("processing_steps", [])),
                            "model_info": data.get("model_info", {}),
                        }
                    )

                return result

            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text[:200]}",
                    "response_time_ms": response_time,
                    "api_type": api_type,
                    "embedding_model": embedding_model,
                }

        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Request timeout (120s)",
                "response_time_ms": 120000,
                "api_type": api_type,
                "embedding_model": embedding_model,
            }
        except Exception as e:
            response_time = int((time.time() - start_time) * 1000)
            return {
                "success": False,
                "error": str(e),
                "response_time_ms": response_time,
                "api_type": api_type,
                "embedding_model": embedding_model,
            }

    def calculate_answer_quality_score(
        self, answer: str, expected_keywords: List[str]
    ) -> Dict:
        """Calculate answer quality based on keywords and structure"""
        if not answer or not answer.strip():
            return {
                "keyword_score": 0.0,
                "structure_score": 0.0,
                "completeness_score": 0.0,
            }

        answer_lower = answer.lower()

        # Keyword matching score
        keyword_matches = sum(
            1 for keyword in expected_keywords if keyword.lower() in answer_lower
        )
        keyword_score = (
            keyword_matches / len(expected_keywords) if expected_keywords else 0.0
        )

        # Structure score (presence of paragraphs, lists, etc.)
        structure_indicators = [
            "\n\n" in answer,  # Paragraphs
            "1." in answer or "2." in answer,  # Numbered lists
            "•" in answer or "-" in answer,  # Bullet points
            "**" in answer,  # Bold formatting
            len(answer.split(".")) > 3,  # Multiple sentences
        ]
        structure_score = sum(structure_indicators) / len(structure_indicators)

        # Completeness score (based on length and content)
        length_score = min(len(answer) / 500, 1.0)  # Normalize to 500 chars
        completeness_score = (length_score + keyword_score) / 2

        return {
            "keyword_score": keyword_score,
            "structure_score": structure_score,
            "completeness_score": completeness_score,
            "overall_quality": (keyword_score + structure_score + completeness_score)
            / 3,
        }

    def run_single_test(
        self, question_data: Dict, api_type: str, embedding_model: str
    ) -> Dict:
        """Run single test case"""
        logger.info(f"Testing: {api_type.upper()} + {embedding_model.upper()}")
        logger.info(
            f"Question ({question_data['difficulty']}): {question_data['question'][:60]}..."
        )

        # Call appropriate API
        if api_type == "simple":
            result = self.call_simple_api(question_data["question"], embedding_model)
        else:
            result = self.call_multi_api(question_data["question"], embedding_model)

        # Add question metadata
        result.update(
            {
                "question": question_data["question"],
                "category": question_data["category"],
                "difficulty": question_data["difficulty"],
                "expected_keywords": question_data.get("expected_keywords", []),
            }
        )

        # Calculate quality scores if successful
        if result.get("success", False):
            quality_scores = self.calculate_answer_quality_score(
                result.get("answer", ""), question_data.get("expected_keywords", [])
            )
            result.update(quality_scores)
        else:
            result.update(
                {
                    "keyword_score": 0.0,
                    "structure_score": 0.0,
                    "completeness_score": 0.0,
                    "overall_quality": 0.0,
                }
            )

        return result

    def run_benchmark(self) -> List[Dict]:
        """Run complete benchmark"""
        questions = self.get_comprehensive_test_questions()
        results = []

        combinations = [
            ("simple", "small"),
            ("simple", "large"),
            ("multi", "small"),
            ("multi", "large"),
        ]

        total_tests = len(questions) * len(combinations)
        current_test = 0

        logger.info(f"🚀 Starting comprehensive benchmark with {total_tests} tests")
        logger.info("=" * 80)

        for question_data in questions:
            for api_type, embedding_model in combinations:
                current_test += 1
                logger.info(f"Progress: {current_test}/{total_tests}")

                result = self.run_single_test(question_data, api_type, embedding_model)
                results.append(result)

                # Log result summary
                if result.get("success", False):
                    logger.info(
                        f"✅ Success: {result['response_time_ms']}ms, Quality: {result['overall_quality']:.2f}"
                    )
                else:
                    logger.info(f"❌ Failed: {result.get('error', 'Unknown error')}")

                logger.info("-" * 40)

                # Delay between requests
                time.sleep(3)

        return results

    def analyze_results(self, results: List[Dict]) -> Dict:
        """Comprehensive analysis of benchmark results"""
        df = pd.DataFrame(results)

        # Overall statistics
        total_tests = len(results)
        successful_tests = len(df[df["success"] == True])
        success_rate = successful_tests / total_tests * 100

        # Analysis by combination
        summary = {}
        for api_type in ["simple", "multi"]:
            for embedding_model in ["small", "large"]:
                key = f"{api_type}_{embedding_model}"
                subset = df[
                    (df["api_type"] == api_type)
                    & (df["embedding_model"] == embedding_model)
                ]

                if len(subset) > 0:
                    successful = subset[subset["success"] == True]

                    if len(successful) > 0:
                        summary[key] = {
                            # Basic stats
                            "total_tests": len(subset),
                            "successful_tests": len(successful),
                            "success_rate": len(successful) / len(subset) * 100,
                            # Performance metrics
                            "avg_response_time_ms": successful[
                                "response_time_ms"
                            ].mean(),
                            "median_response_time_ms": successful[
                                "response_time_ms"
                            ].median(),
                            "min_response_time_ms": successful[
                                "response_time_ms"
                            ].min(),
                            "max_response_time_ms": successful[
                                "response_time_ms"
                            ].max(),
                            # Answer quality metrics
                            "avg_answer_length": successful["answer_length"].mean(),
                            "avg_word_count": successful["answer_word_count"].mean(),
                            "avg_num_documents": successful["num_documents"].mean(),
                            "avg_processing_steps": successful[
                                "processing_steps"
                            ].mean(),
                            # Quality scores
                            "avg_keyword_score": successful["keyword_score"].mean(),
                            "avg_structure_score": successful["structure_score"].mean(),
                            "avg_completeness_score": successful[
                                "completeness_score"
                            ].mean(),
                            "avg_overall_quality": successful["overall_quality"].mean(),
                            # Performance by difficulty
                            "performance_by_difficulty": {},
                        }

                        # Breakdown by difficulty
                        for difficulty in ["easy", "medium", "hard", "very_hard"]:
                            diff_subset = successful[
                                successful["difficulty"] == difficulty
                            ]
                            if len(diff_subset) > 0:
                                summary[key]["performance_by_difficulty"][
                                    difficulty
                                ] = {
                                    "count": len(diff_subset),
                                    "avg_response_time_ms": diff_subset[
                                        "response_time_ms"
                                    ].mean(),
                                    "avg_quality": diff_subset[
                                        "overall_quality"
                                    ].mean(),
                                    "success_rate": len(diff_subset)
                                    / len(subset[subset["difficulty"] == difficulty])
                                    * 100,
                                }
                    else:
                        summary[key] = {
                            "total_tests": len(subset),
                            "successful_tests": 0,
                            "success_rate": 0,
                            "note": "No successful tests",
                        }

        # Comparative analysis
        comparison = {
            "best_overall_performance": None,
            "fastest_response": None,
            "highest_quality": None,
            "most_comprehensive": None,
        }

        if summary:
            # Best overall (combination of success rate and quality)
            best_overall = max(
                summary.items(),
                key=lambda x: x[1].get("success_rate", 0)
                * x[1].get("avg_overall_quality", 0),
            )
            comparison["best_overall_performance"] = {
                "combination": best_overall[0],
                "score": best_overall[1].get("success_rate", 0)
                * best_overall[1].get("avg_overall_quality", 0),
            }

            # Fastest response
            fastest = min(
                summary.items(),
                key=lambda x: x[1].get("avg_response_time_ms", float("inf")),
            )
            comparison["fastest_response"] = {
                "combination": fastest[0],
                "avg_time_ms": fastest[1].get("avg_response_time_ms", 0),
            }

            # Highest quality
            highest_quality = max(
                summary.items(), key=lambda x: x[1].get("avg_overall_quality", 0)
            )
            comparison["highest_quality"] = {
                "combination": highest_quality[0],
                "avg_quality": highest_quality[1].get("avg_overall_quality", 0),
            }

            # Most comprehensive
            most_comprehensive = max(
                summary.items(), key=lambda x: x[1].get("avg_num_documents", 0)
            )
            comparison["most_comprehensive"] = {
                "combination": most_comprehensive[0],
                "avg_documents": most_comprehensive[1].get("avg_num_documents", 0),
            }

        return {
            "overall_stats": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": success_rate,
                "timestamp": datetime.now().isoformat(),
            },
            "summary_by_combination": summary,
            "comparative_analysis": comparison,
            "raw_data": results,
        }

    def save_results(self, analysis: Dict) -> str:
        """Save comprehensive results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save complete analysis
        with open(self.output_dir / f"complete_benchmark_{timestamp}.json", "w") as f:
            json.dump(analysis, f, indent=2, default=str)

        # Save CSV for data analysis
        df = pd.DataFrame(analysis["raw_data"])
        df.to_csv(self.output_dir / f"benchmark_data_{timestamp}.csv", index=False)

        # Save summary statistics
        summary_stats = {
            "overall": analysis["overall_stats"],
            "by_combination": analysis["summary_by_combination"],
            "comparison": analysis["comparative_analysis"],
        }

        with open(self.output_dir / f"benchmark_summary_{timestamp}.json", "w") as f:
            json.dump(summary_stats, f, indent=2, default=str)

        logger.info(f"✅ Results saved with timestamp: {timestamp}")
        return timestamp

    def generate_comprehensive_report(self, analysis: Dict, timestamp: str):
        """Generate comprehensive markdown report"""
        report = f"""# Comprehensive RAG Benchmark Report

**Generated:** {analysis['overall_stats']['timestamp']}  
**Timestamp:** {timestamp}

## Executive Summary

- **Total Tests:** {analysis['overall_stats']['total_tests']}
- **Successful Tests:** {analysis['overall_stats']['successful_tests']}
- **Overall Success Rate:** {analysis['overall_stats']['success_rate']:.1f}%

## Test Configuration

### APIs Tested
1. **Simple API** (Port 8081) - Basic RAG pipeline
2. **Multi-Step API** (Port 8080) - Enhanced Multi-Step RAG with agent-based approach

### Embedding Models
1. **Small Embedding** - Faster, less comprehensive
2. **Large Embedding** - Slower, more comprehensive

### Test Categories
- **Definisi** - Basic definitions (Easy)
- **Perizinan** - Licensing procedures (Easy-Medium)
- **Sanksi** - Legal sanctions (Medium)
- **Prosedur** - Administrative procedures (Medium)
- **Etika** - Medical ethics (Medium)
- **Analisis** - Complex analysis (Hard)
- **Komprehensif** - Comprehensive analysis (Very Hard)

## Performance Results

"""

        # Add detailed results for each combination
        for combo, data in analysis["summary_by_combination"].items():
            api_type, embedding = combo.split("_")
            report += f"""
### {api_type.upper()} API + {embedding.upper()} Embedding

**Overall Performance:**
- Success Rate: {data.get('success_rate', 0):.1f}%
- Average Response Time: {data.get('avg_response_time_ms', 0):.0f} ms
- Median Response Time: {data.get('median_response_time_ms', 0):.0f} ms
- Response Time Range: {data.get('min_response_time_ms', 0):.0f} - {data.get('max_response_time_ms', 0):.0f} ms

**Answer Quality:**
- Average Answer Length: {data.get('avg_answer_length', 0):.0f} characters
- Average Word Count: {data.get('avg_word_count', 0):.1f} words
- Keyword Matching Score: {data.get('avg_keyword_score', 0):.3f}
- Structure Score: {data.get('avg_structure_score', 0):.3f}
- Completeness Score: {data.get('avg_completeness_score', 0):.3f}
- **Overall Quality Score: {data.get('avg_overall_quality', 0):.3f}**

**Technical Metrics:**
- Average Documents Retrieved: {data.get('avg_num_documents', 0):.1f}
- Average Processing Steps: {data.get('avg_processing_steps', 0):.1f}

"""

            # Performance by difficulty
            if "performance_by_difficulty" in data:
                report += "**Performance by Difficulty:**\n\n"
                for difficulty, perf in data["performance_by_difficulty"].items():
                    report += f"- **{difficulty.title()}**: {perf['success_rate']:.1f}% success, {perf['avg_response_time_ms']:.0f}ms avg, {perf['avg_quality']:.3f} quality\n"
                report += "\n"

        # Comparative analysis
        comparison = analysis["comparative_analysis"]
        report += f"""
## Comparative Analysis

### Key Findings

**🏆 Best Overall Performance:** {comparison['best_overall_performance']['combination'].replace('_', ' + ').title()}
- Combined score (success rate × quality): {comparison['best_overall_performance']['score']:.2f}

**⚡ Fastest Response:** {comparison['fastest_response']['combination'].replace('_', ' + ').title()}
- Average response time: {comparison['fastest_response']['avg_time_ms']:.0f} ms

**🎯 Highest Quality:** {comparison['highest_quality']['combination'].replace('_', ' + ').title()}
- Average quality score: {comparison['highest_quality']['avg_quality']:.3f}

**📚 Most Comprehensive:** {comparison['most_comprehensive']['combination'].replace('_', ' + ').title()}
- Average documents retrieved: {comparison['most_comprehensive']['avg_documents']:.1f}

## Recommendations

### For Production Use:
1. **High-Performance Requirements:** Use {comparison['fastest_response']['combination'].replace('_', ' + ').title()}
2. **High-Quality Requirements:** Use {comparison['highest_quality']['combination'].replace('_', ' + ').title()}  
3. **Balanced Requirements:** Use {comparison['best_overall_performance']['combination'].replace('_', ' + ').title()}

### Trade-offs:
- **Simple API** offers faster response times but potentially less comprehensive answers
- **Multi API** provides more thorough analysis but takes longer to process
- **Large embeddings** generally provide better quality but slower response times
- **Small embeddings** offer faster processing suitable for real-time applications

## Technical Notes

- All tests conducted with 3-second delays between requests
- Timeout set to 120 seconds per request
- Quality scoring based on keyword matching, answer structure, and completeness
- Results saved with timestamp {timestamp} for reproducibility

---
*Generated by RAG Benchmark System for Thesis Research*
"""

        # Save report
        with open(
            self.output_dir / f"comprehensive_report_{timestamp}.md",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(report)

        logger.info(
            f"📄 Comprehensive report generated: comprehensive_report_{timestamp}.md"
        )

    def generate_latex_table(self, analysis: Dict, timestamp: str):
        """Generate LaTeX table for thesis"""
        latex_content = """
\\begin{table}[h]
\\centering
\\caption{Perbandingan Performa Sistem RAG}
\\label{tab:rag_comparison}
\\begin{tabular}{|l|c|c|c|c|c|c|}
\\hline
\\textbf{Kombinasi} & \\textbf{Success} & \\textbf{Waktu} & \\textbf{Kualitas} & \\textbf{Dokumen} & \\textbf{Kata} & \\textbf{Langkah} \\\\
\\textbf{} & \\textbf{Rate (\%)} & \\textbf{(ms)} & \\textbf{Score} & \\textbf{Rata-rata} & \\textbf{Rata-rata} & \\textbf{Proses} \\\\
\\hline
"""

        for combo, data in analysis["summary_by_combination"].items():
            api_type, embedding = combo.split("_")
            combo_display = f"{api_type.title()} + {embedding.title()}"

            latex_content += f"{combo_display} & "
            latex_content += f"{data.get('success_rate', 0):.1f} & "
            latex_content += f"{data.get('avg_response_time_ms', 0):.0f} & "
            latex_content += f"{data.get('avg_overall_quality', 0):.3f} & "
            latex_content += f"{data.get('avg_num_documents', 0):.1f} & "
            latex_content += f"{data.get('avg_word_count', 0):.0f} & "
            latex_content += f"{data.get('avg_processing_steps', 0):.1f} \\\\\n"

        latex_content += """\\hline
\\end{tabular}
\\end{table}
"""

        # Save LaTeX table
        with open(
            self.output_dir / f"latex_table_{timestamp}.tex", "w", encoding="utf-8"
        ) as f:
            f.write(latex_content)

        logger.info(f"📄 LaTeX table generated: latex_table_{timestamp}.tex")


def main():
    """Main execution function"""
    benchmark = CompleteBenchmark()

    logger.info("🚀 Starting Complete RAG Benchmark for Thesis")
    logger.info(
        "📊 This benchmark will compare Simple API vs Multi API with Small/Large embeddings"
    )
    logger.info("⏱️  Estimated time: 15-20 minutes for complete benchmark")

    # Check API availability
    logger.info("🔍 Checking API availability...")

    try:
        # Test Multi API
        response = requests.get(f"{benchmark.multi_api_base}/health", timeout=10)
        if response.status_code == 200:
            logger.info("✅ Multi API (Port 8080) is available")
        else:
            logger.warning("⚠️ Multi API health check failed")
    except:
        logger.error("❌ Multi API (Port 8080) is not accessible")
        logger.info("💡 Please start Multi API with: docker-compose up")
        return

    try:
        # Test Simple API
        response = requests.get(f"{benchmark.simple_api_base}/health", timeout=10)
        if response.status_code == 200:
            logger.info("✅ Simple API (Port 8081) is available")
        else:
            logger.warning("⚠️ Simple API health check failed")
    except:
        logger.error("❌ Simple API (Port 8081) is not accessible")
        logger.info(
            "💡 Please start Simple API with: docker-compose -f docker-compose-simple.yml up"
        )
        return

    # Run benchmark
    results = benchmark.run_benchmark()

    # Analyze results
    logger.info("📊 Analyzing benchmark results...")
    analysis = benchmark.analyze_results(results)

    # Save results
    timestamp = benchmark.save_results(analysis)

    # Generate reports
    benchmark.generate_comprehensive_report(analysis, timestamp)
    benchmark.generate_latex_table(analysis, timestamp)

    # Print summary
    logger.info("=" * 80)
    logger.info("✅ BENCHMARK COMPLETED SUCCESSFULLY!")
    logger.info(f"📁 Results saved in: {benchmark.output_dir}")
    logger.info(f"🔖 Timestamp: {timestamp}")
    logger.info("=" * 80)

    print("\n📊 QUICK SUMMARY:")
    for combo, data in analysis["summary_by_combination"].items():
        print(
            f"{combo.replace('_', ' + ').title()}: {data.get('success_rate', 0):.1f}% success, {data.get('avg_response_time_ms', 0):.0f}ms avg, {data.get('avg_overall_quality', 0):.3f} quality"
        )

    print(
        f"\n🏆 Best Overall: {analysis['comparative_analysis']['best_overall_performance']['combination'].replace('_', ' + ').title()}"
    )
    print(
        f"⚡ Fastest: {analysis['comparative_analysis']['fastest_response']['combination'].replace('_', ' + ').title()}"
    )
    print(
        f"🎯 Highest Quality: {analysis['comparative_analysis']['highest_quality']['combination'].replace('_', ' + ').title()}"
    )


if __name__ == "__main__":
    main()

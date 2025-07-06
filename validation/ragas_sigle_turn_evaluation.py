"""
🎯 RAG COMPARISON EVALUATION & VISUALIZATION
Script untuk evaluasi dan perbandingan Single RAG vs Multi-Agent RAG dengan visualisasi lengkap

Membandingkan performa kedua sistem menggunakan RAGAS metrics:
- Single RAG: https://lexmedica-chatbot-176465812210.asia-southeast2.run.app/api/chat
- Multi-Agent RAG: https://lexmedica-chatbot-multiagent-176465812210.asia-southeast2.run.app/api/chat

Usage:
    python single_rag_evaluation.py            # Full comparison evaluation
    python single_rag_evaluation.py --test-parsing  # Test response parsing only

Requirements:
    - API_KEY environment variable (untuk RAG systems)
    - OPENAI_API_KEY environment variable (untuk RAGAS evaluation)
"""

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
import requests
import time
from typing import Dict, List, Any
from datasets import Dataset

# Import RAGAS
try:
    from ragas import evaluate
    from ragas.metrics import (
        context_recall,
        context_precision,  # Added: Context Precision
        context_entity_recall,  # Added: Context Entity Recall
        NoiseSensitivity,  # Added: Noise Sensitivity (class)
        faithfulness,
        answer_relevancy,  # Same as response_relevancy
        FactualCorrectness,
    )
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI

    print("✅ RAGAS libraries imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("📦 Install with: pip install ragas langchain-openai datasets")
    exit(1)

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class RAGComparisonEvaluator:
    """Evaluator untuk membandingkan Single RAG vs Multi-Agent RAG"""

    def __init__(self, api_key: str = None, openai_api_key: str = None):
        """Initialize evaluator"""
        self.api_key = api_key or os.getenv("API_KEY", "your_secure_api_key_here")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        # Setup RAGAS evaluator LLM
        if self.openai_api_key:
            llm = ChatOpenAI(model="gpt-4.1", api_key=self.openai_api_key)
            self.evaluator_llm = LangchainLLMWrapper(llm)
        else:
            print("❌ OpenAI API key tidak ditemukan")
            self.evaluator_llm = None

        # RAG endpoints
        self.single_rag_url = (
            "https://lexmedica-chatbot-176465812210.asia-southeast2.run.app/api/chat"
        )
        self.multi_agent_url = "https://lexmedica-chatbot-multiagent-176465812210.asia-southeast2.run.app/api/chat"

        self.timeout = 180
        self.results = None

        print(f"🌐 Single RAG Endpoint: {self.single_rag_url}")
        print(f"🤖 Multi-Agent Endpoint: {self.multi_agent_url}")

    def test_endpoints(self):
        """Test koneksi ke kedua RAG endpoints"""
        print("🔍 Testing RAG endpoints...")

        test_payload = {"query": "Test connection", "embedding_model": "large"}
        headers = {"Content-Type": "application/json", "X-API-Key": self.api_key}

        single_status = False
        multi_status = False

        # Test Single RAG
        try:
            print("   🌐 Testing Single RAG...")
            response = requests.post(
                self.single_rag_url, json=test_payload, headers=headers, timeout=180
            )
            if response.status_code == 200:
                print("   ✅ Single RAG: Online")

                # Debug: Check response structure
                try:
                    data = response.json()
                    print(f"      🔍 Response keys: {list(data.keys())}")
                    if "referenced_documents" in data:
                        print(
                            f"      📋 Found referenced_documents: {len(data['referenced_documents'])} docs"
                        )
                    else:
                        print("      ⚠️ No referenced_documents in response")
                except:
                    print("      ⚠️ Could not parse JSON response")

                single_status = True
            else:
                print(f"   ⚠️ Single RAG: Status {response.status_code}")
                try:
                    print(f"      Response: {response.text[:200]}")
                except:
                    pass
        except Exception as e:
            print(f"   ❌ Single RAG: Error - {e}")

        # Test Multi-Agent RAG
        try:
            print("   🤖 Testing Multi-Agent RAG...")
            response = requests.post(
                self.multi_agent_url, json=test_payload, headers=headers, timeout=180
            )
            if response.status_code == 200:
                print("   ✅ Multi-Agent RAG: Online")

                # Debug: Check response structure
                try:
                    data = response.json()
                    print(f"      🔍 Response keys: {list(data.keys())}")
                    if "referenced_documents" in data:
                        print(
                            f"      📋 Found referenced_documents: {len(data['referenced_documents'])} docs"
                        )
                    else:
                        print("      ⚠️ No referenced_documents in response")
                except:
                    print("      ⚠️ Could not parse JSON response")

                multi_status = True
            else:
                print(f"   ⚠️ Multi-Agent RAG: Status {response.status_code}")
                try:
                    print(f"      Response: {response.text[:200]}")
                except:
                    pass
        except Exception as e:
            print(f"   ❌ Multi-Agent RAG: Error - {e}")

        return single_status, multi_status

    def query_single_rag(self, question: str) -> Dict[str, Any]:
        """Query Single RAG endpoint"""
        headers = {"Content-Type": "application/json", "X-API-Key": self.api_key}
        payload = {"query": question, "embedding_model": "large"}

        try:
            print(f"   ⏳ Querying Single RAG...")
            start_time = time.time()

            response = requests.post(
                self.single_rag_url, json=payload, headers=headers, timeout=self.timeout
            )
            response_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()

                # Extract contexts from referenced_documents
                contexts = []
                if "referenced_documents" in data:
                    print(
                        f"   📋 Found {len(data['referenced_documents'])} referenced documents"
                    )
                    for i, doc in enumerate(data["referenced_documents"], 1):
                        if isinstance(doc, dict) and "content" in doc:
                            contexts.append(doc["content"])
                            print(
                                f"   📄 Doc {i}: {doc.get('source', 'Unknown')[:50]}..."
                            )
                        elif isinstance(doc, str):
                            contexts.append(doc)
                            print(f"   📄 Doc {i}: String format")
                else:
                    print("   ⚠️ No 'referenced_documents' found in response")
                    print(f"   🔍 Available keys: {list(data.keys())}")

                print(
                    f"   ✅ Success in {response_time:.1f}s ({len(contexts)} contexts extracted)"
                )

                return {
                    "status": "success",
                    "answer": data.get("answer", ""),
                    "contexts": contexts,
                    "response_time": response_time,
                }
            else:
                print(f"   ❌ HTTP Error {response.status_code}")
                return {"status": "error", "error": f"HTTP {response.status_code}"}

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return {"status": "error", "error": str(e)}

    def query_multi_agent(self, question: str) -> Dict[str, Any]:
        """Query Multi-Agent RAG endpoint"""
        headers = {"Content-Type": "application/json", "X-API-Key": self.api_key}
        payload = {
            "query": question,
            "embedding_model": "large",
            "use_parallel_execution": False,
            "previous_responses": [],
        }

        try:
            print(f"   ⏳ Querying Multi-Agent RAG...")
            start_time = time.time()

            response = requests.post(
                self.multi_agent_url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()

                # Extract contexts from referenced_documents
                contexts = []
                if "referenced_documents" in data:
                    print(
                        f"   📋 Found {len(data['referenced_documents'])} referenced documents"
                    )
                    for i, doc in enumerate(data["referenced_documents"], 1):
                        if isinstance(doc, dict) and "content" in doc:
                            contexts.append(doc["content"])
                            print(
                                f"   📄 Doc {i}: {doc.get('source', 'Unknown')[:50]}..."
                            )
                        elif isinstance(doc, str):
                            contexts.append(doc)
                            print(f"   📄 Doc {i}: String format")
                else:
                    print("   ⚠️ No 'referenced_documents' found in response")
                    print(f"   🔍 Available keys: {list(data.keys())}")

                print(
                    f"   ✅ Success in {response_time:.1f}s ({len(contexts)} contexts extracted)"
                )

                return {
                    "status": "success",
                    "answer": data.get("answer", ""),
                    "contexts": contexts,
                    "response_time": response_time,
                }
            else:
                print(f"   ❌ HTTP Error {response.status_code}")
                return {"status": "error", "error": f"HTTP {response.status_code}"}

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return {"status": "error", "error": str(e)}

    def test_parsing(
        self, question: str = "Apa definisi kesehatan menurut undang-undang?"
    ):
        """Test parsing response body secara detail"""
        print("🧪 TESTING RESPONSE PARSING")
        print("=" * 50)
        print(f"📝 Test Question: {question}")

        result = self.query_single_rag(question)

        if result["status"] == "success":
            print(f"\n📊 PARSING RESULTS:")
            print(f"   Answer Length: {len(result['answer'])} chars")
            print(f"   Contexts Count: {len(result['contexts'])}")
            print(f"   Response Time: {result['response_time']:.2f}s")

            print(f"\n📄 EXTRACTED CONTEXTS:")
            for i, context in enumerate(result["contexts"], 1):
                print(f"   Context {i}: {context[:100]}...")

            print(f"\n💬 GENERATED ANSWER:")
            print(f"   {result['answer'][:200]}...")

        else:
            print(f"❌ Test failed: {result}")

        return result

    def evaluate_responses(self, dataset_dict: List[Dict]) -> Dict[str, float]:
        """Evaluate dengan RAGAS"""
        if not self.evaluator_llm:
            print("❌ No evaluator LLM available")
            return {"error": "No evaluator LLM"}

        try:
            # Convert to RAGAS dataset format
            dataset = Dataset.from_list(dataset_dict)

            # Define metrics sesuai dokumentasi RAGAS
            metrics = [
                context_recall,  # Context Recall
                context_precision,  # Context Precision
                context_entity_recall,  # Context Entity Recall
                NoiseSensitivity(),  # Noise Sensitivity (instance)
                faithfulness,  # Faithfulness
                answer_relevancy,  # Answer/Response Relevancy
                FactualCorrectness(mode="f1"),  # Factual Correctness
            ]

            print(f"   📊 Evaluating {len(dataset_dict)} responses with RAGAS...")

            # Run RAGAS evaluation
            result = evaluate(dataset=dataset, metrics=metrics, llm=self.evaluator_llm)

            print("   ✅ RAGAS evaluation completed")

            # Convert RAGAS result to dict dengan proper handling
            metrics_dict = {}
            if hasattr(result, "to_pandas"):
                # Jika result adalah RAGAS Result object
                df = result.to_pandas()
                for col in df.columns:
                    if col not in [
                        "user_input",
                        "retrieved_contexts",
                        "response",
                        "reference",
                    ]:
                        metrics_dict[col] = float(df[col].mean())
            else:
                # Fallback: convert langsung ke dict
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        metrics_dict[key] = float(value)
                    elif hasattr(value, "__iter__") and not isinstance(value, str):
                        try:
                            metrics_dict[key] = float(sum(value) / len(value))
                        except:
                            metrics_dict[key] = float(value)

            return metrics_dict

        except Exception as e:
            print(f"   ❌ Evaluation error: {e}")
            return {"error": str(e)}

    def run_comparison_evaluation(self, csv_file: str = "dataset/validasi_ta.csv"):
        """Run evaluasi perbandingan antara Single RAG vs Multi-Agent RAG"""
        print("🚀 RAG COMPARISON EVALUATION")
        print("=" * 60)
        print(f"📊 Using data from: {csv_file}")

        # Test endpoints
        single_status, multi_status = self.test_endpoints()
        if not single_status and not multi_status:
            print("❌ Both endpoints failed")
            return None
        elif not single_status:
            print("⚠️ Single RAG endpoint failed, running Multi-Agent only")
        elif not multi_status:
            print("⚠️ Multi-Agent endpoint failed, running Single RAG only")

        # Load dataset
        print(f"\n📖 Loading evaluation data...")
        try:
            df = pd.read_csv(csv_file, sep=",", quotechar='"', escapechar="\\")
            questions = df["question"].tolist()
            ground_truths = df["ground_truth"].tolist()
            print(f"✅ Loaded {len(questions)} questions from {csv_file}")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None

        results = {}

        # Evaluate Single RAG
        if single_status:
            print(f"\n🔄 Evaluating Single RAG System...")
            print("-" * 40)

            single_responses = []
            single_success_count = 0

            for i, (question, ground_truth) in enumerate(
                zip(questions, ground_truths), 1
            ):
                print(f"\n📝 Q{i}: {question[:80]}...")

                result = self.query_single_rag(question)

                if result["status"] == "success":
                    single_responses.append(
                        {
                            "user_input": question,
                            "retrieved_contexts": result["contexts"],
                            "response": result["answer"],
                            "reference": ground_truth,
                        }
                    )
                    single_success_count += 1
                else:
                    print(
                        f"   ⚠️ Skipping Q{i} due to {result['status']}: {result.get('error', '')}"
                    )

            print(
                f"\n📊 Single RAG: {single_success_count}/{len(questions)} successful"
            )

            if single_responses:
                single_metrics = self.evaluate_responses(single_responses)
                results["single_rag"] = {
                    "evaluation_metrics": single_metrics,
                    "successful_queries": single_success_count,
                    "total_queries": len(questions),
                    "success_rate": single_success_count / len(questions),
                    "raw_data": single_responses,
                }

        # Evaluate Multi-Agent RAG
        if multi_status:
            print(f"\n🔄 Evaluating Multi-Agent RAG System...")
            print("-" * 40)

            multi_responses = []
            multi_success_count = 0

            for i, (question, ground_truth) in enumerate(
                zip(questions, ground_truths), 1
            ):
                print(f"\n📝 Q{i}: {question[:80]}...")

                result = self.query_multi_agent(question)

                if result["status"] == "success":
                    multi_responses.append(
                        {
                            "user_input": question,
                            "retrieved_contexts": result["contexts"],
                            "response": result["answer"],
                            "reference": ground_truth,
                        }
                    )
                    multi_success_count += 1
                else:
                    print(
                        f"   ⚠️ Skipping Q{i} due to {result['status']}: {result.get('error', '')}"
                    )

            print(
                f"\n📊 Multi-Agent RAG: {multi_success_count}/{len(questions)} successful"
            )

            if multi_responses:
                multi_metrics = self.evaluate_responses(multi_responses)
                results["multi_agent"] = {
                    "evaluation_metrics": multi_metrics,
                    "successful_queries": multi_success_count,
                    "total_queries": len(questions),
                    "success_rate": multi_success_count / len(questions),
                    "raw_data": multi_responses,
                }

        if results:
            self.results = {
                **results,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "dataset": csv_file,
                    "single_rag_endpoint": self.single_rag_url,
                    "multi_agent_endpoint": self.multi_agent_url,
                },
            }

            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"result/rag_comparison_evaluation_{timestamp}.json"

            os.makedirs("result", exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)

            print(f"\n💾 Results saved to: {output_file}")
            return self.results
        else:
            print("❌ No successful responses to evaluate")
            return None

    def create_comparison_visualization(self, save_path=None):
        """Buat visualisasi perbandingan metrics antara Single RAG vs Multi-Agent RAG"""
        if not self.results:
            print("❌ No evaluation results available")
            return None

        # Prepare data for both systems
        systems = []
        system_names = []

        if "single_rag" in self.results:
            single_metrics = self.results["single_rag"]["evaluation_metrics"]
            if "error" not in single_metrics:
                systems.append(("Single RAG", single_metrics, "#FF6B6B"))
                system_names.append("Single RAG")

        if "multi_agent" in self.results:
            multi_metrics = self.results["multi_agent"]["evaluation_metrics"]
            if "error" not in multi_metrics:
                systems.append(("Multi-Agent RAG", multi_metrics, "#4ECDC4"))
                system_names.append("Multi-Agent RAG")

        if not systems:
            print("❌ No valid evaluation metrics found")
            return None

        # Metric configuration - Updated dengan semua metrics
        metric_names = [
            "Context Recall",
            "Context Precision",
            "Context Entity Recall",
            "Noise Sensitivity",
            "Faithfulness",
            "Answer Relevancy",
            "Factual Correctness",
        ]
        metric_keys = [
            "context_recall",
            "context_precision",
            "context_entity_recall",
            "noise_sensitivity(mode=relevant)",  # Fixed: add (mode=relevant)
            "faithfulness",
            "answer_relevancy",
            "factual_correctness(mode=f1)",
        ]

        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 16))

        # 1. Side-by-side Bar Chart
        ax1 = plt.subplot(3, 2, 1)
        x = np.arange(len(metric_names))
        width = 0.35

        for i, (system_name, metrics, color) in enumerate(systems):
            scores = [metrics.get(key, 0) for key in metric_keys]
            bars = ax1.bar(
                x + i * width, scores, width, label=system_name, color=color, alpha=0.8
            )

            # Add value labels
            for bar, score in zip(bars, scores):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{score:.3f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=9,
                )

        ax1.set_title(
            "📊 RAG Systems Comparison - Metrics Scores", fontsize=14, fontweight="bold"
        )
        ax1.set_ylabel("Score (0-1)", fontsize=12)
        ax1.set_ylim(0, 1.1)
        ax1.set_xticks(x + width / 2)
        ax1.set_xticklabels(metric_names, rotation=45, ha="right")
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)

        # 2. Radar Chart Comparison
        ax2 = plt.subplot(3, 2, 2, projection="polar")
        angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
        angles += angles[:1]  # Close circle

        for system_name, metrics, color in systems:
            scores = [metrics.get(key, 0) for key in metric_keys]
            scores_radar = scores + scores[:1]  # Close circle

            ax2.plot(
                angles, scores_radar, "o-", linewidth=2, label=system_name, color=color
            )
            ax2.fill(angles, scores_radar, alpha=0.15, color=color)

        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(metric_names)
        ax2.set_ylim(0, 1)
        ax2.set_title(
            "🎯 Performance Radar Comparison", fontsize=14, fontweight="bold", pad=20
        )
        ax2.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0))

        # 3. Success Rate Comparison
        ax3 = plt.subplot(3, 2, 3)
        success_rates = []
        labels = []
        colors = []

        for system_name, metrics, color in systems:
            if system_name == "Single RAG" and "single_rag" in self.results:
                success_rate = self.results["single_rag"]["success_rate"]
            elif system_name == "Multi-Agent RAG" and "multi_agent" in self.results:
                success_rate = self.results["multi_agent"]["success_rate"]
            else:
                success_rate = 0

            success_rates.append(success_rate * 100)
            labels.append(system_name)
            colors.append(color)

        bars = ax3.bar(labels, success_rates, color=colors, alpha=0.8)
        ax3.set_title("✅ Success Rate Comparison", fontsize=14, fontweight="bold")
        ax3.set_ylabel("Success Rate (%)", fontsize=12)
        ax3.set_ylim(0, 110)

        # Add percentage labels
        for bar, rate in zip(bars, success_rates):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{rate:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 4. Individual System Pie Charts
        for i, (system_name, metrics, color) in enumerate(systems):
            ax = plt.subplot(3, 2, 4 + i)
            scores = [metrics.get(key, 0) for key in metric_keys]

            wedges, texts, autotexts = ax.pie(
                scores,
                labels=metric_names,
                autopct="%1.2f",
                startangle=90,
                colors=plt.cm.Set3(np.linspace(0, 1, len(metric_names))),
            )
            ax.set_title(
                f"🥧 {system_name} Distribution", fontsize=12, fontweight="bold"
            )

        # 5. Detailed Summary Table
        ax5 = plt.subplot(3, 2, 6)
        ax5.axis("off")

        # Calculate average scores
        summary_text = "📈 COMPARISON SUMMARY\n" + "=" * 50 + "\n\n"

        for system_name, metrics, color in systems:
            scores = [metrics.get(key, 0) for key in metric_keys]
            avg_score = np.mean(scores)

            if system_name == "Single RAG" and "single_rag" in self.results:
                success_count = self.results["single_rag"]["successful_queries"]
                total_count = self.results["single_rag"]["total_queries"]
                success_rate = self.results["single_rag"]["success_rate"]
            elif system_name == "Multi-Agent RAG" and "multi_agent" in self.results:
                success_count = self.results["multi_agent"]["successful_queries"]
                total_count = self.results["multi_agent"]["total_queries"]
                success_rate = self.results["multi_agent"]["success_rate"]
            else:
                success_count = total_count = success_rate = 0

            summary_text += f"🔹 {system_name}:\n"
            summary_text += (
                f"   Success Rate: {success_rate:.1%} ({success_count}/{total_count})\n"
            )
            summary_text += f"   Context Recall: {scores[0]:.3f}\n"
            summary_text += f"   Context Precision: {scores[1]:.3f}\n"
            summary_text += f"   Context Entity Recall: {scores[2]:.3f}\n"
            summary_text += f"   Noise Sensitivity: {scores[3]:.3f}\n"
            summary_text += f"   Faithfulness: {scores[4]:.3f}\n"
            summary_text += f"   Answer Relevancy: {scores[5]:.3f}\n"
            summary_text += f"   Factual Correctness: {scores[6]:.3f}\n"
            summary_text += f"   📊 Average Score: {avg_score:.3f}\n\n"

        # Winner analysis
        if len(systems) == 2:
            single_avg = np.mean([systems[0][1].get(key, 0) for key in metric_keys])
            multi_avg = np.mean([systems[1][1].get(key, 0) for key in metric_keys])

            if single_avg > multi_avg:
                winner = "Single RAG"
                diff = single_avg - multi_avg
            elif multi_avg > single_avg:
                winner = "Multi-Agent RAG"
                diff = multi_avg - single_avg
            else:
                winner = "TIE"
                diff = 0

            summary_text += f"🏆 WINNER: {winner}\n"
            if winner != "TIE":
                summary_text += f"📈 Performance Gap: {diff:.3f} ({diff*100:.1f}%)\n"

        summary_text += f"\n🕒 Timestamp: {self.results['metadata']['timestamp'][:19]}"

        ax5.text(
            0.05,
            0.95,
            summary_text,
            transform=ax5.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"📊 Comparison visualization saved: {save_path}")

        plt.show()
        return fig

    def run_complete_analysis(self, csv_file: str = "dataset/validasi_ta.csv"):
        """Run evaluasi + visualisasi perbandingan lengkap"""
        print("🎯 RAG COMPARISON - COMPLETE ANALYSIS")
        print("=" * 60)

        # 1. Run evaluation
        results = self.run_comparison_evaluation(csv_file)

        if not results:
            print("❌ Evaluation failed")
            return None

        # 2. Create visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"rag_comparison_results_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n🎨 Creating comparison visualizations...")
        print(f"📁 Output directory: {output_dir}")

        viz_path = f"{output_dir}/rag_comparison_analysis.png"
        self.create_comparison_visualization(viz_path)

        # 3. Print detailed results
        self.print_detailed_results()

        print(f"\n✅ Complete analysis finished!")
        print(f"📁 Check folder: {output_dir}")
        print(f"📊 Visualization: {viz_path}")

        return output_dir

    def print_detailed_results(self):
        """Print hasil evaluasi detail untuk kedua sistem"""
        if not self.results:
            return

        print("\n" + "=" * 60)
        print("📊 DETAILED COMPARISON RESULTS")
        print("=" * 60)

        # Single RAG Results
        if "single_rag" in self.results:
            system_data = self.results["single_rag"]
            print(f"🔹 Single RAG System:")
            print(f"   Success Rate: {system_data['success_rate']:.1%}")

            if (
                "evaluation_metrics" in system_data
                and "error" not in system_data["evaluation_metrics"]
            ):
                metrics = system_data["evaluation_metrics"]
                print(f"   Context Recall: {metrics.get('context_recall', 0):.3f}")
                print(
                    f"   Context Precision: {metrics.get('context_precision', 0):.3f}"
                )
                print(
                    f"   Context Entity Recall: {metrics.get('context_entity_recall', 0):.3f}"
                )
                print(
                    f"   Noise Sensitivity: {metrics.get('noise_sensitivity', 0):.3f}"
                )
                print(f"   Faithfulness: {metrics.get('faithfulness', 0):.3f}")
                print(f"   Answer Relevancy: {metrics.get('answer_relevancy', 0):.3f}")
                print(
                    f"   Factual Correctness: {metrics.get('factual_correctness(mode=f1)', 0):.3f}"
                )

                # Calculate average
                valid_metrics = [
                    v
                    for k, v in metrics.items()
                    if k != "error" and isinstance(v, (int, float))
                ]
                if valid_metrics:
                    avg_score = sum(valid_metrics) / len(valid_metrics)
                    print(f"   📈 Average Score: {avg_score:.3f}")
            else:
                print("   ❌ Evaluation failed")
        else:
            print("🔹 Single RAG System: Not evaluated")

        print()

        # Multi-Agent RAG Results
        if "multi_agent" in self.results:
            system_data = self.results["multi_agent"]
            print(f"🤖 Multi-Agent RAG System:")
            print(f"   Success Rate: {system_data['success_rate']:.1%}")

            if (
                "evaluation_metrics" in system_data
                and "error" not in system_data["evaluation_metrics"]
            ):
                metrics = system_data["evaluation_metrics"]
                print(f"   Context Recall: {metrics.get('context_recall', 0):.3f}")
                print(
                    f"   Context Precision: {metrics.get('context_precision', 0):.3f}"
                )
                print(
                    f"   Context Entity Recall: {metrics.get('context_entity_recall', 0):.3f}"
                )
                print(
                    f"   Noise Sensitivity: {metrics.get('noise_sensitivity', 0):.3f}"
                )
                print(f"   Faithfulness: {metrics.get('faithfulness', 0):.3f}")
                print(f"   Answer Relevancy: {metrics.get('answer_relevancy', 0):.3f}")
                print(
                    f"   Factual Correctness: {metrics.get('factual_correctness(mode=f1)', 0):.3f}"
                )

                # Calculate average
                valid_metrics = [
                    v
                    for k, v in metrics.items()
                    if k != "error" and isinstance(v, (int, float))
                ]
                if valid_metrics:
                    avg_score = sum(valid_metrics) / len(valid_metrics)
                    print(f"   📈 Average Score: {avg_score:.3f}")
            else:
                print("   ❌ Evaluation failed")
        else:
            print("🤖 Multi-Agent RAG System: Not evaluated")

        # Winner Analysis
        if "single_rag" in self.results and "multi_agent" in self.results:
            single_metrics = self.results["single_rag"]["evaluation_metrics"]
            multi_metrics = self.results["multi_agent"]["evaluation_metrics"]

            if "error" not in single_metrics and "error" not in multi_metrics:
                metric_keys = [
                    "context_recall",
                    "context_precision",
                    "context_entity_recall",
                    "noise_sensitivity",
                    "faithfulness",
                    "answer_relevancy",
                    "factual_correctness(mode=f1)",
                ]

                single_avg = np.mean(
                    [single_metrics.get(key, 0) for key in metric_keys]
                )
                multi_avg = np.mean([multi_metrics.get(key, 0) for key in metric_keys])

                print(f"\n🏆 WINNER ANALYSIS:")
                if single_avg > multi_avg:
                    winner = "Single RAG"
                    diff = single_avg - multi_avg
                elif multi_avg > single_avg:
                    winner = "Multi-Agent RAG"
                    diff = multi_avg - single_avg
                else:
                    winner = "TIE"
                    diff = 0

                print(f"   🏆 Winner: {winner}")
                if winner != "TIE":
                    print(f"   📈 Performance Gap: {diff:.3f} ({diff*100:.1f}%)")
                else:
                    print(f"   🤝 Both systems performed equally")

        print("=" * 60)


def main():
    """Main function"""
    print("🎯 RAG COMPARISON EVALUATION & VISUALIZATION")
    print("=" * 60)

    # Get API keys
    api_key = os.getenv("API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    if not api_key:
        print("⚠️  API_KEY tidak ditemukan di environment variables")
    if not openai_key:
        print("⚠️  OPENAI_API_KEY tidak ditemukan di environment variables")

    print(
        f"🔑 Using API Key: {api_key[:10]}..."
        if len(api_key) > 10
        else f"🔑 Using API Key: {api_key}"
    )
    print(
        f"🤖 Using OpenAI Key: {openai_key[:10]}..."
        if len(openai_key) > 10
        else f"🤖 Using OpenAI Key: {openai_key}"
    )

    # Initialize evaluator
    evaluator = RAGComparisonEvaluator(api_key=api_key, openai_api_key=openai_key)

    # Check if user wants to test parsing first
    if len(sys.argv) > 1 and sys.argv[1] == "--test-parsing":
        print("\n🧪 Running parsing test first...")
        evaluator.test_parsing()
        return

    # Run complete analysis
    output_dir = evaluator.run_complete_analysis()

    if output_dir:
        print(f"\n🎉 Comparison Analysis completed!")
        print(f"📁 Results in: {output_dir}")
        print("📊 Comparison Graphs created:")
        print("   - Side-by-side Bar Chart (metric scores)")
        print("   - Radar Chart Comparison")
        print("   - Success Rate Comparison")
        print("   - Individual System Pie Charts")
        print("   - Detailed Summary with Winner Analysis")

    print(
        f"\n💡 Tip: Use 'python single_rag_evaluation.py --test-parsing' to test response parsing only"
    )


if __name__ == "__main__":
    main()

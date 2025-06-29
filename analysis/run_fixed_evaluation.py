#!/usr/bin/env python3
"""
🔧 EVALUASI ULANG DENGAN TIMEOUT FIXED
Script untuk menjalankan ulang evaluasi setelah memperbaiki timeout issue
"""

import os
import json
import pandas as pd
import requests
import time
from datetime import datetime
from typing import Dict, List, Any
from datasets import Dataset

# Import evaluasi library
try:
    from ragas import evaluate
    from ragas.metrics import (
        context_recall,
        faithfulness,
        answer_relevancy,
        FactualCorrectness,
    )
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI

    print("✅ Ragas libraries imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)


class FixedRAGEvaluator:
    """Evaluator dengan timeout yang diperbaiki"""

    def __init__(self, api_key: str = None, openai_api_key: str = None):
        """Initialize evaluator"""
        self.api_key = api_key or "your_secure_api_key_here"
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        # Setup evaluator LLM
        if self.openai_api_key:
            llm = ChatOpenAI(model="gpt-4o-mini", api_key=self.openai_api_key)
            self.evaluator_llm = LangchainLLMWrapper(llm)
        else:
            print("⚠️ Warning: No OpenAI API key provided")
            self.evaluator_llm = None

        # API endpoints
        self.simple_api_url = "http://localhost:8080/simple/query"
        self.multi_api_url = "http://localhost:8080/multi/query"

        # Fixed timeouts - Multi-Agent gets more time
        self.simple_timeout = 60
        self.multi_timeout = 120  # 🔧 FIXED: Increased timeout for Multi-Agent

        print(f"📋 Timeout settings:")
        print(f"   Simple RAG: {self.simple_timeout}s")
        print(f"   Multi-Agent RAG: {self.multi_timeout}s")

    def test_endpoints(self):
        """Test koneksi ke endpoints"""
        print("\n🔍 Testing endpoints...")

        # Test simple API
        try:
            response = requests.get("http://localhost:8080/simple/health", timeout=10)
            if response.status_code == 200:
                print("✅ Simple API: Online")
            else:
                print(f"⚠️ Simple API: Status {response.status_code}")
        except Exception as e:
            print(f"❌ Simple API: Error - {e}")

        # Test multi API
        try:
            response = requests.get("http://localhost:8080/multi/health", timeout=10)
            if response.status_code == 200:
                print("✅ Multi-Agent API: Online")
            else:
                print(f"⚠️ Multi-Agent API: Status {response.status_code}")
        except Exception as e:
            print(f"❌ Multi-Agent API: Error - {e}")

    def query_api(self, url: str, question: str, timeout: int) -> Dict[str, Any]:
        """Query API dengan timeout yang sesuai"""
        headers = {"X-API-Key": self.api_key, "Content-Type": "application/json"}

        payload = {
            "query": question,
            "embedding_model": "text-embedding-3-small",
            "previous_responses": [],
        }

        start_time = time.time()

        try:
            print(f"   ⏳ Querying (timeout: {timeout}s)...")
            response = requests.post(
                url, json=payload, headers=headers, timeout=timeout
            )
            response_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()

                # Extract contexts
                contexts = []
                if "referenced_documents" in data:
                    for doc in data["referenced_documents"]:
                        if isinstance(doc, dict) and "content" in doc:
                            contexts.append(doc["content"])
                        elif isinstance(doc, str):
                            contexts.append(doc)

                print(
                    f"   ✅ Success in {response_time:.1f}s ({len(contexts)} contexts)"
                )

                return {
                    "status": "success",
                    "answer": data.get("answer", ""),
                    "contexts": contexts,
                    "response_time": response_time,
                    "model_info": data.get("model_info", {}),
                }
            else:
                print(f"   ❌ HTTP Error {response.status_code}")
                return {
                    "status": "error",
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "response_time": response_time,
                }

        except requests.exceptions.Timeout:
            response_time = time.time() - start_time
            print(f"   ⏰ Timeout after {response_time:.1f}s")
            return {
                "status": "timeout",
                "error": f"Request timeout after {timeout}s",
                "response_time": response_time,
            }
        except Exception as e:
            response_time = time.time() - start_time
            print(f"   ❌ Error: {e}")
            return {"status": "error", "error": str(e), "response_time": response_time}

    def evaluate_responses(self, dataset_dict: List[Dict]) -> Dict[str, float]:
        """Evaluate dengan Ragas"""
        if not self.evaluator_llm:
            print("❌ No evaluator LLM available")
            return {}

        try:
            # Convert to Ragas dataset format
            dataset = Dataset.from_list(dataset_dict)

            # Define metrics
            metrics = [
                context_recall,
                faithfulness,
                FactualCorrectness(mode="f1"),
                answer_relevancy,
            ]

            print(f"   📊 Evaluating {len(dataset_dict)} responses...")

            # Run evaluation
            result = evaluate(dataset=dataset, metrics=metrics, llm=self.evaluator_llm)

            print("   ✅ Evaluation completed")
            return dict(result)

        except Exception as e:
            print(f"   ❌ Evaluation error: {e}")
            return {"error": str(e)}

    def run_complete_evaluation(self, csv_file: str = "validasi_ta.csv"):
        """Run evaluasi lengkap dengan timeout fix"""
        print("🚀 EVALUASI ULANG DENGAN TIMEOUT FIXED")
        print("=" * 60)

        # Test endpoints first
        self.test_endpoints()

        # Load data
        print(f"\n📖 Loading data dari {csv_file}...")
        try:
            df = pd.read_csv(csv_file)
            questions = df["question"].tolist()
            ground_truths = df["ground_truth"].tolist()
            print(f"✅ Loaded {len(questions)} questions")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None

        results = {}

        # Evaluate Simple RAG
        print(f"\n🔄 Evaluating Simple RAG System...")
        print("-" * 40)

        simple_responses = []
        simple_success_count = 0

        for i, (question, ground_truth) in enumerate(zip(questions, ground_truths), 1):
            print(f"\n📝 Q{i}: {question[:100]}...")

            result = self.query_api(self.simple_api_url, question, self.simple_timeout)

            if result["status"] == "success":
                simple_responses.append(
                    {
                        "user_input": question,
                        "retrieved_contexts": result["contexts"],
                        "response": result["answer"],
                        "reference": ground_truth,
                    }
                )
                simple_success_count += 1
            else:
                print(f"   ⚠️ Skipping Q{i} due to {result['status']}")

        print(f"\n📊 Simple RAG: {simple_success_count}/{len(questions)} successful")

        if simple_responses:
            simple_metrics = self.evaluate_responses(simple_responses)
            results["single_rag"] = {
                "evaluation_metrics": simple_metrics,
                "successful_queries": simple_success_count,
                "total_queries": len(questions),
                "success_rate": simple_success_count / len(questions),
            }
        else:
            results["single_rag"] = {"error": "No successful responses"}

        # Evaluate Multi-Agent RAG
        print(f"\n🔄 Evaluating Multi-Agent RAG System...")
        print("-" * 40)

        multi_responses = []
        multi_success_count = 0

        for i, (question, ground_truth) in enumerate(zip(questions, ground_truths), 1):
            print(f"\n📝 Q{i}: {question[:100]}...")

            # 🔧 Use increased timeout for Multi-Agent
            result = self.query_api(self.multi_api_url, question, self.multi_timeout)

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
                print(f"   ⚠️ Skipping Q{i} due to {result['status']}")

        print(
            f"\n📊 Multi-Agent RAG: {multi_success_count}/{len(questions)} successful"
        )

        if multi_responses:
            multi_metrics = self.evaluate_responses(multi_responses)
            results["multi_agent_rag"] = {
                "evaluation_metrics": multi_metrics,
                "successful_queries": multi_success_count,
                "total_queries": len(questions),
                "success_rate": multi_success_count / len(questions),
            }
        else:
            results["multi_agent_rag"] = {"error": "No successful responses"}

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"benchmark_results/ragas_evaluation_{timestamp}_fixed.json"

        os.makedirs("benchmark_results", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n💾 Results saved to: {output_file}")

        # Print summary
        self.print_summary(results)

        return results

    def print_summary(self, results: Dict[str, Any]):
        """Print evaluasi summary"""
        print("\n" + "=" * 60)
        print("📊 EVALUATION SUMMARY")
        print("=" * 60)

        for system_name, system_data in results.items():
            display_name = (
                "Single RAG" if system_name == "single_rag" else "Multi-Agent RAG"
            )
            print(f"\n🔹 {display_name}:")

            if "error" in system_data:
                print(f"   ❌ Error: {system_data['error']}")
                continue

            success_rate = system_data.get("success_rate", 0)
            print(f"   Success Rate: {success_rate:.1%}")

            if (
                "evaluation_metrics" in system_data
                and "error" not in system_data["evaluation_metrics"]
            ):
                metrics = system_data["evaluation_metrics"]
                print(f"   Context Recall: {metrics.get('context_recall', 0):.3f}")
                print(f"   Faithfulness: {metrics.get('faithfulness', 0):.3f}")
                print(
                    f"   Factual Correctness: {metrics.get('factual_correctness(mode=f1)', 0):.3f}"
                )
                print(f"   Answer Relevancy: {metrics.get('answer_relevancy', 0):.3f}")

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


def main():
    """Main function"""
    # Check if OpenAI API key is available
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("⚠️ Warning: OPENAI_API_KEY not found in environment")
        print("📝 Please set it with: export OPENAI_API_KEY='your-key'")
        return

    print("🔧 STARTING FIXED EVALUATION")
    print("🎯 Multi-Agent RAG timeout increased to 120s")
    print()

    evaluator = FixedRAGEvaluator(openai_api_key=openai_key)
    results = evaluator.run_complete_evaluation()

    if results:
        print("\n🎉 Fixed evaluation completed successfully!")
        print("💡 Check if Multi-Agent RAG performance improved")
    else:
        print("\n❌ Evaluation failed")


if __name__ == "__main__":
    main()

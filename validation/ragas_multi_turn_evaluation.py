"""
Ragas Multi-Turn Conversation Evaluation untuk Multi Agent System
Script lengkap untuk evaluasi multi-turn dengan fokus pada Query Enhancement Based on History

Fitur Utama:
  - Test referensi anaforik ("tadi", "itu", "yang barusan")
  - Test incomplete queries yang memerlukan context
  - Test pronoun resolution dan implicit references
  - AspectCritic evaluation untuk conversation quality

Usage:
  python ragas_multi_turn_evaluation.py --demo     # Quick demo (3 turns)
  python ragas_multi_turn_evaluation.py            # Full evaluation (15 turns)
  python ragas_multi_turn_evaluation.py --help     # Show help

Environment:
  OPENAI_API_KEY=your_openai_api_key_here
"""

import os
import json
import requests
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

# Ragas imports
from ragas.metrics import AspectCritic
from ragas.dataset_schema import MultiTurnSample, EvaluationDataset
from ragas.messages import HumanMessage, AIMessage
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


class MultiAgentRagasEvaluator:
    def __init__(self):
        # Check OpenAI API key
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key or openai_key.strip() == "":
            print(
                "[ERR] Configuration Error: OPENAI_API_KEY environment variable not found."
            )
            print("[TIP] Set your OpenAI API key: export OPENAI_API_KEY=your_key")
            raise ValueError(
                "OPENAI_API_KEY environment variable not found. Set it with: export OPENAI_API_KEY=your_key"
            )

        # Setup evaluator LLM untuk Ragas
        self.evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-nano"))

        # API configuration untuk multi agent (localhost)
        self.api_url = "http://localhost:8080/api/chat"
        self.api_key = "your_secure_api_key_here"
        self.headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}

        # Setup AspectCritic metrics untuk evaluasi multi-turn
        self.setup_evaluation_metrics()

    def setup_evaluation_metrics(self):
        """Setup berbagai AspectCritic metrics untuk evaluasi multi-turn conversation"""

        # 1. Consistency Metric - Apakah agent konsisten dalam menjawab
        consistency_definition = """
        Berikan nilai 1 jika AI mempertahankan konsistensi sepanjang percakapan, yang berarti:
        - AI tidak memberikan pernyataan yang bertentangan dengan jawaban sebelumnya
        - Informasi yang diberikan tetap koheren di semua giliran percakapan
        - AI mengingat konteks dari pertukaran percakapan sebelumnya
        Jika tidak, berikan nilai 0.
        """

        # 2. Legal Accuracy Metric - Khusus untuk domain hukum kesehatan
        legal_accuracy_definition = """
        Berikan nilai 1 jika AI memberikan informasi hukum yang akurat tentang hukum kesehatan Indonesia, yang berarti:
        - Mengutip undang-undang, peraturan, atau ketentuan hukum yang relevan dengan benar
        - Memberikan informasi hukum yang faktual dan terkini
        - Menghindari klaim hukum yang tidak berdasar atau spekulatif
        Jika tidak, berikan nilai 0.
        """

        # 3. Completeness Metric - Apakah agent menyelesaikan semua pertanyaan
        completeness_definition = """
        Berikan nilai 1 jika AI menjawab semua aspek dari pertanyaan pengguna sepanjang percakapan, yang berarti:
        - Semua pertanyaan yang diajukan pengguna telah dijawab
        - AI tidak melupakan atau mengabaikan bagian-bagian dari pertanyaan multi-bagian
        - Pertanyaan lanjutan ditangani dengan tepat
        Jika tidak, berikan nilai 0.
        """

        # 4. Helpfulness Metric - Apakah responnya membantu
        helpfulness_definition = """
        Berikan nilai 1 jika AI memberikan respons yang membantu dan dapat ditindaklanjuti, yang berarti:
        - Respons secara langsung menjawab kebutuhan pengguna
        - Informasi disajikan dengan jelas dan mudah dipahami
        - AI menawarkan informasi tambahan yang relevan bila diperlukan
        Jika tidak, berikan nilai 0.
        """

        self.consistency_metric = AspectCritic(
            name="Consistency Metric",
            definition=consistency_definition,
            llm=self.evaluator_llm,
        )

        self.legal_accuracy_metric = AspectCritic(
            name="Legal Accuracy Metric",
            definition=legal_accuracy_definition,
            llm=self.evaluator_llm,
        )

        self.completeness_metric = AspectCritic(
            name="Completeness Metric",
            definition=completeness_definition,
            llm=self.evaluator_llm,
        )

        self.helpfulness_metric = AspectCritic(
            name="Helpfulness Metric",
            definition=helpfulness_definition,
            llm=self.evaluator_llm,
        )

        self.all_metrics = [
            self.consistency_metric,
            self.legal_accuracy_metric,
            self.completeness_metric,
            self.helpfulness_metric,
        ]

    def call_multi_agent_api(
        self, query: str, previous_responses: List[Dict] = None
    ) -> Dict:
        """Memanggil API multi agent dengan query dan previous responses"""

        if previous_responses is None:
            previous_responses = []

        payload = {
            "query": query,
            "embedding_model": "large",
            "previous_responses": previous_responses,
            "use_parallel_execution": "false",
        }

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=self.headers,
                timeout=300,  # Increased timeout
            )
            response.raise_for_status()
            api_result = response.json()

            # Check for incomplete responses
            if (
                "answer" in api_result
                and "Agent stopped due to max iterations" in api_result["answer"]
            ):
                print(f"⚠️  Warning: Agent hit max iterations limit")

            return api_result
        except requests.exceptions.RequestException as e:
            print(f"Error calling API: {e}")
            return {"error": str(e)}

    def create_multi_turn_conversation(
        self, conversation_script: List[str]
    ) -> MultiTurnSample:
        """Membuat percakapan multi-turn dengan memanggil API secara berurutan"""

        messages = []
        previous_responses = []

        for i, user_query in enumerate(conversation_script):
            print(f"Turn {i+1}: {user_query}")

            # Add user message
            messages.append(HumanMessage(content=user_query))

            # Call API
            api_response = self.call_multi_agent_api(user_query, previous_responses)

            if "error" in api_response:
                ai_content = f"Error: {api_response['error']}"
            else:
                ai_content = api_response.get("answer", "No response received")

            # Add AI message
            messages.append(AIMessage(content=ai_content))

            # Update previous responses untuk context
            previous_responses.append(
                {
                    "query": user_query,
                    "response": ai_content,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            print(f"AI Response: {ai_content[:200]}...")
            print("-" * 50)

            # Longer delay between calls to prevent overload
            time.sleep(3)

        return MultiTurnSample(user_input=messages)

    def evaluate_conversation(self, conversation_sample: MultiTurnSample) -> Dict:
        """Mengevaluasi percakapan menggunakan Ragas metrics"""

        dataset = EvaluationDataset(samples=[conversation_sample])

        print("Evaluating conversation with Ragas...")
        result = evaluate(dataset=dataset, metrics=self.all_metrics)

        return result.to_pandas().to_dict("records")[0]

    def run_demo(self):
        """Demo cepat dengan 1 skenario pendek"""

        print(">> Demo Ragas Multi-Turn Evaluation")
        print("=" * 50)

        # Test API connection
        print("[CONN] Testing API connection...")
        test_response = self.call_multi_agent_api("Test connection")

        if "error" in test_response:
            print(f"[ERR] API connection failed: {test_response['error']}")
            print(
                "[TIP] Pastikan multi agent system running di http://localhost:8080/api/chat"
            )
            return False

        print("[OK] API connection successful!")

        # Quick conversation test with context dependency
        print("\n[TEST] Running quick conversation test...")
        quick_conversation = [
            "Apa yang dimaksud dengan informed consent dalam praktik medis?",
            "Jadi gimana kalau pasiennya tidak sadar tapi butuh tindakan darurat, aturan yang tadi masih berlaku?",
            "Untuk kasus seperti itu, dokternya bisa kena masalah hukum nggak?",
        ]

        print(f"[INFO] Conversation scenario: {len(quick_conversation)} turns")
        for i, turn in enumerate(quick_conversation, 1):
            print(f"   Turn {i}: {turn[:60]}{'...' if len(turn) > 60 else ''}")

        # Create and evaluate conversation
        print("\n[RUN] Creating multi-turn conversation...")
        conversation_sample = self.create_multi_turn_conversation(quick_conversation)

        print("\n[EVAL] Evaluating with Ragas metrics...")
        evaluation_result = self.evaluate_conversation(conversation_sample)

        # Display results
        print(f"\n{'='*50}")
        print("EVALUATION RESULTS")
        print(f"{'='*50}")

        total_metrics = 0
        passed_metrics = 0

        for metric, score in evaluation_result.items():
            if "Metric" in metric:
                total_metrics += 1
                if score == 1:
                    passed_metrics += 1

                status = "[PASS]" if score == 1 else "[FAIL]"
                print(f"  {metric}: {score} {status}")

        # Summary
        print(f"\n[SUMMARY]:")
        print(f"  Total Metrics: {total_metrics}")
        print(f"  Passed: {passed_metrics}")
        print(f"  Failed: {total_metrics - passed_metrics}")
        print(f"  Success Rate: {(passed_metrics/total_metrics)*100:.1f}%")

        print(f"\n[DONE] Demo completed successfully!")
        return True

    def run_full_evaluation(self):
        """Menjalankan evaluasi lengkap dengan semua skenario"""

        scenarios = [
            {
                "name": "Skenario Context-Heavy: Aborsi Legal dengan Follow-up Kompleks",
                "conversation": [
                    "Dalam situasi apa saja tenaga kesehatan atau tenaga medis dapat melakukan tindakan aborsi yang dibenarkan oleh hukum di Indonesia, dan apa saja syarat yang harus dipenuhi?",
                    "Jadi gimana berdasarkan yang tadi, kalau kasusnya perempuan hamil dalam keadaan tidak sadar dan butuh tindakan darurat?",
                    "Terus untuk syarat yang disebutkan barusan, apa bedanya kalau di RS swasta vs pemerintah?",
                ],
            },
            {
                "name": "Skenario Referensi Implisit: Hak Pasien Complex Chain",
                "conversation": [
                    "Apa saja hak fundamental yang dimiliki setiap orang dalam memperoleh pelayanan kesehatan menurut peraturan perundang-undangan yang berlaku?",
                    "Dari hak-hak yang disebutkan tadi, mana yang bisa dikecualikan dalam keadaan darurat?",
                    "Nah untuk pengecualian itu, prosedurnya seperti apa?",
                ],
            },
            {
                "name": "Skenario Query Enhancement: BPJS Funding Deep Dive",
                "conversation": [
                    "Dari mana saja sumber pendanaan Aset BPJS, dan secara spesifik, untuk apa saja Dana Jaminan Sosial yang dikelola oleh BPJS dapat dimanfaatkan sesuai ketentuan?",
                    "Yang tadi disebutkan ada hasil pengembangan aset, itu maksudnya investasi ya? Gimana aturannya?",
                    "Untuk investasi yang barusan dibahas, ada batasan atau larangan tertentu nggak?",
                ],
            },
            {
                "name": "Skenario Advanced Context: Wabah Multi-Layer Questions",
                "conversation": [
                    "Bagaimana definisi dan kriteria Wabah Penyakit Menular serta langkah-langkah penanggulangannya?",
                    "Dari kriteria yang dijelaskan tadi, siapa yang berwenang menetapkan suatu daerah sebagai terjangkit wabah?",
                    "Setelah ditetapkan seperti itu, apa langkah konkret pertama yang harus diambil?",
                ],
            },
            {
                "name": "Skenario Pronoun Resolution: Tanggung Jawab Pemerintah",
                "conversation": [
                    "Apa saja tanggung jawab pemerintah dalam menyelenggarakan upaya kesehatan bagi masyarakat?",
                    "Yang bagian penelitian dan pengembangan itu, implementasinya gimana di tingkat daerah?",
                    "Kalau daerahnya tidak mampu melaksanakan hal tersebut, pusat bisa intervensi nggak?",
                ],
            },
        ]

        results = []

        for scenario in scenarios:
            print(f"\n{'='*60}")
            print(f"RUNNING: {scenario['name']}")
            print(f"{'='*60}")

            try:
                # Create conversation
                conversation_sample = self.create_multi_turn_conversation(
                    scenario["conversation"]
                )

                # Evaluate conversation
                evaluation_result = self.evaluate_conversation(conversation_sample)

                # Store results
                result = {
                    "scenario_name": scenario["name"],
                    "evaluation": evaluation_result,
                    "conversation_turns": len(scenario["conversation"]),
                    "timestamp": datetime.now().isoformat(),
                }

                results.append(result)

                print(f"\nEVALUATION RESULTS for {scenario['name']}:")
                for metric, score in evaluation_result.items():
                    if "Metric" in metric:
                        status = "[PASS]" if score == 1 else "[FAIL]"
                        print(f"  {metric}: {score} {status}")

            except Exception as e:
                print(f"Error in scenario {scenario['name']}: {e}")
                results.append(
                    {
                        "scenario_name": scenario["name"],
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        # Save results
        self.save_results(results)

        # Print summary
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")

        for result in results:
            if "error" not in result:
                print(f"\nScenario: {result['scenario_name']}")
                print(f"Turns: {result['conversation_turns']}")
                evaluation = result["evaluation"]

                for metric, score in evaluation.items():
                    if "Metric" in metric:
                        status = "[PASS]" if score == 1 else "[FAIL]"
                        print(f"  {metric}: {score} {status}")
            else:
                print(
                    f"\nScenario: {result['scenario_name']} - ERROR: {result['error']}"
                )

        return results

    def save_results(self, results: List[Dict], filename: str = None):
        """Menyimpan hasil evaluasi ke file JSON"""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ragas_multi_turn_evaluation_{timestamp}.json"

        filepath = os.path.join("validation/result", filename)
        os.makedirs("validation/result", exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {filepath}")
        return filepath


def show_help():
    """Menampilkan help dan usage information"""

    help_text = """
>> Ragas Multi-Turn Conversation Evaluation
==========================================

Script untuk evaluasi multi-turn conversation menggunakan Ragas AspectCritic
untuk sistem multi agent berbasis hukum kesehatan Indonesia.

USAGE:
  python ragas_multi_turn_evaluation.py --demo     # Quick demo (3 turns with context)
  python ragas_multi_turn_evaluation.py            # Full evaluation (15 turns) 
  python ragas_multi_turn_evaluation.py --help     # Show this help

SETUP:
  1. Set environment variable: export OPENAI_API_KEY=your_openai_api_key_here
  2. Pastikan multi agent running di: http://localhost:8080/api/chat
  3. Install dependencies: pip install ragas langchain-openai python-dotenv

METRICS EVALUATED:
  [OK] Consistency    - Agent konsisten across turns
  [OK] Legal Accuracy - Akurasi informasi hukum kesehatan
  [OK] Completeness   - Semua pertanyaan terjawab
  [OK] Helpfulness    - Response helpful dan actionable

SCENARIOS (Full Evaluation):
  [*] Context-Heavy: Aborsi Legal (3 turns)     - "gimana berdasarkan tadi", referensi implisit
  [*] Referensi Implisit: Hak Pasien (3 turns)  - "yang disebutkan tadi", "pengecualian itu"
  [*] Query Enhancement: BPJS Funding (3 turns) - "yang barusan dibahas", context dependency
  [*] Advanced Context: Wabah Handling (3 turns) - "setelah ditetapkan seperti itu"
  [*] Pronoun Resolution: Tanggung Jawab (3 turns) - "hal tersebut", "mereka", "itu"

OUTPUT:
  [FILE] validation/result/ragas_multi_turn_evaluation_TIMESTAMP.json

EXAMPLE:
  # Quick test
  export OPENAI_API_KEY=sk-...
  python ragas_multi_turn_evaluation.py --demo

  # Full evaluation
  python ragas_multi_turn_evaluation.py
"""
    print(help_text)


def main():
    """Main function dengan argument parsing"""

    parser = argparse.ArgumentParser(
        description="Ragas Multi-Turn Evaluation for Multi Agent"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run quick demo (3 turns with context dependency)",
    )
    parser.add_argument(
        "--help-detailed", action="store_true", help="Show detailed help"
    )

    args = parser.parse_args()

    if args.help_detailed:
        show_help()
        return

    # Default behavior: run full evaluation if no specific mode specified
    run_full_evaluation = not args.demo

    try:
        evaluator = MultiAgentRagasEvaluator()

        if args.demo:
            print("Running Demo Mode...")
            success = evaluator.run_demo()
            if success:
                print(
                    "\n[TIP] For full evaluation with 15 turns testing query enhancement:"
                )
                print("       python ragas_multi_turn_evaluation.py")

        elif run_full_evaluation:
            print("Running Full Evaluation Mode...")

            # Test API connection first
            print("Testing API connection...")
            test_response = evaluator.call_multi_agent_api("Test connection")

            if "error" in test_response:
                print(f"API connection failed: {test_response['error']}")
                print(
                    "Please make sure your multi agent system is running on http://localhost:8080/api/chat"
                )
                return

            print("API connection successful!")

            # Run full evaluation
            results = evaluator.run_full_evaluation()
            print(
                f"\n[DONE] Full evaluation completed! Check validation/result/ for detailed results."
            )

    except ValueError as e:
        print(f"[ERR] Configuration Error: {e}")
        return
    except Exception as e:
        print(f"[ERR] Unexpected error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
🚀 Simple RAG Benchmark Script
Script sederhana untuk membandingkan Simple API vs Multi API
Tanpa dependency RAGAs yang kompleks
"""

import asyncio
import json
import time
import requests
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
from pathlib import Path

class SimpleBenchmark:
    def __init__(self):
        self.api_base = "http://localhost:8080"
        self.api_key = "your_secure_api_key_here"
        self.headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }
        self.setup_output_dir()
    
    def setup_output_dir(self):
        """Setup output directory"""
        self.output_dir = Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)
    
    def get_test_questions(self) -> List[Dict]:
        """Daftar pertanyaan test untuk hukum kesehatan"""
        return [
            {
                "question": "Apa definisi tenaga kesehatan menurut undang-undang?",
                "category": "definisi",
                "difficulty": "easy"
            },
            {
                "question": "Apa sanksi hukum bagi praktik kedokteran tanpa izin?",
                "category": "sanksi", 
                "difficulty": "medium"
            },
            {
                "question": "Bagaimana prosedur pengajuan izin praktik dokter spesialis?",
                "category": "prosedur",
                "difficulty": "medium"
            },
            {
                "question": "Apa kewajiban dokter dalam memberikan informed consent?",
                "category": "etika",
                "difficulty": "medium"
            },
            {
                "question": "Bagaimana hubungan antara UU Praktik Kedokteran dengan UU Rumah Sakit?",
                "category": "analisis",
                "difficulty": "hard"
            }
        ]
    
    def call_multi_api(self, question: str, embedding_model: str) -> Dict:
        """Call Multi-Step API"""
        url = f"{self.api_base}/api/chat"
        payload = {
            "query": question,
            "embedding_model": embedding_model,
            "use_parallel_execution": True,
            "previous_responses": []
        }
        
        start_time = time.time()
        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=60)
            response_time = int((time.time() - start_time) * 1000)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "answer": data.get("answer", ""),
                    "response_time_ms": response_time,
                    "num_documents": len(data.get("referenced_documents", [])),
                    "processing_steps": len(data.get("processing_steps", [])),
                    "answer_length": len(data.get("answer", "")),
                    "api_type": "multi",
                    "embedding_model": embedding_model
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "response_time_ms": response_time,
                    "api_type": "multi",
                    "embedding_model": embedding_model
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response_time_ms": int((time.time() - start_time) * 1000),
                "api_type": "multi",
                "embedding_model": embedding_model
            }
    
    def run_benchmark(self) -> List[Dict]:
        """Run benchmark untuk semua kombinasi"""
        questions = self.get_test_questions()
        results = []
        
        combinations = [
            ("multi", "small"),
            ("multi", "large")
        ]
        
        total_tests = len(questions) * len(combinations)
        current_test = 0
        
        print(f"🚀 Starting benchmark with {total_tests} tests")
        
        for question_data in questions:
            for api_type, embedding_model in combinations:
                current_test += 1
                print(f"Progress: {current_test}/{total_tests} - {api_type} + {embedding_model}")
                print(f"Question: {question_data['question'][:50]}...")
                
                if api_type == "multi":
                    result = self.call_multi_api(question_data["question"], embedding_model)
                
                # Add question metadata
                result.update({
                    "question": question_data["question"],
                    "category": question_data["category"],
                    "difficulty": question_data["difficulty"]
                })
                
                results.append(result)
                time.sleep(2)  # Delay antar request
        
        return results
    
    def analyze_results(self, results: List[Dict]) -> Dict:
        """Analisis hasil benchmark"""
        df = pd.DataFrame(results)
        
        summary = {}
        
        # Group by API type and embedding model
        for api_type in df["api_type"].unique():
            for embedding_model in df["embedding_model"].unique():
                key = f"{api_type}_{embedding_model}"
                subset = df[(df["api_type"] == api_type) & (df["embedding_model"] == embedding_model)]
                
                if len(subset) > 0:
                    successful = subset[subset["success"] == True]
                    
                    summary[key] = {
                        "total_tests": len(subset),
                        "successful_tests": len(successful),
                        "success_rate": len(successful) / len(subset) * 100,
                        "avg_response_time_ms": successful["response_time_ms"].mean() if len(successful) > 0 else 0,
                        "avg_answer_length": successful["answer_length"].mean() if len(successful) > 0 else 0,
                        "avg_num_documents": successful["num_documents"].mean() if len(successful) > 0 else 0,
                        "avg_processing_steps": successful["processing_steps"].mean() if len(successful) > 0 else 0
                    }
        
        return {
            "summary": summary,
            "raw_data": results,
            "timestamp": datetime.now().isoformat()
        }
    
    def save_results(self, analysis: Dict) -> str:
        """Save results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save complete results
        with open(self.output_dir / f"benchmark_results_{timestamp}.json", "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Save summary table
        df = pd.DataFrame(analysis["raw_data"])
        df.to_csv(self.output_dir / f"benchmark_data_{timestamp}.csv", index=False)
        
        print(f"✅ Results saved to: {self.output_dir}")
        return timestamp
    
    def generate_report(self, analysis: Dict, timestamp: str):
        """Generate readable report"""
        report = f"""
# RAG Benchmark Report
Generated: {analysis['timestamp']}

## Summary Results

"""
        
        for combo, data in analysis["summary"].items():
            api_type, embedding = combo.split("_")
            report += f"""
### {api_type.upper()} API + {embedding.upper()} Embedding

- **Success Rate**: {data['success_rate']:.1f}%
- **Average Response Time**: {data['avg_response_time_ms']:.0f} ms
- **Average Answer Length**: {data['avg_answer_length']:.0f} characters
- **Average Documents Retrieved**: {data['avg_num_documents']:.1f}
- **Average Processing Steps**: {data['avg_processing_steps']:.1f}

"""
        
        # Comparison
        if len(analysis["summary"]) > 1:
            report += "\n## Comparison\n\n"
            
            # Find fastest
            fastest = min(analysis["summary"].items(), 
                         key=lambda x: x[1]['avg_response_time_ms'])
            report += f"**Fastest**: {fastest[0]} ({fastest[1]['avg_response_time_ms']:.0f} ms)\n\n"
            
            # Find most comprehensive
            most_docs = max(analysis["summary"].items(), 
                           key=lambda x: x[1]['avg_num_documents'])
            report += f"**Most Documents**: {most_docs[0]} ({most_docs[1]['avg_num_documents']:.1f} docs)\n\n"
        
        report += "\n## Test Questions\n\n"
        for i, result in enumerate(analysis["raw_data"][:5], 1):
            report += f"{i}. **{result['category'].title()}** ({result['difficulty']}): {result['question']}\n"
        
        # Save report
        with open(self.output_dir / f"benchmark_report_{timestamp}.md", "w") as f:
            f.write(report)
        
        print(f"📄 Report generated: benchmark_report_{timestamp}.md")

def main():
    """Main execution"""
    benchmark = SimpleBenchmark()
    
    # Run benchmark
    results = benchmark.run_benchmark()
    
    # Analyze results
    analysis = benchmark.analyze_results(results)
    
    # Save results
    timestamp = benchmark.save_results(analysis)
    
    # Generate report
    benchmark.generate_report(analysis, timestamp)
    
    print("✅ Benchmark completed!")
    print("\n📊 Summary:")
    for combo, data in analysis["summary"].items():
        print(f"{combo}: {data['success_rate']:.1f}% success, {data['avg_response_time_ms']:.0f}ms avg")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
🚀 SIMPLE PARALLEL EXECUTION DEMO
Perbandingan sederhana antara Standard vs Parallel execution
"""

import requests
import time
import json
from datetime import datetime

# Configuration
API_BASE = "http://localhost:8080"
API_KEY = "your_secure_api_key_here"
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

TEST_QUERY = "Apa sanksi hukum yang dapat dikenakan kepada seseorang yang melakukan praktik kedokteran tanpa izin (tanpa memiliki Surat Tanda Registrasi atau Surat Izin Praktik) berdasarkan Undang-Undang Praktik Kedokteran?"


def save_results(standard_result, parallel_result, comparison):
    """Simpan hasil test ke file JSON"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_results_{timestamp}.json"

    results = {
        "timestamp": timestamp,
        "query": TEST_QUERY,
        "standard_execution": {
            "response_time": standard_result["time"],
            "processing_time_ms": standard_result.get("processing_time_ms", 0),
            "documents": standard_result.get("documents", 0),
            "parallel_used": standard_result.get("parallel_used", False),
            "cached": standard_result.get("cached", False),
            "answer": standard_result.get("answer", ""),
            "agent_steps": standard_result.get("agent_steps", []),
            "referenced_documents": standard_result.get("referenced_documents", []),
        },
        "parallel_execution": {
            "response_time": parallel_result["time"],
            "processing_time_ms": parallel_result.get("processing_time_ms", 0),
            "documents": parallel_result.get("documents", 0),
            "parallel_used": parallel_result.get("parallel_used", False),
            "cached": parallel_result.get("cached", False),
            "answer": parallel_result.get("answer", ""),
            "agent_steps": parallel_result.get("agent_steps", []),
            "referenced_documents": parallel_result.get("referenced_documents", []),
        },
        "comparison": {
            "improvement_percentage": comparison["improvement"],
            "time_saved": comparison["time_saved"],
            "speedup_factor": comparison["speedup"],
            "answer_comparison": {
                "same_answer": standard_result.get("answer", "")
                == parallel_result.get("answer", ""),
                "standard_length": len(standard_result.get("answer", "")),
                "parallel_length": len(parallel_result.get("answer", "")),
                "length_difference": abs(
                    len(standard_result.get("answer", ""))
                    - len(parallel_result.get("answer", ""))
                ),
            },
        },
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return filename


def test_execution(use_parallel=True):
    """Test standard atau parallel execution"""
    mode = "PARALLEL" if use_parallel else "STANDARD"
    print(f"\n{'='*60}")
    print(f"🧪 Testing {mode} Execution")
    print(f"{'='*60}")

    payload = {
        "query": TEST_QUERY,
        "embedding_model": "large",
        "use_parallel_execution": use_parallel,
    }

    print(f"📝 Payload: use_parallel_execution = {use_parallel}")
    print(f"⏰ Starting at: {time.strftime('%H:%M:%S')}")

    start_time = time.time()

    try:
        response = requests.post(f"{API_BASE}/api/chat", headers=HEADERS, json=payload)
        end_time = time.time()

        if response.status_code == 200:
            data = response.json()
            model_info = data.get("model_info", {})

            print(f"\n✅ {mode} RESULTS:")
            print(f"⏱️  Response time: {(end_time - start_time):.3f}s")
            print(f"⚙️  Processing time: {data.get('processing_time_ms')}ms")
            print(f"📄 Documents: {len(data.get('referenced_documents', []))}")
            print(f"🎯 Parallel used: {model_info.get('parallel_execution', False)}")
            print(f"💾 Cached: {model_info.get('cached', False)}")

            if model_info.get("parallel_execution"):
                print(
                    f"⚡ Performance boost: {model_info.get('performance_boost', 'N/A')}"
                )
                features = model_info.get("parallel_features", [])
                if features:
                    print(f"🔧 Features: {', '.join(features)}")

            return {
                "success": True,
                "time": end_time - start_time,
                "processing_time_ms": data.get("processing_time_ms", 0),
                "documents": len(data.get("referenced_documents", [])),
                "parallel_used": model_info.get("parallel_execution", False),
                "cached": model_info.get("cached", False),
                "answer": data.get("answer", ""),
                "agent_steps": data.get("agent_steps", []),
                "referenced_documents": data.get("referenced_documents", []),
            }
        else:
            print(f"❌ Error: {response.status_code}")
            return {"success": False}

    except Exception as e:
        print(f"❌ Exception: {e}")
        return {"success": False}


def main():
    print("🚀 PARALLEL EXECUTION DEMO")
    print("Testing comparison between Standard vs Parallel execution")
    print(f"Query: {TEST_QUERY}")

    # Check API
    try:
        health = requests.get(f"{API_BASE}/health")
        if health.status_code != 200:
            print("❌ API not running. Start dengan: python multi_api.py")
            return
        print("✅ API connected")
    except:
        print("❌ Cannot connect to API")
        return

    # Clear cache before standard test
    try:
        requests.delete(f"{API_BASE}/api/cache/clear", headers=HEADERS)
        print("✅ Cache cleared before standard test")
        time.sleep(2)  # Wait for cache clear
    except:
        print("⚠️ Could not clear cache")

    # Test standard execution
    standard_result = test_execution(use_parallel=False)

    # Clear cache again before parallel test
    try:
        requests.delete(f"{API_BASE}/api/cache/clear", headers=HEADERS)
        print("✅ Cache cleared before parallel test")
        time.sleep(2)  # Wait for cache clear
    except:
        print("⚠️ Could not clear cache")

    # Test parallel execution
    parallel_result = test_execution(use_parallel=True)

    # Compare results
    if standard_result.get("success") and parallel_result.get("success"):
        print(f"\n{'='*60}")
        print("📊 COMPARISON")
        print(f"{'='*60}")

        std_time = standard_result["time"]
        par_time = parallel_result["time"]

        if std_time > 0:
            improvement = ((std_time - par_time) / std_time) * 100
            speedup = std_time / par_time if par_time > 0 else float("inf")

            print(f"🔄 Standard: {std_time:.3f}s")
            print(f"🚀 Parallel: {par_time:.3f}s")
            print(f"📈 Improvement: {improvement:.1f}%")
            print(f"⚡ Speedup factor: {speedup:.2f}x")
            print(f"⏰ Time saved: {(std_time - par_time):.3f}s")

            # Compare answers
            std_answer = standard_result.get("answer", "")
            par_answer = parallel_result.get("answer", "")
            print(f"\n📝 ANSWER COMPARISON:")
            print(f"Standard length: {len(std_answer)} chars")
            print(f"Parallel length: {len(par_answer)} chars")
            print(f"Same answer: {'✅ Yes' if std_answer == par_answer else '❌ No'}")

            comparison = {
                "improvement": improvement,
                "time_saved": std_time - par_time,
                "speedup": speedup,
            }

            # Save results to file
            filename = save_results(standard_result, parallel_result, comparison)
            print(f"\n💾 Results saved to: {filename}")

            if improvement >= 25:
                print("✅ EXCELLENT performance boost!")
            elif improvement >= 10:
                print("⚠️ GOOD improvement")
            else:
                print("ℹ️ Minimal improvement (may be cached)")


if __name__ == "__main__":
    main()

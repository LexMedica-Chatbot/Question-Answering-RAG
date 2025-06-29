import requests
import json
import time


def test_rag_endpoint(
    url: str, query: str, endpoint_name: str, api_key: str, is_multi_agent: bool = False
):
    """Test RAG endpoint with the correct format"""
    print(f"\n{'='*60}")
    print(f"Testing {endpoint_name}")
    print(f"URL: {url}")
    print(f"{'='*60}")

    # Prepare headers
    headers = {"Content-Type": "application/json", "X-API-Key": api_key}

    # Prepare payload based on endpoint type
    if is_multi_agent:
        payload = {
            "query": query,
            "embedding_model": "large",
            "previous_responses": [],
            "use_parallel_execution": True,
        }
    else:
        payload = {"query": query, "embedding_model": "large"}

    try:
        print(f"Sending request...")
        print(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")

        response = requests.post(url, json=payload, headers=headers, timeout=60)

        print(f"\nStatus Code: {response.status_code}")

        if response.status_code == 200:
            response_data = response.json()
            print(f"\nResponse JSON:")
            print(json.dumps(response_data, indent=2, ensure_ascii=False))

            # Analyze response structure
            print(f"\n{'='*40}")
            print("RESPONSE ANALYSIS:")
            print(f"{'='*40}")
            print(f"Response type: {type(response_data)}")
            if isinstance(response_data, dict):
                print(f"Available keys: {list(response_data.keys())}")

                # Analyze specific fields
                if "answer" in response_data:
                    print(f"✅ Answer found: {response_data['answer'][:200]}...")

                if is_multi_agent and "processing_steps" in response_data:
                    steps = response_data["processing_steps"]
                    print(
                        f"✅ Processing steps found: {len(steps) if isinstance(steps, list) else 'N/A'} steps"
                    )
                    if isinstance(steps, list) and steps:
                        print(
                            f"  First step tool: {steps[0].get('tool', 'Unknown') if isinstance(steps[0], dict) else 'Unknown'}"
                        )

                if not is_multi_agent and "referenced_documents" in response_data:
                    docs = response_data["referenced_documents"]
                    print(
                        f"✅ Referenced documents found: {len(docs) if isinstance(docs, list) else 'N/A'} documents"
                    )

                if "processing_time_ms" in response_data:
                    print(
                        f"⏱️  Processing time: {response_data['processing_time_ms']} ms"
                    )

            return response_data
        else:
            print(f"❌ Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return None

    except requests.exceptions.Timeout:
        print("❌ Error: Request timeout")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None


def main():
    """Test both RAG endpoints with correct format"""
    print("TESTING RAG ENDPOINTS - UPDATED VERSION")
    print("=" * 80)

    # Get API key
    api_key = input("Enter API key for RAG systems (X-API-Key): ").strip()
    if not api_key:
        print("❌ Error: API key is required.")
        return

    # Test query from the validation file
    test_query = "Dalam situasi apa saja tenaga kesehatan atau tenaga medis dapat melakukan tindakan aborsi yang dibenarkan oleh hukum di Indonesia, dan apa saja syarat yang harus dipenuhi?"

    # Endpoint URLs with correct paths
    single_rag_url = (
        "https://lexmedica-chatbot-176465812210.asia-southeast2.run.app/api/chat"
    )
    multi_agent_url = "https://lexmedica-chatbot-multiagent-176465812210.asia-southeast2.run.app/api/chat"

    print(f"\nTest Query: {test_query[:100]}...")

    # Test Single RAG
    single_response = test_rag_endpoint(
        single_rag_url, test_query, "Single RAG", api_key, is_multi_agent=False
    )
    time.sleep(2)  # Rate limiting

    # Test Multi-Agent RAG
    multi_response = test_rag_endpoint(
        multi_agent_url, test_query, "Multi-Agent RAG", api_key, is_multi_agent=True
    )

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Single RAG: {'✅ Success' if single_response else '❌ Failed'}")
    print(f"Multi-Agent RAG: {'✅ Success' if multi_response else '❌ Failed'}")

    if single_response and multi_response:
        print("\n🎉 Both endpoints are working correctly!")
        print("\n📊 Ready for Ragas evaluation!")
        print("\nTo run the full evaluation:")
        print("python benchmarks/ragas_evaluation.py")

        # Show format differences
        print(f"\n{'='*60}")
        print("FORMAT COMPARISON")
        print(f"{'='*60}")

        if isinstance(single_response, dict) and isinstance(multi_response, dict):
            print("\n📋 Single RAG Response Structure:")
            for key in single_response.keys():
                print(f"  - {key}")

            print("\n📋 Multi-Agent RAG Response Structure:")
            for key in multi_response.keys():
                print(f"  - {key}")
    else:
        print("\n❌ Some endpoints failed. Please check:")
        print("1. API key is correct")
        print("2. Endpoints are accessible")
        print("3. Network connectivity")


if __name__ == "__main__":
    main()

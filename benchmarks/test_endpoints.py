import requests
import json
import time


def test_rag_endpoint(url: str, query: str, endpoint_name: str):
    """Test RAG endpoint with a sample query"""
    print(f"\n{'='*60}")
    print(f"Testing {endpoint_name}")
    print(f"URL: {url}")
    print(f"{'='*60}")

    payload = {
        "query": query,
        "embedding_model": "large",
        "previous_responses": [],
        "use_parallel_execution": "false",
    }

    headers = {"Content-Type": "application/json"}

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

                # Try to identify answer and context fields
                for key in response_data.keys():
                    print(f"  - {key}: {type(response_data[key])}")
                    if (
                        isinstance(response_data[key], str)
                        and len(response_data[key]) > 100
                    ):
                        print(f"    Content preview: {response_data[key][:200]}...")
                    elif isinstance(response_data[key], list):
                        print(f"    List length: {len(response_data[key])}")
                        if response_data[key]:
                            print(f"    First item type: {type(response_data[key][0])}")

            return response_data
        else:
            print(f"Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return None

    except requests.exceptions.Timeout:
        print("Error: Request timeout")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def main():
    """Test both RAG endpoints"""
    # Test query from the validation file
    test_query = "Dalam situasi apa saja tenaga kesehatan atau tenaga medis dapat melakukan tindakan aborsi yang dibenarkan oleh hukum di Indonesia, dan apa saja syarat yang harus dipenuhi?"

    # Endpoint URLs (same exact URLs)
    single_rag_url = "https://lexmedica-chatbot-176465812210.asia-southeast2.run.app"
    multi_agent_url = (
        "https://lexmedica-chatbot-multiagent-176465812210.asia-southeast2.run.app"
    )

    print("TESTING RAG ENDPOINTS")
    print("Query:", test_query[:100] + "...")

    # Test Single RAG
    single_response = test_rag_endpoint(single_rag_url, test_query, "Single RAG")
    time.sleep(2)  # Rate limiting

    # Test Multi-Agent RAG
    multi_response = test_rag_endpoint(multi_agent_url, test_query, "Multi-Agent RAG")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Single RAG: {'✅ Success' if single_response else '❌ Failed'}")
    print(f"Multi-Agent RAG: {'✅ Success' if multi_response else '❌ Failed'}")

    if single_response and multi_response:
        print(
            "\n✅ Both endpoints are working! You can proceed with the full evaluation."
        )
        print("\nTo run the full Ragas evaluation, use:")
        print("python benchmarks/ragas_evaluation.py")
    else:
        print("\n❌ Some endpoints failed. Please check the error messages above.")


if __name__ == "__main__":
    main()

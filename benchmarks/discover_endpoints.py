import requests
import json


def test_endpoint_options(base_url: str, endpoint_name: str):
    """Test various common endpoint paths"""
    common_endpoints = [
        "/",
        "/docs",
        "/openapi.json",
        "/query",
        "/ask",
        "/chat",
        "/generate",
        "/health",
        "/status",
        "/api/query",
        "/api/ask",
        "/api/chat",
        "/v1/query",
        "/v1/ask",
        "/v1/chat",
    ]

    print(f"\n{'='*60}")
    print(f"Discovering endpoints for {endpoint_name}")
    print(f"Base URL: {base_url}")
    print(f"{'='*60}")

    working_endpoints = []

    for endpoint in common_endpoints:
        url = base_url.rstrip("/") + endpoint
        try:
            # Try GET first
            response = requests.get(url, timeout=10)
            if response.status_code in [
                200,
                405,
            ]:  # 405 means method not allowed but endpoint exists
                working_endpoints.append((endpoint, "GET", response.status_code))
                print(f"✅ {endpoint} (GET): {response.status_code}")

                if response.status_code == 200:
                    try:
                        data = response.json()
                        print(f"   Response preview: {str(data)[:200]}...")
                    except:
                        print(f"   Response preview: {response.text[:200]}...")
            else:
                print(f"❌ {endpoint} (GET): {response.status_code}")

            # Try POST for query endpoints
            if "query" in endpoint or "ask" in endpoint or "chat" in endpoint:
                try:
                    test_payload = {
                        "query": "test",
                        "question": "test",
                        "message": "test",
                    }
                    post_response = requests.post(url, json=test_payload, timeout=10)
                    if post_response.status_code not in [404, 501]:
                        working_endpoints.append(
                            (endpoint, "POST", post_response.status_code)
                        )
                        print(f"✅ {endpoint} (POST): {post_response.status_code}")

                        if post_response.status_code == 200:
                            try:
                                data = post_response.json()
                                print(f"   POST Response preview: {str(data)[:200]}...")
                            except:
                                print(
                                    f"   POST Response preview: {post_response.text[:200]}..."
                                )
                except:
                    pass

        except requests.exceptions.Timeout:
            print(f"⏰ {endpoint}: Timeout")
        except requests.exceptions.RequestException as e:
            print(f"❌ {endpoint}: {str(e)[:50]}...")
        except Exception as e:
            print(f"❌ {endpoint}: {str(e)[:50]}...")

    print(f"\nSummary for {endpoint_name}:")
    print(f"Working endpoints: {len(working_endpoints)}")
    for endpoint, method, status in working_endpoints:
        print(f"  - {method} {endpoint}: {status}")

    return working_endpoints


def check_api_documentation(base_url: str, endpoint_name: str):
    """Check for API documentation"""
    print(f"\n{'='*60}")
    print(f"Checking API Documentation for {endpoint_name}")
    print(f"{'='*60}")

    doc_endpoints = ["/docs", "/swagger", "/openapi.json", "/api/docs"]

    for doc_endpoint in doc_endpoints:
        try:
            url = base_url.rstrip("/") + doc_endpoint
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"✅ Documentation found at: {url}")

                if doc_endpoint == "/openapi.json":
                    try:
                        openapi_spec = response.json()
                        if "paths" in openapi_spec:
                            print("Available endpoints from OpenAPI spec:")
                            for path, methods in openapi_spec["paths"].items():
                                for method in methods.keys():
                                    print(f"  - {method.upper()} {path}")
                    except:
                        print("Could not parse OpenAPI spec")

                return True
        except:
            continue

    print("❌ No documentation endpoints found")
    return False


def main():
    """Discover endpoints for both RAG APIs"""
    single_rag_url = "https://lexmedica-chatbot-176465812210.asia-southeast2.run.app"
    multi_agent_url = (
        "https://lexmedica-chatbot-multiagent-176465812210.asia-southeast2.run.app"
    )

    print("DISCOVERING RAG API ENDPOINTS")
    print("=" * 80)

    # Test Single RAG
    single_endpoints = test_endpoint_options(single_rag_url, "Single RAG")
    check_api_documentation(single_rag_url, "Single RAG")

    # Test Multi-Agent RAG
    multi_endpoints = test_endpoint_options(multi_agent_url, "Multi-Agent RAG")
    check_api_documentation(multi_agent_url, "Multi-Agent RAG")

    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")

    if single_endpoints:
        print(f"✅ Single RAG API is accessible")
        print(f"   Working endpoints: {[ep[0] for ep in single_endpoints]}")
    else:
        print(f"❌ Single RAG API: No working endpoints found")

    if multi_endpoints:
        print(f"✅ Multi-Agent RAG API is accessible")
        print(f"   Working endpoints: {[ep[0] for ep in multi_endpoints]}")
    else:
        print(f"❌ Multi-Agent RAG API: No working endpoints found")

    return single_endpoints, multi_endpoints


if __name__ == "__main__":
    main()

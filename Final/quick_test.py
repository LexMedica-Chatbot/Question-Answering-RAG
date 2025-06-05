import requests
import json

# Test single request
url = "http://localhost:8080/api/chat"
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "your_secure_api_key_here"
}
payload = {"query": "Apa itu BPJS?"}

print("🧪 Testing single request...")
response = requests.post(url, headers=headers, json=payload)

if response.status_code == 200:
    result = response.json()
    print(f"✅ Success! Response length: {len(result.get('answer', ''))}")
    print(f"Model info: {result.get('model_info', {})}")
else:
    print(f"❌ Error: {response.status_code}")
    print(response.text) 
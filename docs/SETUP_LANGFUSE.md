# 🎯 LangFuse Setup - RAG Cost & Token Tracking

## 🚀 **5-Minute Setup untuk RAG Observability**

### **Kenapa LangFuse?**

-   ✅ **Langsung track cost** setiap request
-   ✅ **Token usage** real-time
-   ✅ **RAG performance** metrics
-   ✅ **Beautiful dashboard** tanpa setup ribet
-   ✅ **Free tier** 50K traces/month

---

## 📋 **Setup Steps**

### **1. Daftar LangFuse (2 menit)**

```bash
# Buka browser
https://cloud.langfuse.com

# Klik "Sign Up" → daftar dengan email
# Verifikasi email → login
```

### **2. Buat Project (1 menit)**

```
1. Klik "New Project"
2. Nama project: "RAG-Monitoring"
3. Klik "Create"
```

### **3. Dapatkan API Keys (1 menit)**

```
1. Di dashboard project → klik "Settings"
2. Tab "API Keys"
3. Copy:
   - Public Key: pk-lf-...
   - Secret Key: sk-lf-...
```

### **4. Set Environment Variables (1 menit)**

```bash
# Tambahkan ke file .env Anda:
LANGFUSE_PUBLIC_KEY=pk-lf-your-key-here
LANGFUSE_SECRET_KEY=sk-lf-your-secret-here
```

### **5. Restart Application**

```bash
docker compose up --build
```

---

## 🎯 **Yang Akan Anda Dapatkan**

### **📊 Cost Tracking**

-   **Cost per request**: $0.00234 USD
-   **Daily cost**: $2.45 USD
-   **Model breakdown**: gpt-4 vs gpt-3.5 costs
-   **Cost trends** over time

### **🔢 Token Usage**

-   **Input tokens**: 1,234 tokens
-   **Output tokens**: 567 tokens
-   **Embedding tokens**: 89 tokens
-   **Token efficiency** metrics

### **📈 RAG Performance**

-   **Document retrieval time**: 250ms
-   **LLM response time**: 2.3s
-   **End-to-end latency**: 3.1s
-   **Documents retrieved**: 4 docs/request

### **📋 Session Tracking**

-   **User queries** → **AI responses**
-   **Conversation flows**
-   **Error tracking**
-   **Performance bottlenecks**

---

## 🖥️ **How to Use**

### **1. Check Status**

```bash
curl http://localhost:8080/api/observability
```

### **2. Make Some Requests**

```bash
# Test beberapa questions untuk generate data
curl -X POST "http://localhost:8080/api/chat" \
  -H "X-API-Key: your_secure_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{"query":"Apa kewajiban dokter?","embedding_model":"large"}'
```

### **3. View Dashboard**

```bash
# Open LangFuse dashboard
https://cloud.langfuse.com

# Navigate to your project
# Explore Traces, Sessions, Users tabs
```

---

## 📊 **Dashboard Screenshots**

### **Traces View**

```
🔍 rag_chat (3.2s) - $0.0023
├── 📄 document_retrieval (0.25s) - 4 docs
├── 🤖 llm_call (2.8s) - 1,234→567 tokens
└── ✅ rag_session - Success
```

### **Cost Analytics**

```
📈 Total Cost: $12.45 (last 7 days)
┌─────────────────────────────────┐
│ Daily Cost Trend                │
│ ▄▄▆▆██▆▄ $1.2-$2.8/day        │
└─────────────────────────────────┘

💰 Top Models:
• gpt-4.1-mini: $8.23 (67%)
• text-embedding-3-large: $4.22 (33%)
```

### **Performance Insights**

```
⚡ Avg Response Time: 3.1s
🎯 Success Rate: 98.5%
📄 Avg Documents: 4.2/request
🔄 Cache Hit Rate: 23%
```

---

## 🛠️ **Advanced Features**

### **Custom Metadata Tracking**

```python
# Sudah di-setup di code Anda:
- embedding_model: "large" vs "small"
- num_documents: berapa doc retrieved
- processing_time_ms: response time
- estimated_cost_usd: biaya per request
- api_type: "simple" vs "multi"
```

### **Real-time Monitoring**

-   **Live dashboard** updates
-   **Alert setup** untuk high costs
-   **Performance regression** detection
-   **Usage pattern** analysis

### **Team Collaboration**

-   **Share dashboards** dengan team
-   **Export reports** untuk research
-   **API access** untuk custom analytics

---

## 🔧 **Troubleshooting**

### **LangFuse Not Working?**

```bash
# Check environment variables
echo $LANGFUSE_PUBLIC_KEY
echo $LANGFUSE_SECRET_KEY

# Check logs
docker compose logs lexmedica-chatbot | grep -i langfuse
```

### **No Data in Dashboard?**

```bash
# Make sure you're making requests
curl http://localhost:8080/api/observability

# Check if traces are being sent
# Look for "✅ LangFuse observability enabled" in logs
```

### **High Costs?**

```bash
# Check token usage in LangFuse dashboard
# Consider switching to gpt-3.5-turbo for dev
# Use "small" embedding model for testing
```

---

## 🎉 **Example Insights**

### **Daily Report**

```
📊 RAG Performance Summary (Today)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💰 Cost: $2.34 USD
🔢 Requests: 156 total
⚡ Avg Response: 3.1s
📄 Documents: 4.2 avg/request
✅ Success Rate: 98.7%

🔥 Top Questions:
1. "Kewajiban dokter" - 23 times
2. "Informed consent" - 18 times
3. "Malpraktik" - 15 times

📈 Performance:
• Fastest: 1.2s
• Slowest: 8.9s
• P95: 6.1s
```

---

## 🎯 **Benefits untuk Research**

### **Experiment Tracking**

-   Compare **different models**
-   A/B test **prompt changes**
-   Track **performance over time**

### **Cost Optimization**

-   Find **expensive queries**
-   Optimize **token usage**
-   Compare **embedding models**

### **Academic Insights**

-   **Usage patterns** analysis
-   **Performance benchmarks**
-   **Reproducible experiments**

---

**🚀 Ready to go!** Dengan setup ini, Anda akan punya **complete visibility** ke RAG system dengan fokus pada **cost, performance, dan insights** yang Anda butuhkan!

**Next Steps:**

1. Setup LangFuse (5 min)
2. Make some test requests
3. Explore dashboard
4. Share insights dengan supervisor! 📊

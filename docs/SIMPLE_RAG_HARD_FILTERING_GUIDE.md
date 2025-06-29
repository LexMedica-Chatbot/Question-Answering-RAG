# 🛡️ Simple RAG Hard Filtering & All Retrieved Documents Guide

## 🎯 Overview

Simple RAG sekarang dilengkapi dengan **hard filtering** dan field **`all_retrieved_documents`** yang sama seperti Multi-Agent RAG untuk:

1. **Clean Answer Generation**: Hanya menggunakan dokumen aktif (bukan dicabut)
2. **Disharmony Detection**: Menyimpan semua dokumen untuk analisis subsistem lain

## 📊 Response Structure

```json
{
  "answer": "Jawaban berdasarkan dokumen aktif saja",
  "model_info": {...},
  "referenced_documents": [
    // Hanya dokumen aktif yang digunakan dalam jawaban
  ],
  "all_retrieved_documents": [
    // Semua dokumen termasuk yang dicabut
  ],
  "processing_time_ms": 1500
}
```

## 🔧 Implementation Details

### 1. Hard Filtering Process

```python
# Setelah vector store retrieval
docs = retriever.invoke(request.query)

# Store all documents before filtering
all_retrieved_documents = extract_document_info(docs)

# Hard filtering: eliminate revoked documents
active_docs = []
eliminated_count = 0

for doc in docs:
    metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
    status = metadata.get("status", "berlaku").lower()

    if status == "dicabut":
        eliminated_count += 1
    else:
        active_docs.append(doc)

# Log elimination results
if eliminated_count > 0:
    print(f"[SIMPLE RAG] 🚫 Eliminated {eliminated_count} revoked documents")

# Use filtered docs for generation
docs = active_docs
```

### 2. Answer Generation with Filtered Documents

```python
# Generate answer using filtered documents only
formatted_context = format_docs(docs)  # Hanya dokumen aktif
prompt_input = {
    "context": formatted_context,
    "question": request.query
}

answer = (custom_prompt | llm | StrOutputParser()).invoke(prompt_input)
```

### 3. Response Preparation

```python
# referenced_documents: Dari dokumen aktif saja
referenced_documents = extract_document_info(docs)

# all_retrieved_documents: Dari semua dokumen sebelum filtering
all_retrieved_documents = extract_document_info(original_docs)

return ChatResponse(
    answer=answer,
    referenced_documents=referenced_documents,  # Clean docs
    all_retrieved_documents=all_retrieved_documents,  # All docs
    ...
)
```

## 🧪 Test Results

### Query: "Apa tugas menteri kesehatan?"

**Referenced Documents (Clean):**

-   PP No. 32/1996 (Status: berlaku)
-   Perpres No. 161/2024 (Status: berlaku)

**All Retrieved Documents (Complete):**

-   PP No. 32/1996 (Status: berlaku)
-   **UU No. 36/2009 (Status: dicabut)** ← Eliminated from answer
-   Perpres No. 161/2024 (Status: berlaku)

**Hard Filtering Status:** ✅ Working

-   1 revoked document eliminated from answer generation
-   3 active documents used for clean answer

## 📈 Benefits

### For Answer Generation

-   ✅ **Clean answers** using only valid regulations
-   ✅ **No outdated information** from revoked laws
-   ✅ **Legal compliance** with current regulations

### For Disharmony Detection

-   ✅ **Complete document tracking** including revoked ones
-   ✅ **Gap analysis** - identify what was revoked but not replaced
-   ✅ **Regulatory evolution** tracking across time

## 🔍 Usage Example

```python
import requests

response = requests.post(
    'http://localhost:8001/api/chat',
    headers={'X-API-Key': 'your-api-key'},
    json={
        'query': 'Apa tugas menteri kesehatan?',
        'embedding_model': 'large'
    }
)

data = response.json()

# Clean answer from active documents only
print(f"Answer: {data['answer']}")

# Documents used in answer (active only)
print(f"Referenced: {len(data['referenced_documents'])}")

# All documents including revoked (for analysis)
print(f"All retrieved: {len(data['all_retrieved_documents'])}")

# Detect revoked documents for disharmony analysis
revoked_docs = [
    doc for doc in data['all_retrieved_documents']
    if 'dicabut' in doc.get('source', '').lower()
]
print(f"Revoked documents found: {len(revoked_docs)}")
```

## 🚀 Key Features

1. **Automatic Hard Filtering**: Dokumen dicabut otomatis dieliminasi
2. **Dual Document Fields**: Clean + complete untuk berbagai kebutuhan
3. **Logging**: Track elimination results untuk monitoring
4. **Backward Compatibility**: Tidak mengubah existing API contract
5. **Performance**: Efficient filtering dengan minimal overhead

## 🛠️ Configuration

Tidak ada konfigurasi tambahan diperlukan. Hard filtering berjalan otomatis berdasarkan `metadata.status` field dari dokumen.

**Status Values:**

-   `"berlaku"` → Digunakan dalam jawaban
-   `"dicabut"` → Dieliminasi dari jawaban, disimpan untuk analisis

## 📝 Notes

-   Simple RAG tidak menggunakan cache (berbeda dengan Multi-Agent)
-   Logging elimination hanya muncul jika ada dokumen yang dieliminasi
-   Field `all_retrieved_documents` selalu ada dalam response (bisa empty array)
-   Deduplication otomatis dilakukan untuk kedua field

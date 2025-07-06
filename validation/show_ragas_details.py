"""
🔍 RAGAS ACTUAL PROMPTS & IMPLEMENTATION
Berdasarkan implementasi di benchmarks/ragas_evaluation_deploy.py
"""


def show_actual_ragas_implementation():
    """
    Menunjukkan implementasi actual RAGAS di project
    """

    print("🔍 RAGAS ACTUAL IMPLEMENTATION IN PROJECT")
    print("=" * 60)

    print(
        """
📁 File: benchmarks/ragas_evaluation_deploy.py
📦 Method: evaluate_responses(self, dataset_dict: List[Dict])

🔧 Setup LLM Evaluator:
"""
    )

    setup_code = """
# Setup evaluator LLM (di __init__)
if self.openai_api_key:
    llm = ChatOpenAI(model="gpt-4.1", api_key=self.openai_api_key)
    self.evaluator_llm = LangchainLLMWrapper(llm)
"""
    print(setup_code)

    print("\n🔧 Data Format yang Dievaluasi:")

    data_format = """
dataset_dict = [
    {
        "user_input": "Pertanyaan yang diajukan",
        "retrieved_contexts": ["context1", "context2", ...],  # dari RAG system
        "response": "Jawaban yang dihasilkan sistem",
        "reference": "Ground truth jawaban yang benar"
    },
    ...
]
"""
    print(data_format)

    print("\n🔧 Metrics yang Digunakan:")

    metrics_code = """
# ACTUAL IMPLEMENTATION di benchmarks/ragas_evaluation_deploy.py:

from ragas import evaluate  # ✅ INI YANG DIPAKAI!
from ragas.metrics import (
    context_recall,
    faithfulness, 
    answer_relevancy,
    FactualCorrectness,
)

# Define metrics (baris 266-271)
metrics = [
    context_recall,                    # Skor 0-1
    faithfulness,                     # Skor 0-1  
    FactualCorrectness(mode="f1"),    # F1-score 0-1
    answer_relevancy,                 # Skor 0-1
]
"""
    print(metrics_code)

    print("\n🔧 Evaluasi Process:")

    eval_code = """
# ACTUAL IMPLEMENTATION di method evaluate_responses (baris 262-280):

def evaluate_responses(self, dataset_dict: List[Dict]) -> Dict[str, float]:
    # Convert to Ragas dataset format
    dataset = Dataset.from_list(dataset_dict)
    
    # Run evaluation dengan LLM ✅ INI YANG ACTUAL DIPAKAI!
    result = evaluate(dataset=dataset, metrics=metrics, llm=self.evaluator_llm)
    
    return dict(result)

# Result format yang dikembalikan:
{
    "context_recall": 0.85,
    "faithfulness": 0.92,
    "factual_correctness(mode=f1)": 0.78, 
    "answer_relevancy": 0.88
}
"""
    print(eval_code)


def show_ragas_prompts_detail():
    """
    Menunjukkan prompt detail RAGAS untuk setiap metrik
    """

    print("\n\n🤖 RAGAS INTERNAL PROMPTS (berdasarkan source code)")
    print("=" * 60)

    print("\n1️⃣ CONTEXT RECALL PROMPT")
    print("-" * 40)

    context_recall_prompt = """
PROMPT TEMPLATE:
Given a context, and an answer, analyze each sentence in the answer and classify if the sentence can be attributed to the given context or not. 

context: {context}
answer: {ground_truth}

Analyze each sentence in the answer:
1. [sentence 1] - Yes/No
2. [sentence 2] - Yes/No
...

CALCULATION:
context_recall = (sentences_attributed_to_context) / (total_sentences_in_ground_truth)
"""
    print(context_recall_prompt)

    print("\n2️⃣ FAITHFULNESS PROMPT")
    print("-" * 40)

    faithfulness_prompt = """
PROMPT TEMPLATE:
Given a question, an answer and contexts, analyze the answer and classify each sentence in the answer as one that can be directly inferred from the contexts or not.

question: {question}
contexts: {contexts}
answer: {answer}

For each sentence in the answer:
1. [sentence 1] - 1 (dapat diinfer) / 0 (tidak dapat diinfer)
2. [sentence 2] - 1 / 0
...

CALCULATION:
faithfulness = (sentences_inferred_from_context) / (total_sentences_in_answer)
"""
    print(faithfulness_prompt)

    print("\n3️⃣ ANSWER RELEVANCY PROMPT")
    print("-" * 40)

    answer_relevancy_prompt = """
PROMPT TEMPLATE:
Generate questions for the given answer. The questions should be diverse and comprehensive.

answer: {answer}

Generate 3-5 questions that this answer could respond to:
1. [question 1]
2. [question 2]
...

CALCULATION:
1. Generate multiple questions dari answer
2. Hitung cosine similarity antara generated questions dan original question
3. answer_relevancy = mean(similarities)
"""
    print(answer_relevancy_prompt)

    print("\n4️⃣ FACTUAL CORRECTNESS PROMPT")
    print("-" * 40)

    factual_correctness_prompt = """
PROMPT TEMPLATE (Claims Extraction):
Extract atomic facts/claims from the following text. Each claim should be a standalone statement that can be verified independently.

text: {text}

Claims:
1. [claim 1]
2. [claim 2]
...

PROCESS:
1. Extract claims dari answer → claims_answer
2. Extract claims dari ground_truth → claims_ground_truth  
3. Find overlapping claims → overlapping_claims

CALCULATION (F1-Score):
precision = overlapping_claims / total_claims_in_answer
recall = overlapping_claims / total_claims_in_ground_truth
f1_score = 2 * (precision * recall) / (precision + recall)
"""
    print(factual_correctness_prompt)


def show_evaluation_example():
    """
    Contoh evaluasi actual dengan data dari project
    """

    print("\n\n📝 CONTOH EVALUASI DENGAN DATA PROJECT")
    print("=" * 60)

    example_data = """
INPUT DATA:
{
    "user_input": "Dalam situasi apa saja tenaga kesehatan dapat melakukan aborsi?",
    "retrieved_contexts": [
        "Setiap Orang dilarang melakukan aborsi, kecuali dengan kriteria yang diperbolehkan sesuai dengan ketentuan dalam kitab undang-undang hukum pidana...",
        "Pelaksanaan aborsi hanya dapat dilakukan oleh Tenaga Medis dan dibantu Tenaga Kesehatan yang memiliki kompetensi dan kewenangan..."
    ],
    "response": "Tenaga kesehatan dapat melakukan aborsi jika memenuhi kriteria KUHP, dilakukan oleh tenaga medis kompeten, di fasilitas kesehatan yang memenuhi syarat, dengan persetujuan perempuan hamil dan suami kecuali korban perkosaan.",
    "reference": "Berdasarkan dokumen yang tersedia, tenaga kesehatan dapat melakukan tindakan aborsi yang dibenarkan secara hukum di Indonesia jika memenuhi kriteria yang diperbolehkan sesuai dengan ketentuan dalam kitab undang-undang hukum pidana..."
}
"""
    print(example_data)

    print("\n🧠 LLM EVALUATION PROCESS:")

    evaluation_process = """
1. CONTEXT RECALL:
   LLM Input:
   - contexts: [context1, context2]
   - ground_truth: "Berdasarkan dokumen yang tersedia..."
   
   LLM Task: Cek kalimat mana di ground_truth yang bisa diverifikasi dari contexts
   LLM Output: 
   - "tenaga kesehatan dapat melakukan aborsi" → Yes (ada di context)
   - "jika memenuhi kriteria KUHP" → Yes (ada di context)  
   - "dilakukan oleh tenaga medis kompeten" → Yes (ada di context)
   - "dengan persetujuan perempuan dan suami" → Yes (ada di context)
   
   Score: 4/4 = 1.0

2. FAITHFULNESS: 
   LLM Input:
   - question: "Dalam situasi apa saja..."
   - contexts: [context1, context2]
   - answer: "Tenaga kesehatan dapat melakukan aborsi jika..."
   
   LLM Task: Cek klaim di answer yang didukung contexts
   LLM Output:
   - "memenuhi kriteria KUHP" → 1 (didukung context)
   - "tenaga medis kompeten" → 1 (didukung context)
   - "fasilitas kesehatan memenuhi syarat" → 1 (didukung context)
   - "persetujuan perempuan dan suami" → 1 (didukung context)
   
   Score: 4/4 = 1.0

3. ANSWER RELEVANCY:
   LLM Input:
   - answer: "Tenaga kesehatan dapat melakukan aborsi jika..."
   
   LLM Task: Generate questions dari answer
   LLM Output:
   - "Kapan tenaga kesehatan boleh melakukan aborsi?"
   - "Apa syarat aborsi legal di Indonesia?"
   - "Siapa yang berwenang melakukan aborsi?"
   
   Similarity dengan original question: 0.92

4. FACTUAL CORRECTNESS:
   LLM Input (Answer):
   - text: "Tenaga kesehatan dapat melakukan aborsi jika..."
   
   LLM Output (Claims dari Answer):
   - "Aborsi boleh jika memenuhi kriteria KUHP"
   - "Dilakukan oleh tenaga medis kompeten" 
   - "Di fasilitas kesehatan yang memenuhi syarat"
   - "Dengan persetujuan perempuan hamil dan suami"
   
   LLM Input (Ground Truth):
   - text: "Berdasarkan dokumen yang tersedia..."
   
   LLM Output (Claims dari Ground Truth):
   - "Aborsi dibenarkan jika memenuhi kriteria KUHP"
   - "Dilakukan oleh tenaga medis kompeten"
   - "Di fasilitas kesehatan yang memenuhi syarat"
   - "Dengan persetujuan perempuan dan suami"
   - "Kecuali korban perkosaan tidak perlu persetujuan suami"
   
   Overlapping Claims: 4/4 (precision), 4/5 (recall)
   F1-Score: 2 * (1.0 * 0.8) / (1.0 + 0.8) = 0.89
"""
    print(evaluation_process)


def show_final_result():
    """
    Menunjukkan hasil akhir evaluasi
    """

    print("\n\n📊 HASIL AKHIR EVALUASI")
    print("=" * 60)

    final_result = """
OUTPUT JSON (benchmark_results/ragas_evaluation_xxx_complete.json):
{
  "single_rag": {
    "evaluation_metrics": {
      "context_recall": 0.85,
      "faithfulness": 0.92,
      "factual_correctness(mode=f1)": 0.78,
      "answer_relevancy": 0.88
    },
    "successful_queries": 10,
    "total_queries": 10,
    "success_rate": 1.0,
    "raw_data": [...]
  },
  "multi_agent": {
    "evaluation_metrics": {
      "context_recall": 0.91,
      "faithfulness": 0.95,
      "factual_correctness(mode=f1)": 0.82,
      "answer_relevancy": 0.90
    },
    "successful_queries": 10,
    "total_queries": 10,
    "success_rate": 1.0,
    "raw_data": [...]
  }
}

📈 VISUALISASI (analysis/evaluate_and_visualize.py):
- Radar Chart: Perbandingan multi-dimensi
- Bar Chart: Per-metric comparison
- Heatmap: Performance overview
- Summary Table: Detailed metrics
"""
    print(final_result)


if __name__ == "__main__":
    show_actual_ragas_implementation()
    show_ragas_prompts_detail()
    show_evaluation_example()
    show_final_result()

    print("\n\n🎯 RINGKASAN CARA KERJA RAGAS:")
    print("=" * 60)
    print("✅ RAGAS menggunakan GPT-4.1 sebagai evaluator LLM")
    print("✅ Setiap metrik menggunakan prompt khusus untuk analisis LLM")
    print("✅ LLM menganalisis text dan memberikan judgement numerik")
    print("✅ Hasil agregat menghasilkan skor 0.0-1.0 untuk setiap metrik")
    print("✅ Tidak ada fallback - murni evaluasi berbasis LLM")
    print("✅ Output disimpan dalam JSON dan divisualisasikan")
    print(
        "✅ Implementation: from ragas import evaluate - semua perhitungan dari RAGAS"
    )

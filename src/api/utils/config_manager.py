"""
Configuration dan constants untuk Multi-Step RAG system
"""

# Model embedding configuration
EMBEDDING_CONFIG = {
    "small": {"model": "text-embedding-3-small", "table": "documents_small"},
    "large": {"model": "text-embedding-3-large", "table": "documents"},
}

# Definisikan konfigurasi model LLM
MODELS = {
    "MAIN": {"model": "gpt-4.1-mini", "temperature": 0.2},
    # gabungkan refiner + evaluator:
    "REF_EVAL": {"model": "gpt-4.1-nano", "temperature": 0.25},
    "GENERATOR": {"model": "gpt-4.1-mini", "temperature": 0.2},
    "SUMMARY": {"model": "gpt-4.1-mini", "temperature": 0},
}

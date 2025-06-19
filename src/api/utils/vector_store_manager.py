"""
Vector store management utilities untuk Multi-Step RAG system
"""

import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client
from .config_manager import EMBEDDING_CONFIG
from dotenv import load_dotenv

# Muat variabel ENV sedini mungkin
load_dotenv()

# Initialize Supabase client globally untuk sharing
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)


def get_embeddings(embedding_model="large"):
    """Dapatkan objek embedding berdasarkan model yang dipilih"""
    if embedding_model not in EMBEDDING_CONFIG:
        raise ValueError(f"Model embedding tidak valid: {embedding_model}")

    model_name = EMBEDDING_CONFIG[embedding_model]["model"]
    return OpenAIEmbeddings(model=model_name)


def get_vector_store(embedding_model="large"):
    """Initialize vector store dengan model embedding yang dipilih"""
    if embedding_model not in EMBEDDING_CONFIG:
        raise ValueError(f"Model embedding tidak valid: {embedding_model}")

    embeddings = get_embeddings(embedding_model)
    table_name = EMBEDDING_CONFIG[embedding_model]["table"]
    query_name = (
        "match_documents_small" if embedding_model == "small" else "match_documents"
    )

    return SupabaseVectorStore(
        embedding=embeddings,
        client=supabase,
        table_name=table_name,
        query_name=query_name,
    )

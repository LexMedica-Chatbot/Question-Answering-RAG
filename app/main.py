# app/main.py

import os
from typing import List, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.chains import ConversationalRetrievalChain

try:
    import supabase
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Package 'supabase' tidak terpasang. Tambahkan 'supabase' ke requirements.txt."
    ) from e

# ---------------------------------------------------------------------------
# Inisialisasi Supabase & LangChain components
# ---------------------------------------------------------------------------

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise RuntimeError(
        "Environment variable SUPABASE_URL atau SUPABASE_ANON_KEY belum di-setup."
    )

if not OPENAI_API_KEY:
    raise RuntimeError("Environment variable OPENAI_API_KEY belum di-setup.")

# Inisialisasi client Supabase
sb_client = supabase.create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# Embeddings dan VectorStore
embeddings = OpenAIEmbeddings()

# Ganti `documents` dan `match_documents` sesuai schema Supabase vector store Anda
vector_store = SupabaseVectorStore(
    client=sb_client,
    embedding=embeddings,
    table_name="documents",  # <- pastikan sama dengan nama tabel di Supabase
    query_name="match_documents",  # <- pastikan sama dengan nama function RPC di Supabase
)

retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# Language model
llm = ChatOpenAI(model="gpt-4o-mini")

# Conversational Retrieval Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
)

# ---------------------------------------------------------------------------
# FastAPI setup
# ---------------------------------------------------------------------------

app = FastAPI(title="RAG Question Answering API", version="1.0.0")


class AskRequest(BaseModel):
    question: str = Field(..., description="Pertanyaan yang ingin diajukan")
    chat_history: List[Tuple[str, str]] = Field(
        default_factory=list,
        description="Daftar pasangan (user_message, bot_response) untuk memelihara konteks percakapan.",
    )


class AskResponse(BaseModel):
    answer: str
    sources: List[str]


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(payload: AskRequest):
    """Endpoint utama untuk menanyakan pertanyaan dan mendapatkan jawaban dari sistem RAG."""
    try:
        result = qa_chain(
            {"question": payload.question, "chat_history": payload.chat_history}
        )
        answer = result.get("answer", "")
        docs = result.get("source_documents", [])
        sources = [d.metadata.get("source", "") for d in docs]
        return AskResponse(answer=answer, sources=sources)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

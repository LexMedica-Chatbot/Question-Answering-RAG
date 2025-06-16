"""
History Management Utilities untuk Multi-Step RAG system
"""

from typing import List, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from .config_manager import MODELS

def pairs_to_str(prev: List[Any]) -> List[str]:
    """Ubah previous_responses menjadi list string 'Pertanyaan: …\nJawaban: …'."""
    out = []
    for item in prev[-3:]:  # ambil 3 terakhir
        if isinstance(item, (list, tuple)) and len(item) == 2:
            q, a = item
        elif isinstance(item, dict):
            q, a = item.get("query", ""), item.get("answer", "")
        else:  # sudah string
            out.append(str(item))
            continue
        out.append(f"Pertanyaan: {q}\nJawaban: {a}")
    return out

def summarize_pairs(prev: List[Any]) -> str:
    """Ringkas previous responses untuk context"""
    strs = pairs_to_str(prev)
    if not strs:
        return ""
    joined = "\n\n".join(strs)
    
    summary_llm = ChatOpenAI(**MODELS["SUMMARY"])
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "Ringkas tiap pasangan tanya–jawab di bawah ini (≤2 kalimat/pasangan)."),
        ("human", "{pairs}"),
    ])
    
    return summary_llm.invoke(
        summary_prompt.format_prompt(pairs=joined)
    ).content.strip() 
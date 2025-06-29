"""
Response models untuk Multi-Step RAG API
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class StepInfo(BaseModel):
    """Informasi tentang step dalam processing pipeline"""

    tool: str
    tool_input: Dict[str, Any]
    tool_output: str


class CitationInfo(BaseModel):
    """Informasi citation untuk referensi dokumen"""

    text: str
    source_doc: str
    source_text: str


class MultiStepRAGResponse(BaseModel):
    """Response model untuk Multi-Step RAG endpoint"""

    answer: str
    referenced_documents: List[Dict[str, Any]] = (
        []
    )  # Dokumen aktif yang digunakan dalam jawaban
    all_retrieved_documents: List[Dict[str, Any]] = (
        []
    )  # Semua dokumen yang diambil (termasuk dicabut) untuk deteksi disharmony
    processing_steps: Optional[List[StepInfo]] = None
    processing_time_ms: Optional[int] = None
    model_info: Dict[str, Any] = {}

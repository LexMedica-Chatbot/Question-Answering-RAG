"""
Request models untuk Multi-Step RAG API
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union, Literal


class MultiStepRAGRequest(BaseModel):
    """Request model untuk Multi-Step RAG endpoint"""
    
    query: str = Field(..., description="Query yang akan diproses oleh RAG system")
    embedding_model: Literal["small", "large"] = Field(
        default="large", 
        description="Model embedding yang akan digunakan"
    )
    previous_responses: List[Union[List[str], Dict[str, Any], str]] = Field(
        default_factory=list,
        description="Respons-respons sebelumnya untuk konteks percakapan"
    )
    use_parallel_execution: bool = Field(
        default=True,
        description="Enable parallel tool execution untuk 30-40% speed boost"
    ) 
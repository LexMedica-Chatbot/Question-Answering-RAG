"""
Centralized RAG Tracking System with LangFuse
Provides modular tracking for both Simple and Multi-Step RAG APIs
"""

import os
import time
import re
from typing import Dict, Any, Optional, List
from enum import Enum

class APIType(Enum):
    SIMPLE = "simple_api"
    MULTI_STEP = "multi_step_api"

class ExecutionMode(Enum):
    STANDARD = "standard"
    PARALLEL = "parallel"
    CACHED = "cached"

class RAGTracker:
    """Centralized RAG tracking with LangFuse integration"""
    
    def __init__(self):
        self.enabled = False
        self.langfuse_client = None
        self._init_langfuse()
    
    def _init_langfuse(self):
        """Initialize LangFuse connection"""
        try:
            # Check for LangFuse credentials
            public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
            secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
            host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
            
            if not public_key or not secret_key:
                print("⚠️  LangFuse keys not found. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
                print("   Get your keys from: https://cloud.langfuse.com")
                print("   Running without LangFuse tracking...")
                self.enabled = False
                return
            
            # Try to import and initialize LangFuse
            from langfuse import Langfuse
            
            self.langfuse_client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host
            )
            
            # Test connection
            self.langfuse_client.auth_check()
            self.enabled = True
            print("✅ RAG Tracker (LangFuse) initialized successfully!")
            
        except ImportError:
            print("⚠️  LangFuse library not installed. Install with: pip install langfuse")
            self.enabled = False
        except Exception as e:
            print(f"❌ RAG Tracker initialization failed: {e}")
            self.enabled = False
    
    def start_session(
        self, 
        query: str, 
        api_type: APIType, 
        execution_mode: ExecutionMode = ExecutionMode.STANDARD,
        metadata: Dict[str, Any] = None
    ) -> Optional[Any]:
        """Start a new RAG session with comprehensive metadata"""
        if not self.enabled:
            return None
        
        try:
            # Create session name that clearly identifies API and mode
            session_name = f"{api_type.value}_{execution_mode.value}_session"
            
            # Comprehensive metadata
            session_metadata = {
                "api_type": api_type.value,
                "execution_mode": execution_mode.value,
                "timestamp": time.time(),
                "query_length": len(query),
                "session_type": "rag_chat",
                **(metadata or {})
            }
            
            # Add specific tags based on API type
            if api_type == APIType.SIMPLE:
                session_metadata.update({
                    "pipeline": "basic_rag",
                    "complexity": "low",
                    "steps": "single_step"
                })
            elif api_type == APIType.MULTI_STEP:
                session_metadata.update({
                    "pipeline": "enhanced_multi_step_rag", 
                    "complexity": "high",
                    "steps": "multi_step",
                    "agent_based": True
                })
            
            # Add execution mode specific metadata
            if execution_mode == ExecutionMode.PARALLEL:
                session_metadata.update({
                    "performance_optimized": True,
                    "expected_speedup": "30-40%"
                })
            elif execution_mode == ExecutionMode.CACHED:
                session_metadata.update({
                    "cached_response": True,
                    "cache_hit": True
                })
            
            trace = self.langfuse_client.trace(
                name=session_name,
                input={"user_query": query},
                metadata=session_metadata,
                tags=[api_type.value, execution_mode.value, "rag_system"]
            )
            
            print(f"📊 RAG Session started: {session_name}")
            return trace
            
        except Exception as e:
            print(f"❌ Failed to start RAG session: {e}")
            return None
    
    def track_document_retrieval(
        self, 
        trace, 
        query: str, 
        embedding_model: str, 
        num_docs: int, 
        api_type: APIType,
        docs: list = None,
        metadata: Dict[str, Any] = None
    ):
        """Track document retrieval with API-specific context"""
        if not self.enabled or not trace:
            return None
        
        try:
            retrieval_metadata = {
                "operation": "document_retrieval",
                "embedding_model": embedding_model,
                "num_documents": num_docs,
                "api_type": api_type.value,
                "query_complexity": len(query.split()),
                **(metadata or {})
            }
            
            # Add API-specific retrieval context
            if api_type == APIType.SIMPLE:
                retrieval_metadata["retrieval_type"] = "basic_similarity_search"
            elif api_type == APIType.MULTI_STEP:
                retrieval_metadata["retrieval_type"] = "agent_based_search"
                retrieval_metadata["multi_query_support"] = True
            
            span = trace.span(
                name=f"document_retrieval_{api_type.value}",
                input={"query": query, "embedding_model": embedding_model},
                output={"num_documents": num_docs, "documents": docs[:2] if docs else []},
                metadata=retrieval_metadata
            )
            
            return span
            
        except Exception as e:
            print(f"❌ Failed to track document retrieval: {e}")
            return None
    
    def track_llm_generation(
        self, 
        trace, 
        model: str, 
        input_messages: list, 
        response: str, 
        api_type: APIType,
        usage: Dict[str, int] = None,
        metadata: Dict[str, Any] = None
    ):
        """Track LLM generation with cost calculation"""
        if not self.enabled or not trace:
            return None
        
        try:
            # Calculate cost
            cost = self._calculate_cost(model, usage) if usage else 0
            
            generation_metadata = {
                "model": model,
                "api_type": api_type.value,
                "cost_usd": round(cost, 6),
                "response_length": len(response),
                **(metadata or {})
            }
            
            # Add API-specific generation context
            if api_type == APIType.SIMPLE:
                generation_metadata["generation_type"] = "single_pass_generation"
            elif api_type == APIType.MULTI_STEP:
                generation_metadata["generation_type"] = "agent_based_generation"
                generation_metadata["multi_step_reasoning"] = True
            
            if usage:
                generation_metadata.update({
                    "input_tokens": usage.get('prompt_tokens', 0),
                    "output_tokens": usage.get('completion_tokens', 0),
                    "total_tokens": usage.get('total_tokens', 0)
                })
            
            generation = trace.generation(
                name=f"llm_generation_{api_type.value}",
                model=model,
                input=input_messages,
                output=response,
                usage={
                    "input": usage.get('prompt_tokens', 0) if usage else 0,
                    "output": usage.get('completion_tokens', 0) if usage else 0,
                    "total": usage.get('total_tokens', 0) if usage else 0,
                    "unit": "TOKENS"
                } if usage else None,
                metadata=generation_metadata
            )
            
            return generation
            
        except Exception as e:
            print(f"❌ Failed to track LLM generation: {e}")
            return None
    
    def track_processing_steps(
        self, 
        trace, 
        steps: List[Dict[str, Any]], 
        api_type: APIType
    ):
        """Track processing steps for multi-step RAG"""
        if not self.enabled or not trace or api_type != APIType.MULTI_STEP:
            return None
        
        try:
            for i, step in enumerate(steps):
                step_metadata = {
                    "step_number": i + 1,
                    "total_steps": len(steps),
                    "api_type": api_type.value,
                    "tool_name": step.get("tool", "unknown")
                }
                
                trace.span(
                    name=f"processing_step_{i+1}_{step.get('tool', 'unknown')}",
                    input=step.get("tool_input", {}),
                    output=step.get("tool_output", "")[:500] + "..." if len(str(step.get("tool_output", ""))) > 500 else step.get("tool_output", ""),
                    metadata=step_metadata
                )
                
        except Exception as e:
            print(f"❌ Failed to track processing steps: {e}")
    
    def finalize_session(
        self, 
        trace, 
        final_answer: str, 
        api_type: APIType,
        execution_mode: ExecutionMode,
        processing_time_ms: int, 
        estimated_cost: float,
        additional_metadata: Dict[str, Any] = None
    ):
        """Finalize RAG session with comprehensive metrics"""
        if not self.enabled or not trace:
            return
        
        try:
            final_metadata = {
                "api_type": api_type.value,
                "execution_mode": execution_mode.value,
                "processing_time_ms": processing_time_ms,
                "estimated_cost_usd": round(estimated_cost, 6),
                "answer_length": len(final_answer),
                "status": "success",
                "performance_tier": self._get_performance_tier(processing_time_ms),
                **(additional_metadata or {})
            }
            
            # Add API-specific final metrics
            if api_type == APIType.SIMPLE:
                final_metadata["pipeline_efficiency"] = "optimized_for_speed"
            elif api_type == APIType.MULTI_STEP:
                final_metadata["pipeline_efficiency"] = "optimized_for_accuracy"
                
                if execution_mode == ExecutionMode.PARALLEL:
                    final_metadata["speedup_achieved"] = True
            
            trace.update(
                output={"answer": final_answer},
                metadata=final_metadata
            )
            
            # Flush data to ensure it's sent to LangFuse
            if self.langfuse_client:
                self.langfuse_client.flush()
                print(f"📊 RAG Session finalized: {api_type.value}_{execution_mode.value}")
                
        except Exception as e:
            print(f"❌ Failed to finalize session: {e}")
    
    def _calculate_cost(self, model: str, usage: Dict[str, int]) -> float:
        """Calculate cost based on model and usage"""
        PRICING = {
            # OpenAI Pricing (per 1K tokens)
            "gpt-4o": {"input": 0.00250, "output": 0.01000},
            "gpt-4o-mini": {"input": 0.000150, "output": 0.000600},
            "gpt-4.1-mini": {"input": 0.000150, "output": 0.000600},
            "gpt-4-turbo": {"input": 0.01000, "output": 0.03000},
            "gpt-3.5-turbo": {"input": 0.000500, "output": 0.001500},
            
            # Embeddings
            "text-embedding-3-large": {"input": 0.000130, "output": 0},
            "text-embedding-3-small": {"input": 0.000020, "output": 0},
            "text-embedding-ada-002": {"input": 0.000100, "output": 0},
        }
        
        if not usage:
            return 0.0
        
        input_tokens = usage.get('prompt_tokens', 0)
        output_tokens = usage.get('completion_tokens', 0)
        
        # Find pricing for model
        pricing = None
        if model in PRICING:
            pricing = PRICING[model]
        else:
            # Try to find similar model
            for known_model in PRICING:
                if known_model in model.lower():
                    pricing = PRICING[known_model]
                    break
        
        if not pricing:
            return 0.0
        
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
    
    def _get_performance_tier(self, processing_time_ms: int) -> str:
        """Categorize performance based on processing time"""
        if processing_time_ms < 5000:  # < 5s
            return "fast"
        elif processing_time_ms < 15000:  # < 15s
            return "medium"
        elif processing_time_ms < 60000:  # < 1min
            return "slow"
        else:
            return "very_slow"
    
    def get_status(self) -> Dict[str, Any]:
        """Get tracker status"""
        return {
            "enabled": self.enabled,
            "provider": "LangFuse" if self.enabled else "None",
            "features": [
                "API-specific tracing",
                "Execution mode differentiation", 
                "Cost tracking per request",
                "Performance tier analysis",
                "Multi-step process tracking"
            ] if self.enabled else [],
            "dashboard_url": "https://cloud.langfuse.com" if self.enabled else None
        }

    def track_user_pattern(
        self, 
        trace, 
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """Track user query patterns and intent classification"""
        if not self.enabled or not trace:
            return None
        
        try:
            # Classify query intent
            query_intent = self._classify_query_intent(query)
            query_complexity = self._analyze_query_complexity(query)
            
            user_metadata = {
                "user_id": user_id or "anonymous",
                "session_id": session_id,
                "query_intent": query_intent,
                "query_complexity": query_complexity,
                "query_type": self._get_query_type(query),
                "domain_specific_terms": self._extract_domain_terms(query),
                "timestamp": time.time()
            }
            
            trace.span(
                name="user_pattern_analysis",
                input={"query": query},
                output=user_metadata,
                metadata=user_metadata
            )
            
            return user_metadata
            
        except Exception as e:
            print(f"❌ Failed to track user pattern: {e}")
            return None
    
    def track_retrieval_quality(
        self, 
        trace, 
        query: str,
        retrieved_docs: List[Dict],
        relevance_scores: Optional[List[float]] = None
    ):
        """Track quality metrics of document retrieval"""
        if not self.enabled or not trace:
            return None
        
        try:
            quality_metrics = {
                "num_documents": len(retrieved_docs),
                "avg_doc_length": sum(len(doc.get("content", "")) for doc in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0,
                "diversity_score": self._calculate_diversity_score(retrieved_docs),
                "coverage_score": self._calculate_coverage_score(query, retrieved_docs),
                "relevance_scores": relevance_scores or [],
                "avg_relevance": sum(relevance_scores) / len(relevance_scores) if relevance_scores else None
            }
            
            trace.span(
                name="retrieval_quality_analysis",
                input={"query": query, "num_docs": len(retrieved_docs)},
                output=quality_metrics,
                metadata=quality_metrics
            )
            
            return quality_metrics
            
        except Exception as e:
            print(f"❌ Failed to track retrieval quality: {e}")
            return None
    
    def track_answer_quality(
        self, 
        trace, 
        query: str,
        answer: str,
        referenced_docs: List[Dict],
        user_feedback: Optional[Dict] = None
    ):
        """Track answer quality and user satisfaction"""
        if not self.enabled or not trace:
            return None
        
        try:
            quality_metrics = {
                "answer_length": len(answer),
                "answer_completeness": self._assess_completeness(query, answer),
                "citation_count": len(referenced_docs),
                "answer_structure_score": self._analyze_answer_structure(answer),
                "factual_coverage": self._assess_factual_coverage(answer, referenced_docs),
                "user_feedback": user_feedback
            }
            
            trace.span(
                name="answer_quality_analysis",
                input={"query": query, "answer_preview": answer[:200] + "..."},
                output=quality_metrics,
                metadata=quality_metrics
            )
            
            return quality_metrics
            
        except Exception as e:
            print(f"❌ Failed to track answer quality: {e}")
            return None

    def _classify_query_intent(self, query: str) -> str:
        """Classify user query intent"""
        query_lower = query.lower()
        
        # Intent patterns for legal health domain
        if any(word in query_lower for word in ["apa", "apakah", "bagaimana", "mengapa", "kenapa"]):
            return "question"
        elif any(word in query_lower for word in ["jelaskan", "sebutkan", "uraikan", "definisi"]):
            return "explanation"
        elif any(word in query_lower for word in ["pasal", "ayat", "peraturan", "undang-undang"]):
            return "legal_reference"
        elif any(word in query_lower for word in ["sanksi", "hukuman", "pelanggaran", "denda"]):
            return "legal_consequence"
        elif any(word in query_lower for word in ["prosedur", "tata cara", "langkah", "proses"]):
            return "procedure"
        elif any(word in query_lower for word in ["kewajiban", "hak", "tanggung jawab"]):
            return "rights_obligations"
        else:
            return "general_inquiry"
    
    def _analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """Analyze query complexity metrics"""
        words = query.split()
        sentences = query.split('.')
        
        return {
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0,
            "has_legal_terms": self._has_legal_terms(query),
            "complexity_score": self._calculate_complexity_score(query)
        }
    
    def _get_query_type(self, query: str) -> str:
        """Determine the type of legal query"""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ["dokter", "tenaga medis", "praktik"]):
            return "medical_professional"
        elif any(term in query_lower for term in ["rumah sakit", "fasilitas kesehatan"]):
            return "healthcare_facility"
        elif any(term in query_lower for term in ["pasien", "informed consent", "rekam medis"]):
            return "patient_rights"
        elif any(term in query_lower for term in ["obat", "farmasi", "apotek"]):
            return "pharmaceutical"
        elif any(term in query_lower for term in ["sanksi", "pelanggaran", "hukuman"]):
            return "legal_violation"
        else:
            return "general_health_law"
    
    def _extract_domain_terms(self, query: str) -> List[str]:
        """Extract domain-specific legal and medical terms"""
        domain_terms = []
        query_lower = query.lower()
        
        # Legal terms
        legal_terms = ["pasal", "ayat", "undang-undang", "peraturan", "kewajiban", "hak", "sanksi"]
        # Medical terms  
        medical_terms = ["dokter", "pasien", "rumah sakit", "obat", "rekam medis", "informed consent"]
        
        for term in legal_terms + medical_terms:
            if term in query_lower:
                domain_terms.append(term)
        
        return domain_terms
    
    def _has_legal_terms(self, query: str) -> bool:
        """Check if query contains legal terminology"""
        legal_indicators = ["pasal", "ayat", "uu", "pp", "peraturan", "undang-undang", "kewajiban", "hak"]
        return any(term in query.lower() for term in legal_indicators)
    
    def _calculate_complexity_score(self, query: str) -> float:
        """Calculate overall query complexity score (0-1)"""
        words = len(query.split())
        has_legal = self._has_legal_terms(query)
        has_multiple_concepts = len(self._extract_domain_terms(query)) > 2
        
        base_score = min(words / 20, 1.0)  # Normalize by word count
        if has_legal:
            base_score += 0.2
        if has_multiple_concepts:
            base_score += 0.2
        
        return min(base_score, 1.0)
    
    def _calculate_diversity_score(self, docs: List[Dict]) -> float:
        """Calculate diversity of retrieved documents"""
        if not docs:
            return 0.0
        
        # Simple diversity based on unique document types/sources
        sources = set()
        for doc in docs:
            metadata = doc.get("metadata", {})
            source = f"{metadata.get('jenis_peraturan', '')}-{metadata.get('tahun_peraturan', '')}"
            sources.add(source)
        
        return len(sources) / len(docs) if docs else 0.0
    
    def _calculate_coverage_score(self, query: str, docs: List[Dict]) -> float:
        """Calculate how well documents cover the query terms"""
        if not docs:
            return 0.0
        
        query_terms = set(query.lower().split())
        covered_terms = set()
        
        for doc in docs:
            content = doc.get("content", "").lower()
            for term in query_terms:
                if term in content:
                    covered_terms.add(term)
        
        return len(covered_terms) / len(query_terms) if query_terms else 0.0
    
    def _assess_completeness(self, query: str, answer: str) -> Dict[str, Any]:
        """Assess answer completeness"""
        return {
            "has_definition": "definisi" in answer.lower() or "adalah" in answer.lower(),
            "has_examples": "contoh" in answer.lower() or "misalnya" in answer.lower(), 
            "has_references": "[" in answer or "pasal" in answer.lower(),
            "answer_depth": "detailed" if len(answer) > 1000 else "brief",
            "structural_elements": self._count_structural_elements(answer)
        }
    
    def _analyze_answer_structure(self, answer: str) -> Dict[str, Any]:
        """Analyze answer structure quality"""
        return {
            "has_introduction": answer.startswith(("Berdasarkan", "Menurut", "Dalam")),
            "has_numbered_points": bool(re.search(r'\d+\.', answer)),
            "has_conclusion": "demikian" in answer.lower() or "kesimpulan" in answer.lower(),
            "paragraph_count": len(answer.split('\n\n')),
            "citation_format": "proper" if re.search(r'\[.*?\]\s*\(.*?\)', answer) else "basic"
        }
    
    def _assess_factual_coverage(self, answer: str, docs: List[Dict]) -> float:
        """Assess how well answer covers facts from source documents"""
        if not docs:
            return 0.0
        
        doc_facts = []
        for doc in docs:
            content = doc.get("content", "")
            # Extract key facts (simple approach - could be enhanced)
            facts = re.findall(r'[A-Z][^.!?]*[.!?]', content)
            doc_facts.extend(facts[:3])  # Top 3 facts per doc
        
        covered_facts = 0
        for fact in doc_facts:
            key_terms = [word for word in fact.split() if len(word) > 4][:3]
            if any(term.lower() in answer.lower() for term in key_terms):
                covered_facts += 1
        
        return covered_facts / len(doc_facts) if doc_facts else 0.0
    
    def _count_structural_elements(self, answer: str) -> Dict[str, int]:
        """Count structural elements in answer"""
        return {
            "numbered_lists": len(re.findall(r'\d+\.', answer)),
            "bullet_points": answer.count('•') + answer.count('-'),
            "paragraphs": len(answer.split('\n\n')),
            "citations": len(re.findall(r'\[.*?\]', answer))
        }

    def track_error(
        self, 
        trace, 
        error_type: str,
        error_message: str,
        api_type: APIType,
        execution_mode: ExecutionMode,
        context: Dict[str, Any] = None
    ):
        """Track errors and failures in RAG pipeline"""
        if not self.enabled or not trace:
            return None
        
        try:
            error_metadata = {
                "error_type": error_type,
                "error_message": error_message,
                "api_type": api_type.value,
                "execution_mode": execution_mode.value,
                "timestamp": time.time(),
                "severity": self._classify_error_severity(error_type),
                "recovery_possible": self._is_recoverable_error(error_type),
                "context": context or {}
            }
            
            trace.span(
                name=f"error_{error_type.lower()}",
                input={"error_type": error_type},
                output={"handled": True, "message": error_message},
                metadata=error_metadata
            )
            
            print(f"🚨 Error tracked: {error_type} in {api_type.value}")
            return error_metadata
            
        except Exception as e:
            print(f"❌ Failed to track error: {e}")
            return None
    
    def track_ab_test(
        self, 
        trace, 
        test_name: str,
        variant: str,
        api_type: APIType,
        query: str,
        performance_metrics: Dict[str, Any] = None
    ):
        """Track A/B test experiments"""
        if not self.enabled or not trace:
            return None
        
        try:
            ab_metadata = {
                "test_name": test_name,
                "variant": variant,
                "api_type": api_type.value,
                "query_hash": hash(query) % 10000,  # Anonymized query identifier
                "performance_metrics": performance_metrics or {},
                "timestamp": time.time(),
                "experiment_cohort": self._get_experiment_cohort(query)
            }
            
            trace.span(
                name=f"ab_test_{test_name}",
                input={"variant": variant, "test_name": test_name},
                output=ab_metadata,
                metadata=ab_metadata
            )
            
            return ab_metadata
            
        except Exception as e:
            print(f"❌ Failed to track A/B test: {e}")
            return None
    
    def track_system_performance(
        self, 
        trace,
        system_metrics: Dict[str, Any],
        api_type: APIType
    ):
        """Track system-level performance metrics"""
        if not self.enabled or not trace:
            return None
        
        try:
            perf_metadata = {
                "api_type": api_type.value,
                "cpu_usage": system_metrics.get("cpu_percent", 0),
                "memory_usage": system_metrics.get("memory_percent", 0),
                "disk_usage": system_metrics.get("disk_percent", 0),
                "response_time_p95": system_metrics.get("response_time_p95", 0),
                "concurrent_requests": system_metrics.get("concurrent_requests", 0),
                "cache_hit_rate": system_metrics.get("cache_hit_rate", 0),
                "error_rate": system_metrics.get("error_rate", 0),
                "timestamp": time.time()
            }
            
            trace.span(
                name="system_performance_monitoring",
                input={"monitoring_scope": api_type.value},
                output=perf_metadata,
                metadata=perf_metadata
            )
            
            return perf_metadata
            
        except Exception as e:
            print(f"❌ Failed to track system performance: {e}")
            return None
    
    def _classify_error_severity(self, error_type: str) -> str:
        """Classify error severity level"""
        critical_errors = ["database_connection", "llm_api_failure", "auth_failure"]
        high_errors = ["document_retrieval_failure", "embedding_failure"] 
        medium_errors = ["cache_miss", "timeout", "rate_limit"]
        
        if error_type.lower() in critical_errors:
            return "critical"
        elif error_type.lower() in high_errors:
            return "high"
        elif error_type.lower() in medium_errors:
            return "medium"
        else:
            return "low"
    
    def _is_recoverable_error(self, error_type: str) -> bool:
        """Determine if error is recoverable"""
        recoverable_errors = ["timeout", "rate_limit", "cache_miss", "temporary_failure"]
        return error_type.lower() in recoverable_errors
    
    def _get_experiment_cohort(self, query: str) -> str:
        """Assign user to experiment cohort based on query hash"""
        cohort_hash = hash(query) % 100
        if cohort_hash < 50:
            return "control"
        else:
            return "treatment"

# Global tracker instance
rag_tracker = RAGTracker()

# Convenience functions for backward compatibility
def track_rag_session(query: str, embedding_model: str, api_type: str = "simple", execution_mode: str = "standard") -> Optional[Any]:
    """Convenience function for starting RAG session"""
    api_enum = APIType.SIMPLE if api_type == "simple" else APIType.MULTI_STEP
    mode_enum = ExecutionMode.PARALLEL if execution_mode == "parallel" else ExecutionMode.STANDARD
    
    return rag_tracker.start_session(
        query=query, 
        api_type=api_enum, 
        execution_mode=mode_enum,
        metadata={"embedding_model": embedding_model}
    )

def track_document_retrieval(trace, query: str, model: str, num_docs: int, docs: list = None, api_type: str = "simple"):
    """Convenience function for tracking document retrieval"""
    api_enum = APIType.SIMPLE if api_type == "simple" else APIType.MULTI_STEP
    return rag_tracker.track_document_retrieval(trace, query, model, num_docs, api_enum, docs)

def track_llm_call(trace, model: str, messages: list, response: str, usage: Dict[str, int] = None, api_type: str = "simple"):
    """Convenience function for tracking LLM calls"""
    api_enum = APIType.SIMPLE if api_type == "simple" else APIType.MULTI_STEP
    return rag_tracker.track_llm_generation(trace, model, messages, response, api_enum, usage)

def finalize_rag_session(trace, final_answer: str, processing_time_ms: int, estimated_cost: float, api_type: str = "simple", execution_mode: str = "standard"):
    """Convenience function for finalizing sessions"""
    api_enum = APIType.SIMPLE if api_type == "simple" else APIType.MULTI_STEP
    mode_enum = ExecutionMode.PARALLEL if execution_mode == "parallel" else ExecutionMode.STANDARD
    
    return rag_tracker.finalize_session(trace, final_answer, api_enum, mode_enum, processing_time_ms, estimated_cost)

# Legacy compatibility
class CostTracker:
    @classmethod
    def calculate_cost(cls, model: str, input_tokens: int, output_tokens: int = 0) -> float:
        usage = {
            'prompt_tokens': input_tokens,
            'completion_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens
        }
        return rag_tracker._calculate_cost(model, usage)

langfuse_tracker = rag_tracker  # Alias for compatibility 
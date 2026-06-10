import json
import logging
import re
from typing import List, Dict, Any, TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

# Configure a structured tracking log format
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# =====================================================================
# 1. Structural Domain Schemas & Memory Cache State
# =====================================================================
class ProfessionalTelemetry(BaseModel):
    """Normalized metrics captured for strict mathematical alignment scoring."""
    years_of_experience: float = Field(default=0.0, ge=0.0)
    profile_completeness_score: float = Field(default=0.0)
    github_activity_score: float = Field(default=-1.0)
    search_appearance_30d: int = Field(default=0)
    skills_toolkit: List[str] = Field(default_factory=list)
    calculated_timeline_months: int = Field(default=0)

class SanitizedProfilePayload(BaseModel):
    """The polished output payload forwarded directly to the indexing layers."""
    candidate_id: str
    semantic_narrative_block: str
    telemetry: ProfessionalTelemetry

class ConcurrentStreamState(TypedDict):
    """The central state engine memory shared between processing workers."""
    raw_buffer_chunk: List[str]
    validated_structural_records: List[Dict[str, Any]]
    audited_production_pool: List[SanitizedProfilePayload]
    anomaly_telemetry_counter: int

# =====================================================================
# 2. SchemaGuardWorker (Deterministic Pre-Processing Node)
# =====================================================================
class SchemaGuardWorker:
    """Enforces structural contracts at sub-millisecond speeds to drop broken records."""
    def __init__(self):
        self.worker_identity = "SchemaGuardWorker"

    def __call__(self, state: ConcurrentStreamState) -> Dict[str, Any]:
        valid_records = []
        for line in state["raw_buffer_chunk"]:
            line_content = line.strip()
            if not line_content:
                continue
            try:
                record = json.loads(line_content)
                mandatory_facets = ["candidate_id", "profile", "career_history", "skills", "redrob_signals"]
                if all(facet in record for facet in mandatory_facets):
                    valid_records.append(record)
            except (json.JSONDecodeError, TypeError):
                continue
        return {"validated_structural_records": valid_records}

# =====================================================================
# 3. CognitiveHistoryAuditor (High-Performance Pattern Match Node)
# =====================================================================
class CognitiveHistoryAuditor:
    """Processes text patterns to neutralize fraudulent keyword stuffers instantly."""
    def __init__(self):
        self.worker_identity = "CognitiveHistoryAuditor"
        # Compile static regex patterns to catch non-tech title-stuffing mismatches
        self.non_technical_guard = re.compile(
            r'\b(marketing|sales|recruiter|hr|graphic|content|support|operations)\b', 
            re.IGNORECASE
        )
        self.core_ai_guard = re.compile(
            r'\b(rag|llm|pytorch|quantization|embeddings|vllm|langgraph)\b', 
            re.IGNORECASE
        )

    def __call__(self, state: ConcurrentStreamState) -> Dict[str, Any]:
        passed_records = []
        detected_anomalies = 0
        
        for record in state["validated_structural_records"]:
            candidate_id = record["candidate_id"]
            profile = record.get("profile", {})
            history = record.get("career_history", [])
            signals = record.get("redrob_signals", {})
            skills = record.get("skills", [])
            
            # 1. Timeline Chronology Audit Loop
            total_months = sum(int(job.get("duration_months", 0)) for job in history if job.get("duration_months"))
            calculated_years = total_months / 12.0
            stated_years = float(profile.get("years_of_experience", 0.0))
            
            # Check for math anomalies (e.g., experience listed without historical timeline entries)
            if stated_years > 0 and calculated_years == 0:
                detected_anomalies += 1
                continue
                
            # 2. Pattern Match Auditing: Verify current title vs keyword-stuffer profiles
            current_title = profile.get("current_title", "")
            summary = profile.get("summary", "")
            
            # Identify keyword stuffers (e.g., non-tech title stuffed with core AI keywords)
            if self.non_technical_guard.search(current_title) and self.core_ai_guard.search(summary):
                logging.warning(f"[🚨 Profile Audit Alert] Candidate {candidate_id} flagged for layout keyword stuffing.")
                detected_anomalies += 1
                continue
            
            # Formulate text payload block for next retrieval steps
            skills_string = ", ".join([s.get("name", "") for s in skills])
            headline = profile.get("headline", "")
            consolidated_narrative = f"ID: {candidate_id} | Headline: {headline} | Summary: {summary} | Toolkit: {skills_string}"
            
            telemetry_data = ProfessionalTelemetry(
                years_of_experience=stated_years,
                profile_completeness_score=float(signals.get("profile_completeness_score", 0.0)),
                github_activity_score=float(signals.get("github_activity_score", -1.0)),
                search_appearance_30d=int(signals.get("search_appearance_30d", 0)),
                skills_toolkit=[s.get("name", "").lower() for s in skills],
                calculated_timeline_months=total_months
            )
            
            passed_records.append(SanitizedProfilePayload(
                candidate_id=candidate_id,
                semantic_narrative_block=consolidated_narrative,
                telemetry=telemetry_data
            ))
            
        return {
            "audited_production_pool": state.get("audited_production_pool", []) + passed_records,
            "anomaly_telemetry_counter": state.get("anomaly_telemetry_counter", 0) + detected_anomalies
        }

# =====================================================================
# 4. Build and Compile the Processing State Graph
# =====================================================================
def initialize_graph_runtime() -> StateGraph:
    graph_builder = StateGraph(ConcurrentStreamState)
    structural_cleaner = SchemaGuardWorker()
    cognitive_auditor = CognitiveHistoryAuditor()
    
    graph_builder.add_node("execute_structural_cleaning", structural_cleaner)
    graph_builder.add_node("execute_cognitive_auditing", cognitive_auditor)
    
    graph_builder.set_entry_point("execute_structural_cleaning")
    graph_builder.add_edge("execute_structural_cleaning", "execute_cognitive_auditing")
    graph_builder.add_edge("execute_cognitive_auditing", END)
    
    return graph_builder.compile()
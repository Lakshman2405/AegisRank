import json
import logging
import re
from typing import List, Dict, Any, TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# =====================================================================
# 1. Structural Domain Schemas & Memory Cache State
# =====================================================================

class ProfessionalTelemetry(BaseModel):
    """Rich telemetry retained for ranking."""
    years_of_experience: float = Field(default=0.0, ge=0.0)
    profile_completeness_score: float = Field(default=0.0)
    github_activity_score: float = Field(default=-1.0)
    search_appearance_30d: int = Field(default=0)
    skills_toolkit: List[str] = Field(default_factory=list)
    calculated_timeline_months: int = Field(default=0)
    open_to_work_flag: bool = False
    recruiter_response_rate: float = 0.0
    interview_completion_rate: float = 0.0
    offer_acceptance_rate: float = -1.0
    notice_period_days: int = 999
    saved_by_recruiters_30d: int = 0
    profile_views_received_30d: int = 0
    applications_submitted_30d: int = 0
    verified_email: bool = False
    verified_phone: bool = False
    linkedin_connected: bool = False
    skill_assessment_scores: Dict[str, float] = Field(default_factory=dict)
class SanitizedProfilePayload(BaseModel):
    """Clean payload forwarded to vector indexing."""
    candidate_id: str
    semantic_narrative_block: str
    telemetry: ProfessionalTelemetry
class ConcurrentStreamState(TypedDict):
    """Shared graph state."""
    raw_buffer_chunk: List[str]
    validated_structural_records: List[Dict[str, Any]]
    audited_production_pool: List[SanitizedProfilePayload]
    anomaly_telemetry_counter: int

# =====================================================================
# 2. SchemaGuardWorker
# =====================================================================

class SchemaGuardWorker:
    """Drops malformed records."""
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
                mandatory_facets = [
                    "candidate_id",
                    "profile",
                    "career_history",
                    "skills",
                    "redrob_signals"
                ]
                if all(facet in record for facet in mandatory_facets):
                    valid_records.append(record)
            except (json.JSONDecodeError, TypeError):
                continue
        return {
            "validated_structural_records": valid_records
        }


# =====================================================================
# 3. CognitiveHistoryAuditor
# =====================================================================

class CognitiveHistoryAuditor:
    """Fraud detection + candidate enrichment."""
    def __init__(self):
        self.worker_identity = "CognitiveHistoryAuditor"
        self.non_technical_guard = re.compile(
            r"\b(marketing|sales|recruiter|hr|graphic|content|support|operations)\b",
            re.IGNORECASE
        )
        self.core_ai_guard = re.compile(
            r"\b(rag|llm|pytorch|quantization|embeddings|vllm|langgraph)\b",
            re.IGNORECASE
        )

    def __call__( self, state: ConcurrentStreamState) -> Dict[str, Any]:
        passed_records = []
        detected_anomalies = 0
        for record in state["validated_structural_records"]:
            candidate_id = record["candidate_id"]
            profile = record.get("profile", {})
            history = record.get("career_history", [])
            signals = record.get("redrob_signals", {})
            skills = record.get("skills", [])
            # =========================================================
            # Timeline Validation
            # =========================================================
            total_months = sum(
                int(job.get("duration_months", 0))
                for job in history
                if job.get("duration_months")
            )
            calculated_years = total_months / 12.0
            stated_years = float(
                profile.get("years_of_experience", 0.0)
            )
            if stated_years > 0 and calculated_years == 0:
                detected_anomalies += 1
                continue
            # =========================================================
            # Keyword Stuffing Detection
            # =========================================================
            current_title = profile.get(
                "current_title", ""
            )
            summary = profile.get(
                "summary", ""
            )
            if (self.non_technical_guard.search(current_title) and self.core_ai_guard.search(summary)):
                logging.warning(
                    f"[🚨 Profile Audit Alert] Candidate "
                    f"{candidate_id} flagged for keyword stuffing."
                )
                detected_anomalies += 1
                continue
            # =========================================================
            # Rich Candidate Narrative
            # =========================================================
            headline = profile.get(
                "headline", ""
            )
            current_company = profile.get(
                "current_company", ""
            )
            skills_string = ", ".join(
                [
                    s.get("name", "")
                    for s in skills
                    if s.get("name")
                ]
            )
            career_segments = []
            for role in history:
                role_title = role.get(
                    "title",
                    ""
                )
                role_company = role.get(
                    "company",
                    ""
                )
                role_description = role.get(
                    "description",
                    ""
                )
                role_description = role_description[:100]
                career_segments.append(
                    f"{role_title} at "
                    f"{role_description}"
                )
            career_history_text = " ".join(
                career_segments
            )
            assessment_scores = signals.get(
                "skill_assessment_scores",
                {}
            )
            assessment_text = ", ".join(
                [
                    f"{skill}:{score}"
                    for skill, score
                    in assessment_scores.items()
                ]
            )
            consolidated_narrative = f"""
                        Current Title:
                        {current_title}

                        Current Company:
                        {current_company}

                        Professional Summary:
                        {summary}

                        Career History:
                        {career_history_text}

                        Skills:
                        {skills_string}

                        """.strip()

            # =========================================================
            # Rich Telemetry
            # =========================================================
            telemetry_data = ProfessionalTelemetry(
                years_of_experience=stated_years,
                profile_completeness_score=float(
                    signals.get(
                        "profile_completeness_score",
                        0.0
                    )
                ),
                github_activity_score=float(
                    signals.get(
                        "github_activity_score",
                        -1.0
                    )
                ),
                search_appearance_30d=int(
                    signals.get(
                        "search_appearance_30d",
                        0
                    )
                ),
                skills_toolkit=[
                    s.get("name", "").lower()
                    for s in skills
                    if s.get("name")
                ],
                calculated_timeline_months=total_months,
                open_to_work_flag=bool(
                    signals.get(
                        "open_to_work_flag",
                        False
                    )
                ),
                recruiter_response_rate=float(
                    signals.get(
                        "recruiter_response_rate",
                        0.0
                    )
                ),
                interview_completion_rate=float(
                    signals.get(
                        "interview_completion_rate",
                        0.0
                    )
                ),
                offer_acceptance_rate=float(
                    signals.get(
                        "offer_acceptance_rate",
                        -1.0
                    )
                ),
                notice_period_days=int(
                    signals.get(
                        "notice_period_days",
                        999
                    )
                ),
                saved_by_recruiters_30d=int(
                    signals.get(
                        "saved_by_recruiters_30d",
                        0
                    )
                ),
                profile_views_received_30d=int(
                    signals.get(
                        "profile_views_received_30d",
                        0
                    )
                ),
                applications_submitted_30d=int(
                    signals.get(
                        "applications_submitted_30d",
                        0
                    )
                ),
                verified_email=bool(
                    signals.get(
                        "verified_email",
                        False
                    )
                ),
                verified_phone=bool(
                    signals.get(
                        "verified_phone",
                        False
                    )
                ),
                linkedin_connected=bool(
                    signals.get(
                        "linkedin_connected",
                        False
                    )
                ),
                skill_assessment_scores={
                    str(k): float(v)
                    for k, v
                    in assessment_scores.items()
                }
            )
            passed_records.append(
                SanitizedProfilePayload(
                    candidate_id=candidate_id,
                    semantic_narrative_block=consolidated_narrative,
                    telemetry=telemetry_data
                )
            )

        return {
            "audited_production_pool":
                state.get(
                    "audited_production_pool",
                    []
                ) + passed_records,

            "anomaly_telemetry_counter":
                state.get(
                    "anomaly_telemetry_counter",
                    0
                ) + detected_anomalies
        }


# =====================================================================
# 4. Build and Compile Processing Graph
# =====================================================================

def initialize_graph_runtime() -> StateGraph:
    graph_builder = StateGraph(
        ConcurrentStreamState
    )
    structural_cleaner = SchemaGuardWorker()
    cognitive_auditor = CognitiveHistoryAuditor()
    graph_builder.add_node(
        "execute_structural_cleaning",
        structural_cleaner
    )
    graph_builder.add_node(
        "execute_cognitive_auditing",
        cognitive_auditor
    )
    graph_builder.set_entry_point(
        "execute_structural_cleaning"
    )
    graph_builder.add_edge(
        "execute_structural_cleaning",
        "execute_cognitive_auditing"
    )
    graph_builder.add_edge(
        "execute_cognitive_auditing",
        END
    )
    return graph_builder.compile()
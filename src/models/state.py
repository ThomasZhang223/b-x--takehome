"""
State models for agent workflow.
"""

from pydantic import BaseModel

from src.models.schema import RetrievedContext, StudyBlock, TopicNode


class SessionState(BaseModel):
    courses: list[str] = ["PHYS 234", "SYSD 300", "HLTH 204"]
    course_weights: dict[str, float] = {}
    topics: list[TopicNode] = []
    retrieved_contexts: list[RetrievedContext] = []
    study_blocks: list[StudyBlock] = []
    daily_max_hours: float = 4.0
    midterm_dates: dict[str, str] = {}
    start_date: str = ""

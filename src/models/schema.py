"""
Schema definitions for data models.

All models are JSON-serializable for ADK session.state compatibility.
"""

from typing import Literal, Optional

from pydantic import BaseModel

# ADK Session State Keys:
# "courses" -> List[str] e.g. ["PHYS 234", "SYSD 300", "HLTH 204"]
# "course_configs" -> Dict[str, dict]  (CourseConfig.model_dump() for each course)
# "topics" -> List[dict]  (TopicNode.model_dump() for each topic)
# "retrieved_contexts" -> List[dict]  (RetrievedContext.model_dump())
# "retrieved_chunk_ids" -> List[str]  (set of all retrieved child_chunk_ids, stored as list for JSON)
# "topic_embeddings_cache" -> Dict[str, List[str]]  (topic_id -> list of child_chunk_ids)
# "validation_gaps" -> List[dict]
# "validation_passed" -> bool  (used by LoopAgent to decide exit)
# "study_blocks" -> List[dict]  (StudyBlock.model_dump())
# "output_path" -> str  (final study_plan.md path)


class CourseConfig(BaseModel):
    """User configuration for a single course."""
    course: str
    depth: Literal["deep_dive", "moderate", "light_review"]
    days_until_midterm: int


class TopicNode(BaseModel):
    topic_id: str
    course: str
    raw_name: str
    keywords: list[str]
    required_depth: Optional[str] = None


# Using a parent-child chunking strategy here
# parent is larger section of text that contains many child chunks
class ParentChunk(BaseModel):
    parent_id: str
    source_file: str
    page_number: int
    text: str
    child_ids: list[str] = []


# Child chunk is specific keywords/formulas/ideas
# is embedded into vector db
class ChildChunk(BaseModel):
    child_id: str
    parent_id: str
    text: str
    source_file: str
    page_number: int


# retrieves top k child chunks then looks up the parent chunk associated with it
class RetrievedContext(BaseModel):
    topic_id: str
    parent_chunk_id: str = ""  #unique identifier for the parent document
    child_chunk_id: str = ""   #the vector embedding ID that was matched
    parent_chunk_text: str
    source_file: str
    source_page: int
    relevance_score: float = 0.0  #similarity score from vector search
    contains_formula: bool = False # Assuming formulas will take extra time to learn


# for final markdown plan
class StudyBlock(BaseModel):
    date: str
    course: str
    topic: str
    duration_minutes: int
    action_items: list[str]
    source_references: list[str] = []

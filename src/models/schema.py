"""
Schema definitions for data models.
"""

from pydantic import BaseModel


class TopicNode(BaseModel):
    topic_id: str
    course: str
    raw_name: str
    keywords: list[str]
    required_depth: str = "conceptual"


class ParentChunk(BaseModel):
    parent_id: str
    source_file: str
    page_number: int
    text: str
    child_ids: list[str] = []


class ChildChunk(BaseModel):
    child_id: str
    parent_id: str
    text: str
    source_file: str
    page_number: int


class RetrievedContext(BaseModel):
    topic_id: str
    parent_chunk_text: str
    source_file: str
    source_page: int
    contains_formula: bool = False


class StudyBlock(BaseModel):
    date: str
    course: str
    topic: str
    duration_minutes: int
    action_items: list[str]
    source_references: list[str] = []

"""
Schema definitions for data models.
"""

from typing import Optional

from pydantic import BaseModel

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
    parent_chunk_text: str
    source_file: str
    source_page: int
    contains_formula: bool = False # Assuming formulas will take extra time to learn

# for final markdown plan
class StudyBlock(BaseModel):
    date: str
    course: str
    topic: str
    duration_minutes: int
    action_items: list[str] 
    source_references: list[str] = []

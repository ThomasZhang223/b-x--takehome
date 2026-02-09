"""
RAG client tool for retrieval-augmented generation.

Provides interface for querying the vector database and retrieving relevant context.
"""

import json
import logging
import re

from pinecone import Pinecone

from src.config import settings
from src.ingestion.embedder import embed_query
from src.models.schema import ParentChunk, RetrievedContext, TopicNode

logger = logging.getLogger(__name__)


class RAGClient:
    def __init__(self):
        pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index = pc.Index(settings.pinecone_index_name)

        with open("data/processed/parents.json") as f:
            parents_data = json.load(f)

        self.parents_dict: dict[str, ParentChunk] = {
            p["parent_id"]: ParentChunk(**p) for p in parents_data
        }

        logger.info(f"RAGClient initialized with {len(self.parents_dict)} parent chunks")

    def query(
        self,
        query_text: str,
        top_k: int = 8,
        exclude_ids: set[str] | None = None,
        exclude_parent_ids: set[str] | None = None,
    ) -> tuple[list[RetrievedContext], list[str], list[str]]:
        #Query Pinecone and return retrieved contexts with parent text.
        #Returns (contexts, new_chunk_ids, new_parent_ids) tuple.

        exclude_ids = exclude_ids or set()
        exclude_parent_ids = exclude_parent_ids or set()

        #fetch extra to account for filtering
        fetch_k = top_k * 3 if (exclude_ids or exclude_parent_ids) else top_k

        embedding = embed_query(query_text)

        results = self.index.query(
            vector=embedding,
            top_k=fetch_k,
            include_metadata=True,
        )

        contexts = []
        new_chunk_ids = []
        new_parent_ids = []
        seen_parents = set(exclude_parent_ids)  #start with already-excluded parents

        for match in results["matches"]:
            child_id = match["id"]
            parent_id = match["metadata"]["parent_id"]
            score = match.get("score", 0.0)

            #skip if child already retrieved
            if child_id in exclude_ids:
                continue

            #skip if parent already retrieved (avoid duplicate content)
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)

            parent = self.parents_dict.get(parent_id)
            if not parent:
                continue

            parent_text = parent.text
            # comprehensive regex to check for any latex for formulas
            has_formula = bool(
                re.search(r'(\\[a-zA-Z]+|\\[\^_{}]|\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\)|\\\[\\\[.*?\\\]\]|\\{.*?\\})', parent_text)
            )

            contexts.append(RetrievedContext(
                topic_id="",
                parent_chunk_id=parent_id,
                child_chunk_id=child_id,
                parent_chunk_text=parent_text,
                source_file=match["metadata"]["source_file"],
                source_page=match["metadata"]["page_number"],
                relevance_score=score,
                contains_formula=has_formula,
            ))
            new_chunk_ids.append(child_id)
            new_parent_ids.append(parent_id)

            #stop once we have enough
            if len(contexts) >= top_k:
                break

        return contexts, new_chunk_ids, new_parent_ids
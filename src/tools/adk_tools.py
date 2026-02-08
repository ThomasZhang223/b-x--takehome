"""
ADK-compatible function tools for the study planner agents.

These are plain Python functions that can be called by ADK agents.
For custom BaseAgent agents, we call tools directly (not via LLM).
"""

#lazy init singleton for RAGClient
_rag_client = None


def _get_rag_client():
    """Lazily initialize the RAGClient singleton."""
    global _rag_client
    if _rag_client is None:
        from src.tools.rag_client import RAGClient
        _rag_client = RAGClient()
    return _rag_client


def retrieve_topic_content(
    topic_id: str,
    query: str,
    top_k: int = 8,
    exclude_chunk_ids: set[str] | None = None,
    exclude_parent_ids: set[str] | None = None,
) -> dict:
    """Retrieves relevant textbook content for a study topic using RAG.

    Args:
        topic_id: The unique ID of the topic being researched.
        query: Search query combining topic name and keywords.
        top_k: Number of results to retrieve (higher = more thorough).
        exclude_chunk_ids: Set of child_chunk_ids to exclude (for deduplication).
        exclude_parent_ids: Set of parent_chunk_ids to exclude (avoid duplicate content).

    Returns:
        dict with 'topic_id', 'results' (list of retrieved context dicts),
        'count', 'new_chunk_ids', and 'new_parent_ids'.
    """
    client = _get_rag_client()
    results, new_chunk_ids, new_parent_ids = client.query(
        query,
        top_k=top_k,
        exclude_ids=exclude_chunk_ids,
        exclude_parent_ids=exclude_parent_ids,
    )

    #set topic_id on each result
    for rc in results:
        rc.topic_id = topic_id

    return {
        "topic_id": topic_id,
        "results": [rc.model_dump() for rc in results],
        "count": len(results),
        "new_chunk_ids": new_chunk_ids,
        "new_parent_ids": new_parent_ids,
    }


def validate_topic_coverage(topic_keywords: list, retrieved_texts: list) -> dict:
    """Validates that retrieved content covers the expected topic keywords.

    Args:
        topic_keywords: List of expected keywords from the syllabus.
        retrieved_texts: List of retrieved parent chunk text strings.

    Returns:
        dict with 'coverage' (float 0-1), 'found' (list), 'missing' (list), 'passed' (bool).
    """
    #combine all retrieved texts into one lowercase string
    combined_text = " ".join(retrieved_texts).lower()

    found = []
    missing = []

    for keyword in topic_keywords:
        if keyword.lower() in combined_text:
            found.append(keyword)
        else:
            missing.append(keyword)

    total = len(topic_keywords)
    coverage = len(found) / total if total > 0 else 0.0
    passed = coverage >= 0.5  #50% threshold is fine for direct matching

    return {
        "coverage": coverage,
        "found": found,
        "missing": missing,
        "passed": passed,
    }

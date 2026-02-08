"""
Researcher Agent — retrieves textbook content for each topic via RAG.

This is the "generator" in the generator-critic loop.
Preserves state across loop iterations to support retry logic.
"""

import logging
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

from src.models.schema import TopicNode
from src.tools.adk_tools import retrieve_topic_content


class ResearcherAgent(BaseAgent):
    """Researcher Agent — retrieves textbook content for each topic via RAG.

    Reads from state: topics, course_configs, retry_topic_ids, validation_iteration
    Writes to state: retrieved_contexts, retrieved_chunk_ids, retrieved_parent_ids, topic_embeddings_cache
    """

    model_config = {"extra": "allow"}

    def __init__(self):
        super().__init__(name="ResearcherAgent")
        self.logger = logging.getLogger("agent.Researcher")

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator:
        topics = [TopicNode(**t) for t in ctx.session.state.get("topics", [])]
        course_configs = ctx.session.state.get("course_configs", {})

        #load existing state (don't reset on retry!)
        all_contexts = ctx.session.state.get("retrieved_contexts", [])
        retrieved_chunk_ids = set(ctx.session.state.get("retrieved_chunk_ids", []))
        retrieved_parent_ids = set(ctx.session.state.get("retrieved_parent_ids", []))
        topic_embeddings_cache = ctx.session.state.get("topic_embeddings_cache", {})

        #check if this is a retry pass
        iteration = ctx.session.state.get("validation_iteration", 0)
        retry_topic_ids = set(ctx.session.state.get("retry_topic_ids", []))
        validation_gaps = ctx.session.state.get("validation_gaps", [])

        print()
        print("=" * 60)
        if iteration == 0:
            print("  Researcher Agent: Retrieving content...")
        else:
            print(f"  Researcher Agent: Retry pass {iteration + 1}...")
            if retry_topic_ids:
                print(f"  Retrying {len(retry_topic_ids)} topics with individual keyword searches")
        print("=" * 60)
        print()

        new_this_pass = 0

        for topic in topics:
            #on retry, only research topics flagged for retry
            if iteration > 0 and topic.topic_id not in retry_topic_ids:
                continue

            #check if topic already has content (skip if not a retry target)
            if iteration == 0:
                existing_ctxs = [c for c in all_contexts if c.get("topic_id") == topic.topic_id]
                if existing_ctxs:
                    print(f"  [SKIP] {topic.raw_name}: already has {len(existing_ctxs)} chunks")
                    continue

            cc = course_configs.get(topic.course, {})
            depth = cc.get("depth", "moderate")

            #map depth to top_k (fetch more on retry)
            top_k_map = {"deep_dive": 12, "moderate": 8, "light_review": 5}
            base_top_k = top_k_map.get(depth, 8)
            
            if iteration > 0:
                # KEYWORD INDIVIDUALIZATION:
                # Find the gap entry for this topic
                gap = next((g for g in validation_gaps if g["topic_id"] == topic.topic_id), None)
                missing_keywords = gap["missing_keywords"] if gap else topic.keywords
                
                print(f"  [RETRY] {topic.raw_name}: Searching for {len(missing_keywords)} missing keywords...")
                
                topic_new_count = 0
                for kw in missing_keywords:
                    # Construct specific query for this keyword
                    kw_query = f"{topic.raw_name} {kw}"
                    result = retrieve_topic_content(
                        topic.topic_id,
                        kw_query,
                        top_k=3, # fetch a few focused results for each keyword
                        exclude_chunk_ids=retrieved_chunk_ids,
                        exclude_parent_ids=retrieved_parent_ids,
                    )
                    
                    if result["count"] > 0:
                        all_contexts.extend(result["results"])
                        new_chunk_ids = result.get("new_chunk_ids", [])
                        new_parent_ids = result.get("new_parent_ids", [])
                        retrieved_chunk_ids.update(new_chunk_ids)
                        retrieved_parent_ids.update(new_parent_ids)
                        topic_embeddings_cache.setdefault(topic.topic_id, []).extend(new_chunk_ids)
                        topic_new_count += result["count"]
                
                new_this_pass += topic_new_count
                print(f"  [OK]    {topic.raw_name}: {topic_new_count} new chunks found via keyword search")

            else:
                # Standard pass (Iteration 0)
                query = f"{topic.raw_name} {' '.join(topic.keywords)}"
                result = retrieve_topic_content(
                    topic.topic_id,
                    query,
                    top_k=base_top_k,
                    exclude_chunk_ids=retrieved_chunk_ids,
                    exclude_parent_ids=retrieved_parent_ids,
                )

                if result["count"] > 0:
                    all_contexts.extend(result["results"])
                    new_chunk_ids = result.get("new_chunk_ids", [])
                    new_parent_ids = result.get("new_parent_ids", [])
                    retrieved_chunk_ids.update(new_chunk_ids)
                    retrieved_parent_ids.update(new_parent_ids)
                    topic_embeddings_cache.setdefault(topic.topic_id, []).extend(new_chunk_ids)
                    new_this_pass += result["count"]
                    print(f"  [OK] {topic.raw_name}: {result['count']} chunks")
                else:
                    print(f"  [--] {topic.raw_name}: No new results")

        #save state
        ctx.session.state["retrieved_contexts"] = all_contexts
        ctx.session.state["retrieved_chunk_ids"] = list(retrieved_chunk_ids)
        ctx.session.state["retrieved_parent_ids"] = list(retrieved_parent_ids)
        ctx.session.state["topic_embeddings_cache"] = topic_embeddings_cache

        #clear retry flags (validator will set new ones if needed)
        ctx.session.state["retry_topic_ids"] = []

        print()
        print(f"  This pass: {new_this_pass} new chunks")
        print(f"  Total: {len(all_contexts)} context chunks")

        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part(text=f"Research complete. {len(all_contexts)} total chunks ({new_this_pass} new).")]
            )
        )

"""
Estimator Agent — calculates study time and manages content dynamically.

Uses natural language interface for user interactions.
Supports adding/reducing content, deleting topics, and adjusting time estimates.
"""

import logging
from typing import AsyncGenerator

from google import genai
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types
from pydantic import BaseModel

from src.config import settings
from src.models.schema import StudyBlock, TopicNode
from src.tools.adk_tools import retrieve_topic_content


#intent parsing response schema
class ParsedIntent(BaseModel):
    intent: str  #add_content, reduce_content, delete_topic, adjust_time, accept, clarify
    topic_id: str | None = None
    topic_name: str | None = None
    chunk_count: int | None = None
    minutes: int | None = None
    message: str | None = None  #for clarify intent


class EstimatorAgent(BaseAgent):
    """Estimator Agent — HITL 2 for time estimates and content management.

    Uses natural language to understand user requests for:
    - Adding more content for a topic
    - Reducing content for a topic
    - Deleting topics entirely
    - Adjusting time estimates

    Reads from state: topics, retrieved_contexts, course_configs
    Writes to state: study_blocks, retrieved_contexts, retrieved_chunk_ids, topic_embeddings_cache
    """

    model_config = {"extra": "allow"}

    def __init__(self):
        super().__init__(name="EstimatorAgent")
        self.logger = logging.getLogger("agent.Estimator")
        self.client = genai.Client(api_key=settings.embedding_gemini_api_key)

    def _parse_intent(self, user_input: str, topics: list[TopicNode]) -> ParsedIntent:
        """Use LLM to parse natural language input into structured intent."""

        #build topic list for the prompt
        topic_list = "\n".join([
            f"- ID: {t.topic_id}, Name: {t.raw_name}, Course: {t.course}"
            for t in topics
        ])

        prompt = f"""You are an intent parser for a study planning system. Parse the user's request and return JSON.

Available topics:
{topic_list}

User message: "{user_input}"

Return JSON with:
{{
  "intent": "add_content" | "reduce_content" | "delete_topic" | "adjust_time" | "accept" | "clarify",
  "topic_id": "matched_topic_id" or null,
  "topic_name": "matched_topic_name" or null,
  "chunk_count": number or null (for add_content, default 5),
  "minutes": number or null (for adjust_time),
  "message": "clarification message" or null (only for clarify intent)
}}

Rules:
- Match topic names fuzzy (e.g., "quantum" matches "Quantum Mechanics")
- If user says "looks good", "accept", "done", "proceed", or just presses enter → intent: "accept"
- If user wants "more" content without specifying count → chunk_count: 5
- If user wants to "reduce" or "cut" content without specifying → reduce by half (no chunk_count needed)
- If ambiguous or no topic match found → intent: "clarify" with helpful message
- For time adjustments, extract the minutes value"""

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": ParsedIntent,
                },
            )
            return ParsedIntent.model_validate_json(response.text)
        except Exception as e:
            self.logger.error(f"Intent parsing failed: {e}")
            return ParsedIntent(intent="clarify", message=f"Sorry, I couldn't understand that. Error: {str(e)}")

    def _calculate_estimates(
        self,
        topics: list[TopicNode],
        contexts: list[dict],
        course_configs: dict,
    ) -> dict[str, dict]:
        """Calculate time estimates per topic based on content and depth."""

        estimates = {}

        for topic in topics:
            topic_ctxs = [c for c in contexts if c.get("topic_id") == topic.topic_id]
            chunk_count = len(topic_ctxs)

            #base time per chunk depends on depth
            cc = course_configs.get(topic.course, {})
            depth = cc.get("depth", "moderate")
            base_minutes_per_chunk = {"deep_dive": 8, "moderate": 5, "light_review": 3}.get(depth, 5)

            #formula penalty: add 50% more time for formula-heavy content
            formula_count = sum(1 for c in topic_ctxs if c.get("contains_formula", False))
            formula_multiplier = 1.0 + (0.5 * formula_count / max(chunk_count, 1))

            total_minutes = int(chunk_count * base_minutes_per_chunk * formula_multiplier * 30)
            total_minutes = max(total_minutes, 30)  

            estimates[topic.topic_id] = {
                "topic": topic,
                "chunk_count": chunk_count,
                "formula_count": formula_count,
                "minutes": total_minutes,
                "depth": depth,
            }

        return estimates

    def _display_estimates(self, estimates: dict, course_configs: dict, courses: list[str]):
        """Display formatted estimates to the user."""

        print()
        print("=" * 65)
        print("  Study Time Estimates")
        print("=" * 65)
        print()

        for course in courses:
            cc = course_configs.get(course, {})
            depth = cc.get("depth", "moderate")
            days = cc.get("days_until_midterm", "?")

            print(f"  {course} ({depth}, {days} days)")
            print("  " + "-" * 60)

            course_estimates = [
                e for e in estimates.values()
                if e["topic"].course == course
            ]

            for est in course_estimates:
                topic = est["topic"]
                chunks = est["chunk_count"]
                mins = est["minutes"]
                formula_tag = " [formula-heavy]" if est["formula_count"] > 0 else ""

                print(f"    {topic.raw_name:<35} {mins:>3} min ({chunks} chunks){formula_tag}")

            course_total = sum(e["minutes"] for e in course_estimates)
            print(f"    {'':35} -----")
            print(f"    {'Course total:':<35} {course_total:>3} min")
            print()

        total = sum(e["minutes"] for e in estimates.values())
        print(f"  {'TOTAL STUDY TIME:':<40} {total} min ({total // 60}h {total % 60}m)")
        print()

    def _build_study_blocks(self, estimates: dict, contexts: list[dict]) -> list[dict]:
        """Convert estimates to StudyBlock objects."""

        blocks = []
        for topic_id, est in estimates.items():
            topic = est["topic"]

            # Collect source references
            topic_ctxs = [c for c in contexts if c.get("topic_id") == topic.topic_id]
            refs = sorted(list(set(
                f"{c.get('source_file')} (Page {c.get('source_page')})"
                for c in topic_ctxs if c.get("source_file") and c.get("source_page") is not None
            )))

            #generate action items based on content
            action_items = [f"Review {topic.raw_name} concepts"]
            if est["formula_count"] > 0:
                action_items.append("Practice formula derivations and applications")
            action_items.append(f"Keywords: {', '.join(topic.keywords[:5])}")

            block = StudyBlock(
                date="",  #planner will assign dates
                course=topic.course,
                topic=topic.raw_name,
                duration_minutes=est["minutes"],
                action_items=action_items,
                source_references=refs,
            )
            blocks.append(block.model_dump())

        return blocks

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator:
        #load state
        topics = [TopicNode(**t) for t in ctx.session.state.get("topics", [])]
        contexts = ctx.session.state.get("retrieved_contexts", [])
        course_configs = ctx.session.state.get("course_configs", {})
        courses = ctx.session.state.get("courses", [])

        #load or initialize deduplication tracking
        retrieved_chunk_ids = set(ctx.session.state.get("retrieved_chunk_ids", []))
        retrieved_parent_ids = set(ctx.session.state.get("retrieved_parent_ids", []))
        topic_embeddings_cache = ctx.session.state.get("topic_embeddings_cache", {})

        #initialize cache from existing contexts if empty
        if not topic_embeddings_cache:
            for ctx_item in contexts:
                tid = ctx_item.get("topic_id", "")
                cid = ctx_item.get("child_chunk_id", "")
                pid = ctx_item.get("parent_chunk_id", "")
                if tid and cid:
                    topic_embeddings_cache.setdefault(tid, []).append(cid)
                    retrieved_chunk_ids.add(cid)
                if pid:
                    retrieved_parent_ids.add(pid)

        print()
        print("=" * 65)
        print("  Estimator Agent: Calculating study time...")
        print("=" * 65)

        while True:
            #calculate and display estimates
            estimates = self._calculate_estimates(topics, contexts, course_configs)
            self._display_estimates(estimates, course_configs, courses)

            #prompt user
            print("-" * 65)
            print("  What would you like to adjust?")
            print("  Examples: 'get 5 more chunks for Quantum', 'remove Thermodynamics',")
            print("            'change Calculus to 60 minutes', 'reduce content for Physics'")
            print("  Or just press Enter / say 'looks good' to proceed.")
            print("-" * 65)
            print()

            user_input = input("  > ").strip()

            #empty input = accept
            if not user_input:
                user_input = "accept"

            #parse intent
            parsed = self._parse_intent(user_input, topics)

            if parsed.intent == "accept":
                print("\n. Estimates accepted. Building study blocks...")
                break

            elif parsed.intent == "clarify":
                print(f"\n  {parsed.message}")
                print(f"  Available topics: {', '.join(t.raw_name for t in topics)}")
                continue

            elif parsed.intent == "add_content":
                if not parsed.topic_id:
                    print("\n  Could not identify which topic. Please try again.")
                    continue

                topic = next((t for t in topics if t.topic_id == parsed.topic_id), None)
                if not topic:
                    print(f"\n  Topic '{parsed.topic_name}' not found.")
                    continue

                chunk_count = parsed.chunk_count or 5
                print(f"\n  Retrieving {chunk_count} more chunks for {topic.raw_name}...")

                query = f"{topic.raw_name} {' '.join(topic.keywords)}"
                result = retrieve_topic_content(
                    topic.topic_id,
                    query,
                    top_k=chunk_count,
                    exclude_chunk_ids=retrieved_chunk_ids,
                    exclude_parent_ids=retrieved_parent_ids,
                )

                if result["count"] > 0:
                    contexts.extend(result["results"])
                    new_chunk_ids = result.get("new_chunk_ids", [])
                    new_parent_ids = result.get("new_parent_ids", [])
                    retrieved_chunk_ids.update(new_chunk_ids)
                    retrieved_parent_ids.update(new_parent_ids)
                    topic_embeddings_cache.setdefault(topic.topic_id, []).extend(new_chunk_ids)
                    print(f"  Added {result['count']} new chunks for {topic.raw_name}")
                else:
                    print(f"  No new content found (all results were duplicates)")

            elif parsed.intent == "reduce_content":
                if not parsed.topic_id:
                    print("\n  Could not identify which topic. Please try again.")
                    continue

                topic = next((t for t in topics if t.topic_id == parsed.topic_id), None)
                if not topic:
                    print(f"\n  Topic '{parsed.topic_name}' not found.")
                    continue

                #find contexts for this topic
                topic_ctxs = [c for c in contexts if c.get("topic_id") == topic.topic_id]
                if len(topic_ctxs) <= 2:
                    print(f"\n  {topic.raw_name} only has {len(topic_ctxs)} chunks, can't reduce further.")
                    continue

                #keep first half (most relevant), remove second half
                keep_count = len(topic_ctxs) // 2
                keep_count = max(keep_count, 2)  #keep at least 2

                #remove the latter contexts
                remove_ctxs = topic_ctxs[keep_count:]
                removed_chunk_ids = [c.get("child_chunk_id") for c in remove_ctxs if c.get("child_chunk_id")]
                removed_parent_ids = [c.get("parent_chunk_id") for c in remove_ctxs if c.get("parent_chunk_id")]

                contexts = [c for c in contexts if c not in remove_ctxs]
                retrieved_chunk_ids -= set(removed_chunk_ids)
                retrieved_parent_ids -= set(removed_parent_ids)

                #update cache
                if topic.topic_id in topic_embeddings_cache:
                    topic_embeddings_cache[topic.topic_id] = [
                        cid for cid in topic_embeddings_cache[topic.topic_id]
                        if cid not in removed_chunk_ids
                    ]

                print(f"  Reduced {topic.raw_name} from {len(topic_ctxs)} to {keep_count} chunks")

            elif parsed.intent == "delete_topic":
                if not parsed.topic_id:
                    print("\n  Could not identify which topic. Please try again.")
                    continue

                topic = next((t for t in topics if t.topic_id == parsed.topic_id), None)
                if not topic:
                    print(f"\n  Topic '{parsed.topic_name}' not found.")
                    continue

                #remove topic
                topics = [t for t in topics if t.topic_id != topic.topic_id]

                #remove contexts
                topic_ctxs = [c for c in contexts if c.get("topic_id") == topic.topic_id]
                contexts = [c for c in contexts if c.get("topic_id") != topic.topic_id]

                #clean up chunk and parent tracking
                removed_chunk_ids = [c.get("child_chunk_id") for c in topic_ctxs if c.get("child_chunk_id")]
                removed_parent_ids = [c.get("parent_chunk_id") for c in topic_ctxs if c.get("parent_chunk_id")]
                retrieved_chunk_ids -= set(removed_chunk_ids)
                retrieved_parent_ids -= set(removed_parent_ids)

                if topic.topic_id in topic_embeddings_cache:
                    del topic_embeddings_cache[topic.topic_id]

                print(f"  Removed topic '{topic.raw_name}' and {len(topic_ctxs)} associated chunks")

            elif parsed.intent == "adjust_time":
                if not parsed.topic_id:
                    print("\n  Could not identify which topic. Please try again.")
                    continue

                if not parsed.minutes:
                    print("\n  Could not determine the new time. Please specify minutes.")
                    continue

                topic = next((t for t in topics if t.topic_id == parsed.topic_id), None)
                if not topic:
                    print(f"\n  Topic '{parsed.topic_name}' not found.")
                    continue

                #store manual override in estimates (will be applied after recalc)
                #for now, just show message - actual override happens in block building
                print(f"  Will set {topic.raw_name} to {parsed.minutes} minutes")

                #add a manual time override to the context metadata
                for c in contexts:
                    if c.get("topic_id") == topic.topic_id:
                        c["manual_time_override"] = parsed.minutes
                        break

        #build study blocks from final estimates
        estimates = self._calculate_estimates(topics, contexts, course_configs)

        #apply manual time overrides
        for ctx_item in contexts:
            override = ctx_item.get("manual_time_override")
            if override and ctx_item.get("topic_id") in estimates:
                estimates[ctx_item["topic_id"]]["minutes"] = override

        study_blocks = self._build_study_blocks(estimates, contexts)

        #save state
        ctx.session.state["topics"] = [t.model_dump() for t in topics]
        ctx.session.state["retrieved_contexts"] = contexts
        ctx.session.state["retrieved_chunk_ids"] = list(retrieved_chunk_ids)
        ctx.session.state["retrieved_parent_ids"] = list(retrieved_parent_ids)
        ctx.session.state["topic_embeddings_cache"] = topic_embeddings_cache
        ctx.session.state["study_blocks"] = study_blocks

        print(f"\n  Created {len(study_blocks)} study blocks")
        print(f"  Total time: {sum(b['duration_minutes'] for b in study_blocks)} minutes")

        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part(text=f"Estimation complete. {len(study_blocks)} study blocks created.")]
            )
        )

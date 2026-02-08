"""
Validator Agent — checks keyword coverage of retrieved content.

This is the "critic" in the generator-critic loop.
If validation fails badly on first pass, flags topics for retry.
If validation passes (or is acceptable), escalates to exit the loop.
"""

import logging
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from src.models.schema import TopicNode
from src.tools.adk_tools import validate_topic_coverage

#threshold for triggering retry (below this = needs more content)
RETRY_COVERAGE_THRESHOLD = 0.4  #40% coverage is too low


class ValidatorAgent(BaseAgent):
    """Validator Agent — checks keyword coverage of retrieved content.

    Generator-Critic pattern: If coverage is acceptable, escalates to exit the LoopAgent.
    If coverage is poor on first pass, flags topics for retry and allows loop to continue.

    Reads from state: topics, retrieved_contexts, validation_iteration
    Writes to state: validation_gaps, validation_passed, retry_topic_ids, validation_iteration
    """

    model_config = {"extra": "allow"}

    def __init__(self):
        super().__init__(name="ValidatorAgent")
        self.logger = logging.getLogger("agent.Validator")

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator:
        topics = [TopicNode(**t) for t in ctx.session.state.get("topics", [])]
        contexts = ctx.session.state.get("retrieved_contexts", [])
        iteration = ctx.session.state.get("validation_iteration", 0)

        print()
        print("=" * 60)
        print(f"  Validator Agent: Checking coverage (pass {iteration + 1})...")
        print("=" * 60)
        print()

        gaps = []
        retry_topic_ids = []
        fully_covered = 0

        for topic in topics:
            topic_ctxs = [c for c in contexts if c.get("topic_id") == topic.topic_id]
            texts = [c.get("parent_chunk_text", "") for c in topic_ctxs]

            if not texts:
                gaps.append({
                    "topic_id": topic.topic_id,
                    "topic_name": topic.raw_name,
                    "course": topic.course,
                    "missing_keywords": topic.keywords,
                    "coverage": 0.0
                })
                retry_topic_ids.append(topic.topic_id)
                print(f"  [MISS] {topic.raw_name}: NO content retrieved")

                #add placeholder context so planner has something to work with
                contexts.append({
                    "topic_id": topic.topic_id,
                    "parent_chunk_text": f"[No textbook content found -- manual review needed: {topic.raw_name}]",
                    "source_file": "N/A",
                    "source_page": 0,
                    "contains_formula": False
                })
                continue

            result = validate_topic_coverage(topic.keywords, texts)

            if result["missing"]:
                gaps.append({
                    "topic_id": topic.topic_id,
                    "topic_name": topic.raw_name,
                    "course": topic.course,
                    "missing_keywords": result["missing"],
                    "coverage": result["coverage"]
                })

                #flag for retry if coverage is below threshold
                if result["coverage"] < RETRY_COVERAGE_THRESHOLD:
                    retry_topic_ids.append(topic.topic_id)
                    print(f"  [GAP!] {topic.raw_name}: {result['coverage']:.0%} -- NEEDS RETRY")
                else:
                    print(f"  [GAP]  {topic.raw_name}: {result['coverage']:.0%} -- missing: {', '.join(result['missing'][:3])}")
            else:
                fully_covered += 1
                print(f"  [OK]   {topic.raw_name}: full keyword coverage")

        #save state
        ctx.session.state["validation_gaps"] = gaps
        ctx.session.state["retrieved_contexts"] = contexts
        ctx.session.state["validation_iteration"] = iteration + 1
        ctx.session.state["retry_topic_ids"] = retry_topic_ids

        total = len(topics)
        print()
        print(f"  Coverage: {fully_covered}/{total} fully covered, {len(gaps)} with gaps")

        #generator-critic exit logic:
        #- first pass with low-coverage topics: allow retry (don't escalate)
        #- second pass or acceptable coverage: exit loop
        should_retry = iteration == 0 and len(retry_topic_ids) > 0

        if should_retry:
            print(f"  → {len(retry_topic_ids)} topics need retry, allowing another pass...")
            ctx.session.state["validation_passed"] = False

            yield Event(
                author=self.name,
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=f"Validation found {len(retry_topic_ids)} low-coverage topics. Retrying...")]
                )
            )
            #don't escalate — let LoopAgent continue
        else:
            ctx.session.state["validation_passed"] = True
            if iteration > 0:
                print(f"  → Retry complete, proceeding with current coverage")
            else:
                print(f"  → Coverage acceptable, proceeding")

            yield Event(
                author=self.name,
                actions=EventActions(escalate=True)  #signal LoopAgent to exit
            )

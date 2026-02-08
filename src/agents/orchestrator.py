"""
Orchestrator Agent for HITL 1 — collects user preferences via CLI.

Uses natural language processing for conversational configuration.
"""

import json
import logging
from typing import AsyncGenerator

from google import genai
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types
from pydantic import BaseModel

from src.config import settings
from src.models.schema import TopicNode


#intent parsing response schema
class ParsedConfigIntent(BaseModel):
    intent: str  #set_depth, set_days, set_both, set_all, remove_course, accept, clarify
    course: str | None = None  #matched course name
    depth: str | None = None  #deep_dive, moderate, light_review
    days: int | None = None
    message: str | None = None  #for clarify intent


class OrchestratorAgent(BaseAgent):
    """HITL Orchestrator Agent — collects user preferences via CLI.

    Uses natural language to understand user preferences for:
    - Setting study depth per course
    - Setting days until midterm per course
    - Removing courses from the plan

    Reads parsed syllabi, presents courses, collects depth and timeline per course.
    Writes to session.state: courses, course_configs, topics.
    """

    model_config = {"extra": "allow"}

    def __init__(self):
        super().__init__(name="OrchestratorAgent")
        self.logger = logging.getLogger("agent.OrchestratorAgent")
        self.client = genai.Client(api_key=settings.retrieval_gemini_api_key)

    def _parse_intent(self, user_input: str, courses: list[str], current_configs: dict) -> ParsedConfigIntent:
        """Use LLM to parse natural language input into structured intent."""

        #build current config summary for context
        config_summary = "\n".join([
            f"- {course}: depth={cc.get('depth', 'not set')}, days={cc.get('days_until_midterm', 'not set')}"
            for course, cc in current_configs.items()
        ]) if current_configs else "No courses configured yet."

        prompt = f"""You are an intent parser for a study planning configuration system. Parse the user's request and return JSON.

Available courses:
{', '.join(courses)}

Current configuration:
{config_summary}

User message: "{user_input}"

Return JSON with:
{{
  "intent": "set_depth" | "set_days" | "set_both" | "set_all" | "remove_course" | "accept" | "clarify",
  "course": "matched course name" or null,
  "depth": "deep_dive" | "moderate" | "light_review" or null,
  "days": number or null,
  "message": "clarification message" or null (only for clarify intent)
}}

Rules:
- Match course names fuzzy (e.g., "physics" matches "PHYS 234", "health" matches "HLTH 204", "systems" matches "SYSD 300")
- "set_depth": user wants to change study depth for a course
- "set_days": user wants to set days until midterm for a course
- "set_both": user provides both depth and days in one message
- "set_all": user wants to apply same settings to all courses (e.g., "set all to moderate")
- "remove_course": user wants to skip/remove a course from the plan
- "accept": user says "done", "looks good", "proceed", "that's it", "confirm", or presses enter
- "clarify": ambiguous input or missing information

Depth mapping:
- "deep", "thorough", "focus", "hard", "difficult", "struggling" → deep_dive
- "normal", "regular", "standard", "okay", "fine" → moderate
- "light", "quick", "easy", "review", "refresh", "skim" → light_review

Examples:
- "Physics is my hardest, 5 days" → set_both, course=PHYS 234, depth=deep_dive, days=5
- "I need to focus on health" → set_depth, course=HLTH 204, depth=deep_dive
- "Systems exam is in 10 days" → set_days, course=SYSD 300, days=10
- "Skip physics" → remove_course, course=PHYS 234
- "All courses moderate, 7 days each" → set_all, depth=moderate, days=7"""

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": ParsedConfigIntent,
                },
            )
            return ParsedConfigIntent.model_validate_json(response.text)
        except Exception as e:
            self.logger.error(f"Intent parsing failed: {e}")
            return ParsedConfigIntent(intent="clarify", message=f"Sorry, I couldn't understand that. Error: {str(e)}")

    def _display_config(self, courses: list[str], course_configs: dict, topic_counts: dict):
        """Display current configuration state."""

        print()
        print("=" * 65)
        print("  Current Configuration")
        print("=" * 65)
        print()

        for course in courses:
            cc = course_configs.get(course, {})
            depth = cc.get("depth", "---")
            days = cc.get("days_until_midterm", "---")
            topics = topic_counts.get(course, 0)

            status = "✓" if depth != "---" and days != "---" else "○"
            print(f"  {status} {course:<12} | {depth:<12} | {days if days != '---' else '---':>3} days | {topics} topics")

        print()

        #check if all configured
        all_configured = all(
            course_configs.get(c, {}).get("depth") and course_configs.get(c, {}).get("days_until_midterm")
            for c in courses
        )

        if all_configured:
            print("  All courses configured! Say 'done' to proceed or adjust further.")
        else:
            unconfigured = [c for c in courses if not (course_configs.get(c, {}).get("depth") and course_configs.get(c, {}).get("days_until_midterm"))]
            print(f"  Still need: {', '.join(unconfigured)}")

        print()

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator:
        """ADK entry point. Yields events."""

        #load parsed topics from artifact store
        syllabi_path = "data/artifact_store/syllabi.json"
        with open(syllabi_path) as f:
            syllabi_data = json.load(f)

        courses = list(syllabi_data.keys())
        all_topics = []
        topic_counts = {}

        for course, topic_list in syllabi_data.items():
            topic_counts[course] = len(topic_list)
            for t in topic_list:
                all_topics.append(TopicNode(**t) if isinstance(t, dict) else t)

        #display header
        print("=" * 65)
        print("  EXAM STUDY PLANNER -- Configuration")
        print("=" * 65)
        print()
        print("  I found these courses in your syllabi:")
        for course in courses:
            print(f"    • {course}: {topic_counts[course]} topics")
        print()
        print("-" * 65)
        print("  Tell me about each course in natural language!")
        print("  Examples:")
        print("    'Physics is my hardest exam, 5 days away'")
        print("    'Light review for health, 10 days'")
        print("    'All courses moderate with 7 days'")
        print("    'Skip systems dynamics'")
        print("-" * 65)

        course_configs = {}

        while True:
            self._display_config(courses, course_configs, topic_counts)

            print("-" * 65)
            user_input = input("  > ").strip()

            #empty input with all configured = accept
            if not user_input:
                all_configured = all(
                    course_configs.get(c, {}).get("depth") and course_configs.get(c, {}).get("days_until_midterm")
                    for c in courses
                )
                if all_configured:
                    user_input = "accept"
                else:
                    print("\n  Please configure all courses first, or say 'skip [course]' to remove one.")
                    continue

            #parse intent
            parsed = self._parse_intent(user_input, courses, course_configs)

            if parsed.intent == "accept":
                #verify all configured
                all_configured = all(
                    course_configs.get(c, {}).get("depth") and course_configs.get(c, {}).get("days_until_midterm")
                    for c in courses
                )
                if not all_configured:
                    unconfigured = [c for c in courses if not (course_configs.get(c, {}).get("depth") and course_configs.get(c, {}).get("days_until_midterm"))]
                    print(f"\n  ⚠ Still need to configure: {', '.join(unconfigured)}")
                    continue

                print("\n  Configuration complete!")
                break

            elif parsed.intent == "clarify":
                print(f"\n  {parsed.message}")
                print(f"  Available courses: {', '.join(courses)}")
                continue

            elif parsed.intent == "set_all":
                depth = parsed.depth or "moderate"
                days = parsed.days

                if not days:
                    print("\n  Please specify days until midterm for all courses.")
                    continue

                for course in courses:
                    if course not in course_configs:
                        course_configs[course] = {}
                    course_configs[course]["depth"] = depth
                    course_configs[course]["days_until_midterm"] = days
                    course_configs[course]["course"] = course

                print(f"\n  Set all {len(courses)} courses to {depth}, {days} days")

            elif parsed.intent == "remove_course":
                if not parsed.course or parsed.course not in courses:
                    print(f"\n  Course not found. Available: {', '.join(courses)}")
                    continue

                courses.remove(parsed.course)
                if parsed.course in course_configs:
                    del course_configs[parsed.course]

                #remove topics for this course
                all_topics = [t for t in all_topics if (t.course if isinstance(t, TopicNode) else t["course"]) != parsed.course]
                del topic_counts[parsed.course]

                print(f"\n  Removed {parsed.course} from the study plan")

            elif parsed.intent in ("set_depth", "set_days", "set_both"):
                if not parsed.course or parsed.course not in courses:
                    print(f"\n  Course not found. Available: {', '.join(courses)}")
                    continue

                if parsed.course not in course_configs:
                    course_configs[parsed.course] = {"course": parsed.course}

                updates = []
                if parsed.depth:
                    course_configs[parsed.course]["depth"] = parsed.depth
                    updates.append(f"depth={parsed.depth}")
                if parsed.days:
                    course_configs[parsed.course]["days_until_midterm"] = parsed.days
                    updates.append(f"{parsed.days} days")

                if updates:
                    print(f"\n  {parsed.course}: {', '.join(updates)}")
                else:
                    print(f"\n  No changes detected for {parsed.course}")

        #update topic depths based on course config
        updated_topics = []
        for topic in all_topics:
            t = topic if isinstance(topic, TopicNode) else TopicNode(**topic)
            cc = course_configs.get(t.course, {})
            t.required_depth = cc.get("depth", "moderate")
            updated_topics.append(t)

        #print final summary
        print()
        print("=" * 65)
        print("  Final Configuration")
        print("=" * 65)
        total_topics = 0
        for course in courses:
            cc = course_configs.get(course, {})
            count = topic_counts.get(course, 0)
            total_topics += count
            print(f"  {course}: {cc.get('depth')} | {cc.get('days_until_midterm')} days | {count} topics")
        print()
        print(f"  Total: {len(courses)} courses, {total_topics} topics")
        print()

        #write to shared session state
        ctx.session.state["courses"] = courses
        ctx.session.state["course_configs"] = course_configs
        ctx.session.state["topics"] = [t.model_dump() if isinstance(t, TopicNode) else t for t in updated_topics]

        #yield a completion event
        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part(text=f"Configuration complete. {len(courses)} courses, {total_topics} topics loaded.")]
            )
        )

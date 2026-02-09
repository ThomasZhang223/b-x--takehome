"""
Planner Agent — bin-packing scheduler, outputs study_plan.md.

Schedules study blocks across available days and generates markdown output.
"""

import datetime
import logging
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

from src.models.schema import StudyBlock


class PlannerAgent(BaseAgent):
    """Planner Agent — bin-packing scheduler, outputs study_plan.md.

    Reads from state: study_blocks, course_configs, courses, validation_gaps
    Writes to state: output_path
    """

    model_config = {"extra": "allow"}

    def __init__(self):
        super().__init__(name="PlannerAgent")
        self.logger = logging.getLogger("agent.Planner")

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator:
        blocks = [StudyBlock(**sb) for sb in ctx.session.state.get("study_blocks", [])]
        course_configs = ctx.session.state.get("course_configs", {})
        courses = ctx.session.state.get("courses", [])
        validation_gaps = ctx.session.state.get("validation_gaps", [])

        print()
        print("=" * 60)
        print("  Planner Agent: Building day-by-day schedule...")
        print("=" * 60)
        print()

        today = datetime.date.today()
        max_days = max(cc["days_until_midterm"] for cc in course_configs.values())

        #sort: nearest midterm first, then longest duration first
        sorted_blocks = sorted(blocks, key=lambda sb: (
            course_configs.get(sb.course, {}).get("days_until_midterm", 999),
            -sb.duration_minutes
        ))

        #bin-packing into daily 8-hour slots
        schedule = {}  #date_str -> [StudyBlock]

        for block in sorted_blocks:
            cc = course_configs.get(block.course, {})
            midterm_day = today + datetime.timedelta(days=cc.get("days_until_midterm", max_days))

            assigned = False
            for day_offset in range(max_days):
                date = today + datetime.timedelta(days=day_offset)
                if date >= midterm_day:
                    break

                date_str = date.isoformat()
                day_blocks = schedule.get(date_str, [])
                day_total = sum(b.duration_minutes for b in day_blocks)

                if day_total + block.duration_minutes <= 480:  
                    block.date = date_str
                    schedule.setdefault(date_str, []).append(block)
                    assigned = True
                    break

            if not assigned:
                #overflow: assign to last day before midterm
                last_date = midterm_day - datetime.timedelta(days=1)
                block.date = last_date.isoformat()
                schedule.setdefault(last_date.isoformat(), []).append(block)

        #generate markdown
        lines = ["# Exam Study Plan", "", f"Generated: {today.isoformat()}", ""]

        lines.append("## Course Overview")
        lines.append("")
        for course in courses:
            cc = course_configs.get(course, {})
            midterm = today + datetime.timedelta(days=cc.get("days_until_midterm", 0))
            course_blocks = [sb for sb in blocks if sb.course == course]
            total = sum(sb.duration_minutes for sb in course_blocks)
            lines.append(f"- **{course}**: {cc.get('depth', '?')} | Midterm: {midterm.isoformat()} ({cc.get('days_until_midterm', '?')} days) | {total} min total")
        lines.append("")

        if validation_gaps:
            lines.append("## Topics Requiring Manual Review")
            lines.append("")
            for gap in validation_gaps:
                lines.append(f"- **{gap['topic_name']}** ({gap['course']}): Missing: {', '.join(gap['missing_keywords'][:5])}")
            lines.append("")

        lines.append("## Day-by-Day Schedule")
        lines.append("")

        for date_str in sorted(schedule.keys()):
            date_obj = datetime.date.fromisoformat(date_str)
            day_name = date_obj.strftime("%A, %B %d")
            day_blocks = schedule[date_str]
            day_total = sum(b.duration_minutes for b in day_blocks)

            lines.append(f"### {day_name} ({day_total} min)")
            lines.append("")
            for block in day_blocks:
                lines.append(f"**{block.course} -- {block.topic}** ({block.duration_minutes} min)")
                for item in block.action_items:
                    lines.append(f"  - {item}")
                if block.source_references:
                    lines.append("  - *Source References:*")
                    for ref in block.source_references:
                        lines.append(f"    - {ref}")
                lines.append("")

        output_path = "study_plan.md"
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        ctx.session.state["output_path"] = output_path

        print(f"  Schedule written to: {output_path}")
        print(f"  Total days: {len(schedule)}")
        print(f"  Total study time: {sum(sb.duration_minutes for sb in blocks)} min")

        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part(text=f"Study plan saved to {output_path}")]
            )
        )

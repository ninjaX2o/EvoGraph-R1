"""Prompt helpers for E-VQA multimodal tool-use QA."""

from __future__ import annotations


def build_vqa_user_prompt(
    *,
    question: str,
    image_id: str = "",
    image_path: str = "",
) -> str:
    """Build the user-side VQA prompt used by multimodal records and smoke evals."""

    lines = [
        (
            "Answer the question using the internal knowledge base and the "
            "question image. You may query multiple times if needed."
        ),
        "",
        (
            "Always reason first inside <think>...</think>. If additional "
            "information is required, call a tool using "
            "<tool_call>...</tool_call> per the system-provided tool "
            "specifications and formats."
        ),
        "",
        "Retrieval and tool priority (strict):",
        (
            '- 1) Visual grounding: Start with kb_search({"query":"<img>"}) '
            "to retrieve candidate entities. Take the top-ranked visual entity "
            "as the anchor."
        ),
        (
            "- 2) Factual lookup: Form a text query using the anchoring entity "
            "name and question cues, then call kb_search. Prefer multiple rounds "
            "of refined text kb_search to gather sufficient evidence."
        ),
        (
            "- 3) Fallback: Use websearch only when refined kb_search attempts "
            "remain clearly insufficient, irrelevant, missing key knowledge, or "
            "conflicting."
        ),
        (
            "- 4) Knowledge maintenance: After websearch, resolve conflicts with "
            "the KB and integrate new information by applying graph edit operations "
            "(insert, update, delete) as needed."
        ),
        "",
        (
            "When you have the final answer, output it inside "
            "<answer>...</answer>."
        ),
        "",
    ]
    lines.append(f"Question: {question}")
    return "\n".join(lines)


def build_vqa_system_prompt(tools_json: str) -> str:
    """Build a compact system prompt for real multimodal VQA agent smoke tests."""

    return "\n".join(
        [
            "You are a careful VQA agent that answers using tool-retrieved evidence.",
            "",
            "# Tools",
            "You may call one function at a time.",
            "Function signatures are provided within <tools></tools>:",
            "<tools>",
            tools_json,
            "</tools>",
            "",
            (
                '1) Visual grounding: Start with kb_search({"query":"<img>"}) '
                "to retrieve candidate entities. Take the top-ranked visual entity "
                "as the anchor."
            ),
            (
                "2) Factual lookup: Form a text query using the anchoring entity "
                "name and question cues, then call kb_search. Prefer multiple "
                "rounds of refined text kb_search to gather sufficient evidence."
            ),
            (
                "3) Fallback: Use websearch only when refined kb_search attempts "
                "remain clearly insufficient, irrelevant, missing key knowledge, "
                "or conflicting."
            ),
            (
                "4) Knowledge maintenance: After websearch, resolve conflicts with "
                "the KB and integrate new information by applying graph edit operations "
                "(insert, update, delete) as needed."
            ),
            "Do not use tools that are not listed.",
        ]
    )

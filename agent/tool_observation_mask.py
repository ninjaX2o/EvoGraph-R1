"""Helpers for masking tool observation spans during policy training."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from typing import Any


DEFAULT_OBSERVATION_TAGS = ("knowledge", "pipeline")


def iter_tool_observation_token_spans(
    response: str,
    tokenizer: Any,
    *,
    tags: Iterable[str] = DEFAULT_OBSERVATION_TAGS,
) -> Iterator[tuple[int, int]]:
    """Yield token spans covering tool observation blocks in a decoded response."""
    for tag in tags:
        start_tag = f"<|im_start|>user\n<{tag}>"
        end_tag = f"</{tag}><|im_end|>\n<|im_start|>assistant"
        start_positions = [
            match.start() for match in re.finditer(re.escape(start_tag), response)
        ]
        end_positions = [
            match.start() + len(end_tag)
            for match in re.finditer(re.escape(end_tag), response)
        ]
        for start, end in zip(start_positions, end_positions):
            prefix_to_start = response[:start]
            state_section = response[start:end]
            start_token_pos = len(
                tokenizer.encode(prefix_to_start, add_special_tokens=False)
            )
            end_token_pos = start_token_pos + len(
                tokenizer.encode(state_section, add_special_tokens=False)
            )
            yield start_token_pos, end_token_pos

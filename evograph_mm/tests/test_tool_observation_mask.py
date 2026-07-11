from agent.tool_observation_mask import iter_tool_observation_token_spans


class _CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)


def test_tool_observation_mask_includes_knowledge_and_pipeline_blocks():
    response = (
        "before"
        "<|im_start|>user\n<knowledge>search result</knowledge><|im_end|>\n<|im_start|>assistant"
        "middle"
        "<|im_start|>user\n<pipeline>{\"success\": true}</pipeline><|im_end|>\n<|im_start|>assistant"
        "after"
    )

    spans = list(iter_tool_observation_token_spans(response, _CharTokenizer()))

    assert len(spans) == 2
    masked = ["".join(response[start:end]) for start, end in spans]
    assert "<knowledge>search result</knowledge>" in masked[0]
    assert '<pipeline>{"success": true}</pipeline>' in masked[1]

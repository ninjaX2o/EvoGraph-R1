import argparse
import json
import os
import sys
from openai import OpenAI

# Ensure we import tools from this EVO-Graph-R1 repo, not a sibling Graph-R1
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.tool.tool_env import ToolEnv, step_batch
from agent.tool.tools import _default_tools
import re
import copy

# ANSI color codes for colored output
COLORS = {
    "user": "\033[1;34m",      # Bold Blue
    "assistant": "\033[1;32m",  # Bold Green
    "tool": "\033[1;33m",       # Bold Yellow
    "tool_call": "\033[1;35m",  # Bold Purple
    "reset": "\033[0m",         # Reset to default
    "bg_user": "\033[44m",      # Blue background
    "bg_assistant": "\033[42m", # Green background
    "bg_tool": "\033[43m",      # Yellow background
    "bg_tool_call": "\033[45m", # Purple background
}

def configure_utf8_stdio():
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            continue

def parse_args():
    parser = argparse.ArgumentParser(description='Run VLLM inference with configurable parameters')
    parser.add_argument('--api-key', type=str, default="EMPTY",
                        help='OpenAI API key')
    parser.add_argument('--api-base', type=str, default="http://localhost:8002/v1",
                        help='OpenAI API base URL')
    parser.add_argument('--model', type=str, default="agent",
                        help='Model name for inference')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for sampling')
    parser.add_argument('--top-p', type=float, default=1.0,
                        help='Top-p for nucleus sampling')
    parser.add_argument('--max-tokens', type=int, default=4096,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--max-turns', type=int, default=20,
                        help='Maximum turns of search')
    parser.add_argument('--question', type=str, default="Which magazine came out first, Tit-Bits or Illustreret Nyhedsblad?",
                        help='Question to ask the model')
    parser.add_argument('--no-color', action='store_true',
                        help='Disable colored output')
    parser.add_argument('--google-results', type=int, default=3,
                        help='Number of Google search results (default: 3)')
    parser.add_argument('--search-language', type=str, default="en",
                        help='Language for Google search (default: en)')
    parser.add_argument('--tool-env', type=str, default="all",
                        choices=["all", "no_insert", "ablation", "mm_search", "mm_all"],
                        help='Tool environment: all (default), no_insert, ablation, mm_search, or mm_all')
    return parser.parse_args()


def build_system_prompt(tool_env: str, env: ToolEnv) -> str:
    mm_retrieval_priority = (
        'Retrieval priority: use kb_search first. When the image entity is uncertain '
        'or important to the question, call kb_search {"query":"<img>"} to identify '
        'likely visual entities; use a natural-language text query for factual lookup. '
        "Use websearch only if kb_search is clearly insufficient or irrelevant. "
        "Prefer multiple refined kb_search attempts before websearch."
    )
    if tool_env == "ablation":
        knowledge_maintenance_note = ""
        retrieval_priority = mm_retrieval_priority
    else:
        knowledge_maintenance_note = (
            "\nKnowledge maintenance: After using websearch, maintain knowledge state "
            "using insert/update/delete when applicable."
        )
        retrieval_priority = mm_retrieval_priority

    return (
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n"
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        + retrieval_priority + knowledge_maintenance_note + "\n\n"
        + env.tools_format_func() + "\n"
    )


def extract_answer_text(response_text: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", response_text, re.DOTALL)
    return match.group(1).strip() if match else ""


def process_tool_call(responses_str):

    eos_token = "<|im_end|>"
    tool_pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)

    processed = []
    masks = []
    for resp in responses_str:
        m = tool_pattern.search(resp)
        if not m:
            processed.append(resp + eos_token)
            masks.append(False)
            continue
        trimmed = resp[:m.end()] + eos_token
        processed.append(trimmed)
        masks.append(True)

    return processed, masks

def execute_tool_calls_batch(response_strs, env, active_masks):
    active_envs = []
    active_responses = []
    active_indices = []
    
    for i, (resp, active) in enumerate(zip(response_strs, active_masks)):
        if active:
            active_envs.append(env)
            active_responses.append(resp)
            active_indices.append(i)
    
    # Initialize result list with empty strings
    tool_responses = [""] * len(response_strs)
    
    if not active_envs:
        return tool_responses
        
    # Use the independent step_batch function for active environments
    batch_results = step_batch(active_envs, active_responses)
    
    # Map results back to original indices
    for list_pos, (idx, result) in enumerate(zip(active_indices, batch_results)):
        if result is None:
            tool_responses[idx] = ""
        else:
            tool_response = result[0]
            info = result[3] if len(result) > 3 and isinstance(result[3], dict) else {}
            # Determine which tag to use based on the tool name in the corresponding response
            # Parse tool name from the assistant response's <tool_call> JSON
            try:
                resp_with_call = active_responses[list_pos]
                m = re.search(r'<tool_call>(.*?)</tool_call>', resp_with_call, re.DOTALL)
                tool_call_json = m.group(1) if m else "{}"
                tc = json.loads(tool_call_json)
                tool_name = tc.get("tool", "") if isinstance(tc, dict) else ""
            except Exception:
                tool_name = ""

            tag = "search" if tool_name == "websearch" else "knowledge"
            if info and (
                info.get("action_is_valid") is False
                or info.get("action_is_effective") is False
            ):
                tag = "knowledge"
                tool_response = (
                    "TOOL_CALL_ERROR: "
                    f"{tool_response}\n"
                    "Retry the tool call using exactly one valid "
                    '<tool_call>{"tool": "<name>", "args": {...}}</tool_call> '
                    "block. Do not provide a final answer until the corrected "
                    "tool call succeeds."
                )
            tool_custom_response_template = f"<|im_start|>user\n<{tag}>\n{{tool_response}}\n</{tag}><|im_end|>\n<|im_start|>assistant\n<think>"
            tool_responses[idx] = tool_custom_response_template.format(tool_response=tool_response)
    return tool_responses

def colorprint(mode, r_str, t_str, use_colors):
    if not r_str.startswith("<think>\n"):
        r_str = "<think>\n" + r_str

    think_m = re.search(r'<think>(.*?)</think>', r_str, re.DOTALL)
    think = think_m.group(1) if think_m else ""
    if think and not think.endswith("\n"):
        think += "\n"

    if mode is True:
        tc_m = re.search(r'<tool_call>(.*?)</tool_call>', r_str, re.DOTALL)
        tool_call_json = tc_m.group(1) if tc_m else "{}"
        try:
            tool_call = json.loads(tool_call_json)
        except Exception:
            tool_call = {"tool": "unknown", "args": tool_call_json}

        # Parse both <knowledge> (search results) and <pipeline> (knowledge management) tags
        knowledge_texts = re.findall(r'<knowledge>(.*?)</knowledge>', t_str, re.DOTALL)
        pipeline_texts = re.findall(r'<pipeline>(.*?)</pipeline>', t_str, re.DOTALL)
        
        # Handle search results (<knowledge>)
        knowledge = knowledge_texts[0] if knowledge_texts else ""
        pretty_knowledge = knowledge
        try:
            k_obj = json.loads(knowledge)
            if isinstance(k_obj, dict) and "results" in k_obj:
                items = k_obj["results"]
                if isinstance(items, list):
                    pretty_knowledge = "\n" + "\n".join([str(x) for x in items])
                else:
                    pretty_knowledge = json.dumps(k_obj, ensure_ascii=False, indent=2)
            else:
                pretty_knowledge = json.dumps(k_obj, ensure_ascii=False, indent=2)
        except Exception:
            pass
        
        # Handle knowledge pipeline results (<pipeline>)
        pipeline = pipeline_texts[0] if pipeline_texts else ""
        pretty_pipeline = pipeline
        try:
            p_obj = json.loads(pipeline)
            if isinstance(p_obj, dict) and "results" in p_obj:
                items = p_obj["results"]
                if isinstance(items, list):
                    pretty_pipeline = "\n" + "\n".join([str(x) for x in items])
                else:
                    pretty_pipeline = json.dumps(p_obj, ensure_ascii=False, indent=2)
            else:
                pretty_pipeline = json.dumps(p_obj, ensure_ascii=False, indent=2)
        except Exception:
            pass

        search_texts = re.findall(r'<search>(.*?)</search>', t_str, re.DOTALL)
        search = search_texts[0] if search_texts else ""
        pretty_search = search
        try:
            s_obj = json.loads(search)
            if isinstance(s_obj, dict) and "results" in s_obj:
                items = s_obj["results"]
                if isinstance(items, list):
                    pretty_search = "\n" + "\n".join([str(x) for x in items])
                else:
                    pretty_search = json.dumps(s_obj, ensure_ascii=False, indent=2)
            else:
                pretty_search = json.dumps(s_obj, ensure_ascii=False, indent=2)
        except Exception:
            pass

        if use_colors:
            print(f"\n{COLORS['bg_tool_call']} Think {COLORS['reset']} {COLORS['tool_call']}{think}{COLORS['reset']}")
            print(f"{COLORS['tool_call']}Tool call:{COLORS['reset']}\n{json.dumps(tool_call, ensure_ascii=False)}{COLORS['reset']}")
            if knowledge_texts:
                print(f"\n{COLORS['bg_tool']} Knowledge {COLORS['reset']} {COLORS['tool']}{pretty_knowledge}{COLORS['reset']}")
            if pipeline_texts:
                print(f"\n{COLORS['bg_tool']} Pipeline {COLORS['reset']} {COLORS['tool']}{pretty_pipeline}{COLORS['reset']}")
            if search_texts:
                print(f"\n{COLORS['bg_tool']} Search {COLORS['reset']} {COLORS['tool']}{pretty_search}{COLORS['reset']}")
        else:
            print(f"\n[Think] {think}")
            print(f"Tool call:\n{json.dumps(tool_call, ensure_ascii=False)}")
            if knowledge_texts:
                print(f"\nKnowledge: {pretty_knowledge}")
            if pipeline_texts:
                print(f"\nPipeline: {pretty_pipeline}")
            if search_texts:
                print(f"\nSearch: {pretty_search}")
    else:
        answer = extract_answer_text(r_str)
        if use_colors:
            print(f"\n{COLORS['bg_tool_call']} Think {COLORS['reset']} {COLORS['tool_call']}{think}{COLORS['reset']}")
            print(f"{COLORS['tool_call']}Answer:{COLORS['reset']}\n{answer}{COLORS['reset']}")
        else:
            print(f"\n[Think] {think}")
            print(f"Answer:\n{answer}")

        print("\n")

def main():
    configure_utf8_stdio()
    args = parse_args()
    use_colors = not args.no_color
    OPENAI_API_KEY = args.api_key
    OPENAI_API_BASE = args.api_base
    MODEL_NAME = args.model
    TEMPERATURE = args.temperature
    TOP_P = args.top_p
    MAX_TOKENS = args.max_tokens
    MAX_TURNS = args.max_turns
    
    # Initialize OpenAI client
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
    )
    
    # Set up tools
    tools = _default_tools(args.tool_env)
    env = ToolEnv(tools=tools, max_turns=MAX_TURNS)
    
    # Propagate Google Search config to env for auto-injection
    os.environ["GOOGLE_SEARCH_NUM_RESULTS"] = str(args.google_results)
    os.environ["GOOGLE_SEARCH_LANGUAGE"] = str(args.search_language)

    # Create message with question
    question_raw = args.question
    
    system_prompt = build_system_prompt(args.tool_env, env)
    # Adjust user message based on tool environment
    if args.tool_env == "ablation":
        user_instructions = (
            '⚠️ MANDATORY: Always reason first inside `<think>`...`</think>` before any tool call.\n\n'
            'Answer the given question using the knowledge base. Query as many times as needed.\n'
            'Retrieval priority: use kb_search first. When the image entity is uncertain or important, use image kb_search with {"query":"<img>"}; use websearch only if kb_search is insufficient.\n\n'
            'REQUIRED WORKFLOW (strictly follow this order):\n'
            '1. FIRST: Think inside `<think>`...`</think>`\n'
            '2. THEN: If tools needed, call with <tool_call>...</tool_call>\n'
            '3. FINALLY: Provide answer in `<think>`...`</think>`<answer>...</answer>\n\n'
        )
    else:
        user_instructions = (
            '⚠️ MANDATORY: Always reason first inside `<think>`...`</think>` before any tool call.\n\n'
            'Answer the given question using the knowledge base. Query as many times as needed.\n'
            'Retrieval priority: use kb_search first. When the image entity is uncertain or important, use image kb_search with {"query":"<img>"}; use websearch only if kb_search is insufficient.\n'
            'Knowledge maintenance: After websearch, use insert/update/delete to maintain knowledge state.\n\n'
            'REQUIRED WORKFLOW (strictly follow this order):\n'
            '1. FIRST: Think inside `<think>`...`</think>`\n'
            '2. THEN: If tools needed, call with <tool_call>...</tool_call>\n'
            '3. FINALLY: Provide answer in `<think>`...`</think>`<answer>...</answer>\n\n'
        )
    
    messages = [{
        "role": "user", 
        "content": '<|im_start|>system\n' + system_prompt +
                   '<|im_end|>\n<|im_start|>user\n'
                   + user_instructions +
                   'Tool call format:\n'
                   '<think>...</think>\n<tool_call>{"tool": "<name>", "args": {...}}</tool_call>\n\n'
                   'Answer format:\n'
                   '<think>...</think>\n<answer>...</answer>\n'
                   'Question: ' + question_raw + '<|im_end|>\n<|im_start|>assistant\n'
    }]
    
    print(f"Running inference with model: {MODEL_NAME}")
    print(f"Tool environment: {args.tool_env}")
    print(f"Google search config: num_results={args.google_results}, language={args.search_language}")
    if use_colors:
        print(f"{COLORS['bg_user']} User {COLORS['reset']} {COLORS['user']}{question_raw}{COLORS['reset']}")
    else:
        print(f"User: {question_raw}")
        
    
    # Run inference loop
    for step in range(MAX_TURNS):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
            )
            
            # Defensive checks on response
            if not hasattr(response, "choices") or not response.choices:
                print("[Warn] Inference API returned no choices.")
                break

            # Get the response message
            response_message = response.choices[0].message
            if response_message is None or response_message.content is None:
                print("[Warn] Inference API returned empty message content.")
                break
            responses_str = [response_message.content]
            
            responses_str, active_masks = process_tool_call(responses_str)
            
            tool_responses = execute_tool_calls_batch(responses_str, env, active_masks)

            colorprint(active_masks[0], copy.deepcopy(responses_str[0]), copy.deepcopy(tool_responses[0]), use_colors)
            
            if active_masks[0] is True:
                prompt = messages[0]["content"]+responses_str[0]+tool_responses[0]
                messages = [{
                    "role": "user",
                    "content": prompt
                }] 
                # print(messages[0]["content"])
            else:
                prompt = messages[0]["content"]+responses_str[0]
                # print(prompt)
                break

        except Exception as e:
            # Always log warnings regardless of color settings
            try:
                print(f"[Warn] Inference step error: {e}")
            except Exception:
                pass
            continue
    env.close()

if __name__ == "__main__":
    main()

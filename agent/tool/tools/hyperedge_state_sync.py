import json
import os
from datetime import datetime
from typing import Dict, Iterable, List, Optional

from agent.tool.tools.hyperedge_sync import (
    format_hyperedge_content,
    normalize_hyperedge_content,
)

HYPEREDGE_LOOKUP_FILE = "hyperedge_content_lookup.json"
HYPEREDGE_RECENT_MUTATIONS_FILE = "hyperedge_recent_mutations.json"
DEFAULT_RECENT_MUTATION_LIMIT = 256


def normalize_lookup_content(raw: str) -> str:
    return normalize_hyperedge_content(raw or "")


def build_hyperedge_content_lookup(
    hyperedges_data: Dict,
    skip_deleted: bool = True,
) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for hyperedge_id, hyperedge_data in hyperedges_data.items():
        if not isinstance(hyperedge_data, dict):
            continue
        if skip_deleted and hyperedge_data.get("deleted", False):
            continue
        raw_content = hyperedge_data.get("content") or hyperedge_data.get("hyperedge_name") or ""
        normalized_content = normalize_lookup_content(raw_content)
        if normalized_content and normalized_content not in lookup:
            lookup[normalized_content] = hyperedge_id
    return lookup


def build_hyperedge_lookup_payload(hyperedges_data: Dict) -> Dict:
    searchable_lookup = build_hyperedge_content_lookup(hyperedges_data, skip_deleted=False)
    return {
        "version": 1,
        "generated_at": datetime.now().isoformat(),
        "active": build_hyperedge_content_lookup(hyperedges_data, skip_deleted=True),
        "searchable": searchable_lookup,
        "all": searchable_lookup,
    }


def write_hyperedge_lookup(working_dir: str, hyperedges_data: Dict) -> str:
    lookup_path = os.path.join(working_dir, HYPEREDGE_LOOKUP_FILE)
    tmp_lookup_path = lookup_path + ".tmp"
    payload = build_hyperedge_lookup_payload(hyperedges_data)
    with open(tmp_lookup_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_lookup_path, lookup_path)
    return lookup_path


def load_hyperedge_lookup_payload(working_dir: str) -> Dict:
    lookup_path = os.path.join(working_dir, HYPEREDGE_LOOKUP_FILE)
    if not os.path.exists(lookup_path):
        return {}
    try:
        with open(lookup_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def load_hyperedge_lookup(
    working_dir: str,
    skip_deleted: bool = True,
) -> Dict[str, str]:
    payload = load_hyperedge_lookup_payload(working_dir)
    section = "active" if skip_deleted else ("searchable" if "searchable" in payload else "all")
    lookup = payload.get(section)
    return lookup if isinstance(lookup, dict) else {}


def iter_recent_active_contents(mutations: Iterable[Dict]) -> Iterable[str]:
    seen = set()
    for mutation in mutations:
        if not isinstance(mutation, dict):
            continue
        if not mutation.get("active", False):
            continue
        raw_content = mutation.get("content") or mutation.get("plain_content") or ""
        formatted = format_hyperedge_content(raw_content)
        if formatted in seen:
            continue
        seen.add(formatted)
        yield formatted


def _recent_mutation_dedup_key(mutation: Dict) -> tuple:
    hyperedge_id = mutation.get("hyperedge_id") or ""
    if hyperedge_id:
        return ("hyperedge_id", hyperedge_id)
    return ("plain_content", mutation.get("plain_content") or "")


def load_recent_hyperedge_mutations(working_dir: str) -> List[Dict]:
    mutations_path = os.path.join(working_dir, HYPEREDGE_RECENT_MUTATIONS_FILE)
    if not os.path.exists(mutations_path):
        return []
    try:
        with open(mutations_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        mutations = payload.get("mutations") if isinstance(payload, dict) else payload
        return mutations if isinstance(mutations, list) else []
    except Exception:
        return []


def write_recent_hyperedge_mutations(
    working_dir: str,
    mutations: List[Dict],
) -> str:
    mutations_path = os.path.join(working_dir, HYPEREDGE_RECENT_MUTATIONS_FILE)
    tmp_mutations_path = mutations_path + ".tmp"
    payload = {
        "version": 1,
        "generated_at": datetime.now().isoformat(),
        "mutations": mutations,
    }
    with open(tmp_mutations_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_mutations_path, mutations_path)
    return mutations_path


def append_recent_hyperedge_mutations(
    working_dir: str,
    events: List[Dict],
    max_entries: int = DEFAULT_RECENT_MUTATION_LIMIT,
) -> str:
    mutations = load_recent_hyperedge_mutations(working_dir)
    normalized_events = []
    for event in events:
        if not isinstance(event, dict):
            continue
        plain_content = normalize_lookup_content(event.get("content") or event.get("plain_content") or "")
        if not plain_content:
            continue
        normalized_events.append(
            {
                "hyperedge_id": event.get("hyperedge_id", ""),
                "action": event.get("action", "update"),
                "content": format_hyperedge_content(plain_content),
                "plain_content": plain_content,
                "active": bool(event.get("active", False)),
                "searchable": bool(event.get("searchable", True)),
                "timestamp": event.get("timestamp") or datetime.now().isoformat(),
            }
        )

    if not normalized_events:
        return os.path.join(working_dir, HYPEREDGE_RECENT_MUTATIONS_FILE)

    combined = (normalized_events + mutations)[:max_entries]
    deduped: List[Dict] = []
    seen = set()
    for mutation in combined:
        key = _recent_mutation_dedup_key(mutation)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(mutation)
        if len(deduped) >= max_entries:
            break
    return write_recent_hyperedge_mutations(working_dir, deduped)


def ensure_hyperedge_state_sidecars(working_dir: str, hyperedges_data: Dict) -> None:
    os.makedirs(working_dir, exist_ok=True)
    write_hyperedge_lookup(working_dir, hyperedges_data)


def select_lookup_section(
    lookup_payload: Optional[Dict],
    skip_deleted: bool = True,
) -> Dict[str, str]:
    if not isinstance(lookup_payload, dict):
        return {}
    section = "active" if skip_deleted else "all"
    lookup = lookup_payload.get(section)
    return lookup if isinstance(lookup, dict) else {}

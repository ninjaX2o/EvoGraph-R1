"""
Specific tool implementations.

Imports are intentionally lazy so optional tool dependencies do not break
unrelated modules such as the BGE model manager.
"""

__all__ = [
    "KBSearchTool",
    "WebSearchTool",
    "GraphR1InsertTool",
    "HyperedgeUpdateTool",
    "HyperedgeSoftDeleteTool",
]

_LAZY_EXPORTS = {
    "KBSearchTool": ("agent.tool.tools.kb_search_tool", "KBSearchTool"),
    "WebSearchTool": ("agent.tool.tools.websearch_tool", "WebSearchTool"),
    "GraphR1InsertTool": ("agent.tool.tools.graphr1_insert_tool", "GraphR1InsertTool"),
    "HyperedgeUpdateTool": ("agent.tool.tools.hyperedge_update_tool", "HyperedgeUpdateTool"),
    "HyperedgeSoftDeleteTool": (
        "agent.tool.tools.hyperedge_soft_delete_tool",
        "HyperedgeSoftDeleteTool",
    ),
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    from importlib import import_module

    attr = getattr(import_module(module_name), attr_name)
    globals()[name] = attr
    return attr


def _default_tools(env):
    if env == "search":
        from agent.tool.tools.kb_search_tool import KBSearchTool

        return [KBSearchTool()]
    elif env == "websearch":
        from agent.tool.tools.websearch_tool import WebSearchTool

        return [WebSearchTool()]
    elif env == "mm_search":
        from agent.tool.tools.mm import MMKBSearchTool

        return [MMKBSearchTool()]
    elif env == "mm_all":
        from agent.tool.tools.websearch_tool import WebSearchTool
        from agent.tool.tools.mm import (
            MMGraphR1InsertTool,
            MMHyperedgeSoftDeleteTool,
            MMHyperedgeUpdateTool,
            MMKBSearchTool,
        )

        return [
            MMKBSearchTool(),
            WebSearchTool(),
            MMGraphR1InsertTool(),
            MMHyperedgeUpdateTool(),
            MMHyperedgeSoftDeleteTool(),
        ]
    elif env == "all":
        from agent.tool.tools.graphr1_insert_tool import GraphR1InsertTool
        from agent.tool.tools.hyperedge_soft_delete_tool import HyperedgeSoftDeleteTool
        from agent.tool.tools.hyperedge_update_tool import HyperedgeUpdateTool
        from agent.tool.tools.kb_search_tool import KBSearchTool
        from agent.tool.tools.websearch_tool import WebSearchTool

        return [
            KBSearchTool(),
            WebSearchTool(),
            GraphR1InsertTool(),
            HyperedgeUpdateTool(),
            HyperedgeSoftDeleteTool(),
        ]
    elif env == "no_insert":
        from agent.tool.tools.hyperedge_soft_delete_tool import HyperedgeSoftDeleteTool
        from agent.tool.tools.hyperedge_update_tool import HyperedgeUpdateTool
        from agent.tool.tools.kb_search_tool import KBSearchTool
        from agent.tool.tools.websearch_tool import WebSearchTool

        return [
            KBSearchTool(),
            WebSearchTool(),
            HyperedgeUpdateTool(),
            HyperedgeSoftDeleteTool(),
        ]
    elif env == "ablation":
        from agent.tool.tools.kb_search_tool import KBSearchTool
        from agent.tool.tools.websearch_tool import WebSearchTool

        return [
            KBSearchTool(),
            WebSearchTool(),
        ]
    else:
        raise NotImplementedError

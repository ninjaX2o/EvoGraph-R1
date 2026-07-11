"""Multimodal KB ownership markers."""

KB_BOUNDARY_PATH = "evograph_mm/kb"
KB_BOUNDARY_DOCUMENT = "evograph_mm/kb/README.md"

KB_OWNED_CAPABILITIES = (
    "asset audit and local manifest planning",
    "schema definition after local asset review",
    "dataset adapters",
    "KB build orchestration",
    "store layout",
    "indexing",
    "retrieval",
    "graph editing",
)

IMPLEMENTATION_RULES = (
    "Do not scatter multimodal KB implementation into root scripts.",
    "Do not modify graphr1 internals for the multimodal branch.",
    "Keep root script_mm entry points as thin wrappers.",
    "Write multimodal runtime artifacts under expr_mm/<dataset>.",
)

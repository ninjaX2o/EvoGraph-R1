# Multimodal Knowledge Base

`evograph_mm/kb/` contains the multimodal data preparation, retrieval, indexing,
and graph editing code used by EvoGraph-R1.

## Main Entry Points

- `prepare_echosight.py`: local asset planning for EchoSight-style E-VQA and
  InfoSeek resources.
- `validate_echosight.py`: read-only validation for locally provided
  multimodal assets.
- `process_mm.py`: metadata preprocessing for multimodal QA records.
- `build.py`: multimodal KB store, text graph, image index, and graph sidecar
  construction.
- `api.py`: FastAPI service for multimodal retrieval and graph-edit tools.
- `graph_edit.py`: helpers for editable per-run KB copies.

Root-level wrappers are provided for the common CLI paths:

```bash
python script_prepare_echosight_mm.py --help
python script_validate_echosight_mm.py --help
python script_process_mm.py --help
python script_build_mm.py --help
python script_api_mm.py --help
```

## Local Data Contract

The public repository does not include raw datasets, images, KB archives,
indexes, or generated graph artifacts. Keep those files outside Git under a
local root such as:

```text
datasets_mm/<dataset>/
expr_mm/<dataset>/
```

The training README in the repository root describes the expected layout and
dataset sources.

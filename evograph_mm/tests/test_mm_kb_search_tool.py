from agent.tool.tools.mm.kb_search_tool import MMKBSearchTool


def test_text_query_does_not_send_image_context():
    tool = MMKBSearchTool()

    payload = tool._build_single_payload(
        {
            "query": "What structure spirals around this stepwell?",
            "image_id": "52a183470c330a38",
            "image_path": "datasets_mm/E-VQA/subsets/x/images/52a183470c330a38.jpg",
            "context_query": "What structure spirals around this stepwell?",
        },
        "What structure spirals around this stepwell?",
    )

    assert payload == {"queries": ["What structure spirals around this stepwell?"]}


def test_image_query_sends_image_context():
    tool = MMKBSearchTool()

    payload = tool._build_single_payload(
        {
            "query": "<img>",
            "image_id": "52a183470c330a38",
            "context_query": "What structure spirals around this stepwell?",
        },
        "<img>",
    )

    assert payload["queries"] == ["<img>"]
    assert payload["image_ids"] == ["52a183470c330a38"]
    assert payload["context_queries"] == ["What structure spirals around this stepwell?"]
    assert payload["visual_entity_top_k"] == 3


def test_image_query_normalizes_windows_dataset_path():
    tool = MMKBSearchTool()

    payload = tool._build_single_payload(
        {
            "query": "<img>",
            "image_path": (
                r"C:\Users\tester\EvoGraph-R1\datasets_mm\E-VQA\subsets"
                r"\paper_vqa_evidence_aligned_5120_128_seed0\images"
                r"\8b0312472238553e.jpg"
            ),
        },
        "<img>",
    )

    assert payload["image_paths"] == [
        "datasets_mm/E-VQA/subsets/paper_vqa_evidence_aligned_5120_128_seed0/images/8b0312472238553e.jpg"
    ]


def test_batch_text_queries_do_not_send_image_context():
    tool = MMKBSearchTool()

    payload = tool._build_batch_payload(
        [
            {
                "query": "Guggenheim Museum Bilbao acquired paintings",
                "image_id": "8b0312472238553e",
                "image_path": "datasets_mm/E-VQA/subsets/x/images/8b0312472238553e.jpg",
                "context_query": "whose paintings did this museum acquire",
            },
            {
                "query": "Adalaj Stepwell spiral structure",
                "image_id": "52a183470c330a38",
                "image_path": "datasets_mm/E-VQA/subsets/x/images/52a183470c330a38.jpg",
                "context_query": "What structure spirals around this stepwell?",
            },
        ],
        [0, 1],
        {},
    )

    assert payload == {
        "queries": [
            "Guggenheim Museum Bilbao acquired paintings",
            "Adalaj Stepwell spiral structure",
        ]
    }


def test_image_response_returns_only_three_entities_with_images():
    response = {
        "results": [
            {
                "entity": f"Entity {index}",
                "image_path": f"/tmp/entity-{index}.jpg",
                "description": "must not be returned",
                "related_hyperedges": [{"text": "must not be returned"}],
                "<coherence>": 1.0,
            }
            for index in range(7)
        ]
    }

    compact = MMKBSearchTool._coerce_response_item(response, image_query=True)

    assert compact == (
        '{"results": [{"entity": "Entity 0", "image_path": "/tmp/entity-0.jpg"}, '
        '{"entity": "Entity 1", "image_path": "/tmp/entity-1.jpg"}, '
        '{"entity": "Entity 2", "image_path": "/tmp/entity-2.jpg"}]}'
    )


def test_image_response_compacts_json_string():
    response = (
        '{"results": ['
        '{"entity": "Guggenheim Museum Bilbao", "description": "drop"},'
        '{"entity": "Adalaj Stepwell", "related_hyperedges": [{"text": "drop"}]}'
        "]}"
    )

    compact = MMKBSearchTool._coerce_response_item(response, image_query=True)

    assert compact == (
        '{"results": [{"entity": "Guggenheim Museum Bilbao"}, '
        '{"entity": "Adalaj Stepwell"}]}'
    )


def test_text_response_keeps_full_payload():
    response = {
        "results": [
            {
                "<knowledge>": "The museum acquired paintings by Willem de Kooning.",
                "<coherence>": 1.0,
            }
        ]
    }

    full = MMKBSearchTool._coerce_response_item(response, image_query=False)

    assert "<knowledge>" in full
    assert "Willem de Kooning" in full

"""Section-type routing decides which node types the model is ever asked for.

Two node types have already gone missing from the graph this way — `Limitation`, which was
reachable only from `limitations`/`discussion` sections that most papers do not have, and
`Hardware`, reachable only from `appendix` (2.9% of chunks), which produced 12 nodes across
8 of 270 papers in a corpus where the large majority state a device or qubit count. Neither
raised an error anywhere: the extractor, schema and prompt were each correct in isolation,
and only the composition with the real heading distribution was wrong.

These tests pin the routing so a type cannot be silently unreachable again.
"""

from __future__ import annotations

import pytest

from rpsg.extraction.prompts import _DEFAULT_TYPES, _SECTION_TYPES, build_user_prompt
from rpsg.extraction.schema import NodeType


def _nodes_for(section_type: str) -> list[NodeType]:
    return _SECTION_TYPES.get(section_type, _DEFAULT_TYPES)[0]


def test_every_node_type_is_reachable_from_some_section() -> None:
    """The regression guard: a type no section asks for cannot enter the graph."""
    reachable = {t for nodes, _ in _SECTION_TYPES.values() for t in nodes} | set(_DEFAULT_TYPES[0])
    # Paper/Author/Venue come from the Semantic Scholar metadata path, not extraction.
    from_metadata = {NodeType.PAPER, NodeType.AUTHOR, NodeType.VENUE}
    unreachable = set(NodeType) - reachable - from_metadata
    assert not unreachable, f"no section asks for: {sorted(t.value for t in unreachable)}"


@pytest.mark.parametrize("section", ["abstract", "method", "results", "availability", "appendix"])
def test_hardware_is_askable_where_papers_state_it(section: str) -> None:
    """Experimental setup, run configuration and availability statements all name devices —
    and so does the abstract, which is where the headline device usually appears. A
    repro_gold audit scored one paper 0/3 on Google/Sycamore/23-qubits because that
    sentence lives in the abstract and Hardware was unreachable from there."""
    assert NodeType.HARDWARE in _nodes_for(section)


@pytest.mark.parametrize("section", ["abstract", "method", "results", "availability", "appendix"])
def test_the_repro_hint_follows_hardware(section: str) -> None:
    """The hint carries the `qubit_count` / vendor instruction; it is useless where
    `Hardware` cannot be extracted, and required wherever it can."""
    prompt = build_user_prompt("p1", "Setup", section, "We ran on 127 qubits.")
    assert "qubit_count" in prompt


@pytest.mark.parametrize("section", ["introduction", "related_work", "discussion"])
def test_the_repro_hint_is_absent_where_hardware_is_not_asked_for(section: str) -> None:
    prompt = build_user_prompt("p1", "Intro", section, "Some prose.")
    assert "qubit_count" not in prompt
    assert NodeType.HARDWARE not in _nodes_for(section)


def test_limitation_stays_reachable_from_the_conclusion() -> None:
    """Regression: most papers state caveats in the conclusion, not under a heading."""
    assert NodeType.LIMITATION in _nodes_for("conclusion")
    assert NodeType.LIMITATION in _DEFAULT_TYPES[0]


def test_prompt_names_the_section_and_its_allowed_types() -> None:
    prompt = build_user_prompt("paper-1", "5.2 Experimental Setup", "method", "Body text.")
    assert "paper-1" in prompt
    assert "5.2 Experimental Setup" in prompt
    assert "(type: method)" in prompt
    for t in _nodes_for("method"):
        assert t.value in prompt


def test_unknown_section_types_fall_back_to_the_defaults() -> None:
    prompt = build_user_prompt("p1", "Odd Heading", "no-such-type", "Body.")
    for t in _DEFAULT_TYPES[0]:
        assert t.value in prompt

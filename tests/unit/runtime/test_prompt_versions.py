import pytest

from agentic_rag.runtime.model_gateway import load_prompt


def test_versioned_prompts_and_path_boundary():
    assert load_prompt("router_v2").content
    assert load_prompt("router_v1").content
    for name in ["../router_v2", "router", "router_v0", "router_v2.md/../router_v1"]:
        with pytest.raises(ValueError):
            load_prompt(name)

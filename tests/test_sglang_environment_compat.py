"""SGLang environment compatibility checks migrated from the PR."""
import ast
import os
import re
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
_source = ROOT / "slime/backends/sglang_utils/sglang_engine.py"
_tree = ast.parse(_source.read_text())
_tree.body = [n for n in _tree.body if getattr(n, "name", None) == "prepare_server_environment"]
exec(compile(_tree, str(_source), "exec"))

@pytest.mark.parametrize(
    "allocator,expected",
    [
        ("expandable_segments:True", None),
        (
            "pinned_use_cuda_host_register:True,expandable_segments:False,max_split_size_mb:128",
            "pinned_use_cuda_host_register:True,max_split_size_mb:128",
        ),
        ("max_split_size_mb:128, expandable_segments:1", "max_split_size_mb:128"),
        ("pinned_use_cuda_host_register:True", "pinned_use_cuda_host_register:True"),
    ],
)
def test_server_environment_preserves_supported_allocator_options(allocator, expected):
    environ = {"PYTORCH_ALLOC_CONF": allocator, "PYTORCH_CUDA_ALLOC_CONF": allocator}
    prepare_server_environment(environ)
    assert environ.get("PYTORCH_ALLOC_CONF") == expected
    assert environ.get("PYTORCH_CUDA_ALLOC_CONF") == expected
    assert environ["MC_MIN_PRC_PORT"] == "27000"
    assert environ["MC_MAX_PRC_PORT"] == "28000"


@pytest.mark.parametrize("name", ["MC_MIN_RPC_PORT", "MC_MAX_RPC_PORT", "MC_MIN_PRC_PORT", "MC_MAX_PRC_PORT"])
def test_server_environment_preserves_explicit_mooncake_range(name):
    environ = {name: "31000"}
    prepare_server_environment(environ)
    assert environ == {name: "31000"}

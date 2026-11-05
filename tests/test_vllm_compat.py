"""Tests for the vLLM version-compat layer.

No vLLM, no torch, no GPU: every test here builds a fake package tree on disk
or a fake module in ``sys.modules``. That is the point of the layer -- it is
the half of the plugin that can be checked without the stack.

The negative controls matter more than the positive ones. A compat checker
that cannot be made to fail is indistinguishable from one that always passes,
and the latter is worse than nothing because it is believed.
"""

import importlib.util
import os
import sys
import textwrap

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.path.join(HERE, "..", "atom", "plugin", "vllm")


def _load(name, filename):
    """Load a plugin module straight off disk.

    ``atom.plugin.vllm.__init__`` imports ``register``, which imports torch, so
    importing these by package path would drag the whole stack in. Both modules
    under test are written to avoid that; loading them this way is what proves
    it stays true.
    """
    spec = importlib.util.spec_from_file_location(name, os.path.join(PLUGIN, filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


compat = _load("_atom_compat_under_test", "_compat.py")
checker = _load("_atom_compat_check_under_test", "compat_check.py")


# ---------------------------------------------------------------------------
# Version parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0.33.0", (0, 33, 0)),
        ("0.33.0+rocm10.1.0", (0, 33, 0)),
        ("0.34.0.dev123+g1234abc", (0, 34, 0)),
        ("0.33.0rc1", (0, 33, 0)),
        ("dev", None),
        ("", None),
    ],
)
def test_version_parsing(raw, expected):
    assert compat._parse_version(raw) == expected


def test_tested_max_is_not_below_supported_min():
    assert compat.SUPPORTED_MIN <= compat.TESTED_MAX


# ---------------------------------------------------------------------------
# Relocation resolution
# ---------------------------------------------------------------------------


def _fake_module(name, **attrs):
    module = type(sys)(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


@pytest.fixture
def fake_vllm(monkeypatch):
    created = []

    def make(name, **attrs):
        created.append(name)
        return _fake_module(name, **attrs)

    yield make
    for name in created:
        sys.modules.pop(name, None)


def test_import_first_takes_the_first_that_resolves(fake_vllm):
    fake_vllm("fakepkg.new", wanted="new-one")
    fake_vllm("fakepkg.old", wanted="old-one")
    got = compat.import_first(
        compat.Candidate("fakepkg.new", "wanted"),
        compat.Candidate("fakepkg.old", "wanted"),
    )
    assert got == "new-one"


def test_import_first_falls_back(fake_vllm):
    fake_vllm("fakepkg.old", wanted="old-one")
    got = compat.import_first(
        compat.Candidate("fakepkg.nope", "wanted"),
        compat.Candidate("fakepkg.old", "wanted"),
    )
    assert got == "old-one"


def test_import_first_error_names_every_candidate_and_the_version():
    with pytest.raises(compat.VllmSymbolMissing) as excinfo:
        compat.import_first(
            compat.Candidate("fakepkg.gone", "wanted"),
            compat.Candidate("fakepkg.alsogone", "wanted"),
            what="the thing",
        )
    message = str(excinfo.value)
    assert "fakepkg.gone:wanted" in message
    assert "fakepkg.alsogone:wanted" in message
    # The running vLLM version is the fact a bare ImportError never carries,
    # and the first thing anyone debugging this needs.
    assert "vLLM" in message
    assert "the thing" in message


def test_import_first_optional_returns_none():
    assert (
        compat.import_first(compat.Candidate("fakepkg.gone", "wanted"), required=False)
        is None
    )


# ---------------------------------------------------------------------------
# The arity adapter -- the bug a path-only fix would not have caught
# ---------------------------------------------------------------------------


def test_pool_bytes_per_block_handles_both_signatures(monkeypatch):
    calls = {}

    def new_style(vllm_config, kv_cache_groups):
        calls["style"] = "new"
        return 111

    def old_style(kv_cache_groups):
        calls["style"] = "old"
        return 222

    monkeypatch.setattr(compat, "resolve", lambda key: new_style)
    assert compat.pool_bytes_per_block("CONFIG", ["g"]) == 111
    assert calls["style"] == "new"

    monkeypatch.setattr(compat, "resolve", lambda key: old_style)
    assert compat.pool_bytes_per_block("CONFIG", ["g"]) == 222
    assert calls["style"] == "old"


def test_kv_cache_planning_module_picks_the_one_with_the_symbol(fake_vllm):
    # The 0.33 shape: the old module still imports, and has nothing on it.
    fake_vllm("fakeplanning.old")
    fake_vllm("fakeplanning.new", get_kv_cache_configs=lambda: None)
    compat.KV_CACHE_PLANNING_MODULES = ("fakeplanning.new", "fakeplanning.old")
    try:
        module = compat.kv_cache_planning_module("get_kv_cache_configs")
        assert module.__name__ == "fakeplanning.new"
    finally:
        compat.KV_CACHE_PLANNING_MODULES = (
            "vllm.v1.core.kv_cache_planning",
            "vllm.v1.core.kv_cache_utils",
        )


# ---------------------------------------------------------------------------
# Patch registry
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_registry():
    compat.reset()
    yield
    compat.reset()


def test_patch_records_an_install():
    @compat.patch("demo.installs")
    def installs():
        return None

    installs()
    (record,) = compat.report()
    assert record.installed and record.status == "installed"


def test_patch_records_a_stand_down_with_its_reason():
    @compat.patch("demo.stands_down")
    def stands_down():
        compat.stood_down("vllm.whatever.Thing is gone")
        return False

    stands_down()
    (record,) = compat.report()
    assert not record.installed
    assert record.reasons == ["vllm.whatever.Thing is gone"]


def test_a_bare_false_is_a_stand_down_with_no_reason():
    """The state ``doctor`` exits non-zero on: did not install, did not say why."""

    @compat.patch("demo.silent")
    def silent():
        return False

    silent()
    (record,) = compat.report()
    assert not record.installed
    assert record.reasons == ["returned False and gave no reason"]


def test_idempotent_reentry_stays_installed():
    """Both registration hooks install the shared patches; the second call
    finding its own marker and returning False must not undo the first."""
    state = {"installed": False}

    @compat.patch("demo.idempotent")
    def idempotent():
        if state["installed"]:
            return False
        state["installed"] = True
        return True

    idempotent()
    idempotent()
    (record,) = compat.report()
    assert record.installed
    assert record.calls == 2


def test_patch_records_then_reraises():
    @compat.patch("demo.raises")
    def raises():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        raises()
    (record,) = compat.report()
    assert record.status == "ERROR"
    assert "boom" in record.error


# ---------------------------------------------------------------------------
# The static checker, against synthetic package trees
# ---------------------------------------------------------------------------


def _write(root, relpath, body):
    path = os.path.join(root, relpath)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write(textwrap.dedent(body))
    return path


@pytest.fixture
def trees(tmp_path, monkeypatch):
    """A fake `vllm` package and a fake consumer of it.

    The real relocation table is emptied for these: it names real vLLM modules,
    none of which exist in a synthetic tree, so leaving it in place would add
    four 'broken relocation' rows to every assertion below. The two tests that
    exercise the table put their own entries back.
    """
    monkeypatch.setattr(checker, "RELOCATIONS", {})
    target = tmp_path / "target"
    _write(str(target), "vllm/__init__.py", "__version__ = '9.9.9'\n")
    _write(str(target), "vllm/here.py", "def present():\n    pass\n")
    consumer = tmp_path / "consumer" / "atom"
    os.makedirs(str(consumer), exist_ok=True)
    return str(target), str(consumer)


def test_clean_tree_passes(trees):
    target, consumer = trees
    _write(consumer, "ok.py", "from vllm.here import present\n")
    tree = checker.SourceTree.locate("vllm", target)
    result = checker.analyse(tree, consumer)
    assert result["blocking"] == []


def test_a_missing_symbol_is_reported(trees):
    target, consumer = trees
    _write(consumer, "bad.py", "from vllm.here import absent\n")
    tree = checker.SourceTree.locate("vllm", target)
    result = checker.analyse(tree, consumer)
    assert len(result["blocking"]) == 1
    assert result["blocking"][0]["verdict"] == "missing-symbol"


def test_type_checking_imports_are_not_reported(trees):
    """They never execute, so they cannot fail. Three of LMCache's five
    'breaks' against vLLM 0.33 were exactly this."""
    target, consumer = trees
    _write(
        consumer,
        "typing_only.py",
        """
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            from vllm.here import absent
        """,
    )
    tree = checker.SourceTree.locate("vllm", target)
    assert checker.analyse(tree, consumer)["blocking"] == []


def test_a_fallback_ladder_is_one_group(trees):
    """The new location in the try, the old one in the handler. The handler's
    import is MEANT to fail on a current vLLM."""
    target, consumer = trees
    _write(
        consumer,
        "ladder.py",
        """
        try:
            from vllm.here import present
        except ImportError:
            from vllm.gone import present
        """,
    )
    tree = checker.SourceTree.locate("vllm", target)
    assert checker.analyse(tree, consumer)["blocking"] == []


def test_a_ladder_where_every_arm_fails_is_reported(trees):
    target, consumer = trees
    _write(
        consumer,
        "allgone.py",
        """
        try:
            from vllm.gone import thing
        except ImportError:
            from vllm.alsogone import thing
        """,
    )
    tree = checker.SourceTree.locate("vllm", target)
    assert len(checker.analyse(tree, consumer)["blocking"]) == 1


def test_a_handler_that_renames_still_counts_as_the_same_capability(trees):
    """`from x import old_name as wanted` supplies `wanted`, so grouping must
    key on the bound name, not the imported one."""
    target, consumer = trees
    _write(
        consumer,
        "renamed.py",
        """
        try:
            from vllm.here import wanted
        except ImportError:
            from vllm.here import present as wanted
        """,
    )
    tree = checker.SourceTree.locate("vllm", target)
    assert checker.analyse(tree, consumer)["blocking"] == []


def test_a_handler_that_copes_satisfies_the_group(trees):
    target, consumer = trees
    _write(
        consumer,
        "coping.py",
        """
        def layout():
            try:
                from vllm.gone import get_layout
            except Exception:
                return None
            return get_layout()
        """,
    )
    tree = checker.SourceTree.locate("vllm", target)
    assert checker.analyse(tree, consumer)["blocking"] == []


def test_a_handler_that_only_reraises_does_not(trees):
    """Improving the error message is not coping with the error."""
    target, consumer = trees
    _write(
        consumer,
        "reraises.py",
        """
        try:
            from vllm.gone import thing
        except ImportError as e:
            raise ImportError("install vllm") from e
        """,
    )
    tree = checker.SourceTree.locate("vllm", target)
    assert len(checker.analyse(tree, consumer)["blocking"]) == 1


def test_a_monkeypatch_target_that_moved_is_reported(trees):
    """The failure no import check sees: the module still imports, and the
    attribute being patched onto it is gone."""
    target, consumer = trees
    _write(
        consumer,
        "patcher.py",
        """
        import vllm.here as mod

        def apply():
            original = mod.departed
            mod.departed = original
        """,
    )
    tree = checker.SourceTree.locate("vllm", target)
    result = checker.analyse(tree, consumer)
    assert [t["attr"] for t in result["patch_targets"]] == ["departed"]
    assert result["blocking"]


def test_atom_markers_are_not_treated_as_vllm_attributes(trees):
    target, consumer = trees
    _write(
        consumer,
        "marker.py",
        """
        import vllm.here as mod

        def apply():
            if getattr(mod, "_atom_already", False):
                return
            mod._atom_already = True
        """,
    )
    tree = checker.SourceTree.locate("vllm", target)
    assert checker.analyse(tree, consumer)["patch_targets"] == []


def test_a_stale_relocation_entry_is_reported(trees, monkeypatch):
    """The forward-looking half: once a call site goes through the shim there
    is no import left to scan, so the table itself has to be resolved."""
    target, consumer = trees
    monkeypatch.setattr(
        checker,
        "RELOCATIONS",
        {
            "vllm.here:present": compat.Relocation(
                candidates=(compat.Candidate("vllm.vanished", "present"),)
            )
        },
    )
    tree = checker.SourceTree.locate("vllm", target)
    result = checker.analyse(tree, consumer)
    assert [r["status"] for r in result["relocations"]] == ["broken"]
    assert result["blocking"]


def test_arity_drift_is_reported(trees, monkeypatch):
    """The symbol resolves, so nothing raises -- and the call is wrong."""
    target, consumer = trees
    _write(
        target,
        "vllm/sized.py",
        "def bytes_per_block(vllm_config, groups):\n    return 0\n",
    )
    monkeypatch.setattr(
        checker,
        "RELOCATIONS",
        {
            "vllm.sized:bytes_per_block": compat.Relocation(
                candidates=(
                    compat.Candidate("vllm.sized", "bytes_per_block", ("groups",)),
                )
            )
        },
    )
    tree = checker.SourceTree.locate("vllm", target)
    result = checker.analyse(tree, consumer)
    assert len(result["arity_drift"]) == 1
    assert result["arity_drift"][0]["actual"] == ["vllm_config", "groups"]

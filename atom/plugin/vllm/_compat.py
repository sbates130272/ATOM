"""Everything in the vLLM plugin that depends on *which* vLLM is installed.

ATOM's vLLM plugin reaches into vLLM's private internals: 324 distinct imports,
many into ``vllm.v1.*``, plus ~36 monkeypatches of module-level functions and
methods. None of that is covered by a public API contract, so every vLLM minor
release can move a symbol or change an arity underneath it.

Before this module there was no version-dependent code at all -- not one
reference to ``vllm.__version__`` anywhere under ``atom/`` -- and no ``vllm``
dependency in ``pyproject.toml``, so pip installs any version happily. Skew
surfaced either as an ``ImportError`` deep inside a serving run, or, far worse,
as a monkeypatch whose target had moved: ``getattr(mod, name, None)`` returned
``None``, the patch stood down, and the behaviour it existed to supply was
silently gone.

This module is the one place that knows about those differences:

* :data:`RELOCATIONS` -- the table of symbols that moved, and where to.
  **A new vLLM release is accommodated by editing this table and nothing else.**
* :func:`import_first` -- resolve a symbol across candidate locations, and fail
  with one error that names every location tried *and the running vLLM version*.
* :func:`pool_bytes_per_block` -- the one relocated symbol that also changed
  arity, normalized.
* The patch registry (:func:`patch`, :func:`stood_down`, :func:`report`) --
  turns a silent stand-down into a reported one.

It must stay importable without vLLM installed: ``compat_check`` reads
:data:`RELOCATIONS` out of it on a machine with no GPU and no vLLM. Every
``import vllm`` below is therefore function-local.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import logging
import threading
from dataclasses import dataclass, field

logger = logging.getLogger("atom")

# ---------------------------------------------------------------------------
# Supported range
# ---------------------------------------------------------------------------

# The oldest vLLM this plugin is known to work against: the commit ATOM v0.1.9
# itself targeted (db9527a4, 0.31.0).
SUPPORTED_MIN = (0, 31, 0)

# The newest vLLM this plugin has been exercised against. Not a ceiling -- a
# higher version logs once and runs. Raise it only after `compat_check` is clean
# and `doctor` reports no unexplained stand-down on that version.
TESTED_MAX = (0, 33, 0)

_version_warned = False


def vllm_version() -> tuple[int, ...] | None:
    """The running vLLM's release tuple, or ``None`` if it cannot be read.

    Tolerant of local versions and dev suffixes: ``0.33.0+rocm10.1.0``,
    ``0.34.0.dev123+g1234abc`` and ``0.33.0rc1`` all reduce to their release
    part. ``None`` rather than an exception, because every caller here treats an
    unknown version as "proceed and hope", which is the behaviour that existed
    before this module.
    """
    try:
        import vllm

        raw = vllm.__version__
    except Exception:  # noqa: BLE001 - any failure means "unknown"
        return None
    return _parse_version(raw)


def _parse_version(raw: str) -> tuple[int, ...] | None:
    """Release-tuple of a PEP 440-ish version string. Pure, for testability."""
    if not raw:
        return None
    # Drop the local version (`+rocm...`) and anything from the first
    # non-release segment onwards (`rc1`, `.dev0`, `a1`, `-foo`).
    head = raw.split("+", 1)[0]
    parts: list[int] = []
    for chunk in head.split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
        if digits != chunk:
            # Trailing non-digits (`0rc1`) end the release segment.
            break
    return tuple(parts) if parts else None


def version_str() -> str:
    try:
        import vllm

        return str(vllm.__version__)
    except Exception:  # noqa: BLE001
        return "<vllm not importable>"


def check_version() -> None:
    """Log once if the running vLLM is outside the supported range.

    Deliberately not fatal. A version below :data:`SUPPORTED_MIN` or above
    :data:`TESTED_MAX` may well work -- the point is that when it does not, the
    log says so at startup rather than leaving the user to infer it from a
    stack trace a thousand lines into a serving run.
    """
    global _version_warned
    if _version_warned:
        return
    _version_warned = True
    found = vllm_version()
    if found is None:
        logger.warning(
            "ATOM plugin: cannot read vllm.__version__; the plugin's "
            "version-dependent shims cannot tell whether they apply"
        )
        return
    if found < SUPPORTED_MIN:
        logger.warning(
            "ATOM plugin: vLLM %s is older than the oldest supported %s. "
            "Run `python -m atom.plugin.vllm.compat_check` to see what does "
            "not resolve.",
            version_str(),
            ".".join(map(str, SUPPORTED_MIN)),
        )
    elif found > TESTED_MAX:
        logger.warning(
            "ATOM plugin: vLLM %s is newer than the newest tested %s. The "
            "plugin may still work; run `python -m atom.plugin.vllm.doctor` to "
            "see which patches installed and which stood down.",
            version_str(),
            ".".join(map(str, TESTED_MAX)),
        )


# ---------------------------------------------------------------------------
# Relocation table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One place a symbol may live, and the parameters it takes there.

    ``params`` is the positional parameter list as declared at that location.
    It is not used at import time -- :func:`import_first` does not check it --
    but ``compat_check`` compares it against the real signature in a vLLM
    checkout, which is what catches a symbol that moved *and* changed arity.
    ``None`` means "not a callable, or arity not tracked".
    """

    module: str
    name: str
    params: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Relocation:
    candidates: tuple[Candidate, ...]
    note: str = ""
    # True when resolving nothing is an acceptable outcome: the call site asks
    # with required=False and does without. Keeps the checker from reporting a
    # capability vLLM simply does not have as a break.
    optional: bool = False


def _rel(
    key: str, *candidates: Candidate, note: str = "", optional: bool = False
) -> tuple[str, Relocation]:
    return key, Relocation(candidates=candidates, note=note, optional=optional)


# Keyed by the location ATOM v0.1.9 originally wrote, as "module:name".
# Candidates are tried in order, newest vLLM first: a fresh install should
# resolve on the first try, and the fallbacks keep older vLLM working.
#
# ADDING A VLLM RELEASE: run `python -m atom.plugin.vllm.compat_check
# --vllm /path/to/vllm`, and for each unresolved import add or extend an entry
# here. That is the whole procedure; see UPGRADING-VLLM.md.
RELOCATIONS: dict[str, Relocation] = dict(
    [
        _rel(
            "vllm.v1.core.kv_cache_utils:may_override_num_blocks",
            Candidate(
                "vllm.v1.core.kv_cache_planning",
                "may_override_num_blocks",
                ("vllm_config", "num_blocks"),
            ),
            Candidate(
                "vllm.v1.core.kv_cache_utils",
                "may_override_num_blocks",
                ("vllm_config", "num_blocks"),
            ),
            note="vLLM 0.33 split kv_cache_utils into kv_cache_planning and "
            "kv_cache_config_builder. Moved only; arity unchanged.",
        ),
        _rel(
            "vllm.v1.core.kv_cache_utils:_auto_fit_max_model_len",
            Candidate("vllm.v1.core.kv_cache_planning", "_auto_fit_max_model_len"),
            Candidate("vllm.v1.core.kv_cache_utils", "_auto_fit_max_model_len"),
            note="Same 0.33 split. Private, so there is no deprecation path "
            "upstream and no warning when it moves again.",
        ),
        _rel(
            "vllm.v1.core.kv_cache_utils:_pool_bytes_per_block",
            Candidate(
                "vllm.v1.core.kv_cache_planning",
                "_pool_bytes_per_block",
                ("vllm_config", "kv_cache_groups"),
            ),
            Candidate(
                "vllm.v1.core.kv_cache_utils",
                "_pool_bytes_per_block",
                ("kv_cache_groups",),
            ),
            note="The dangerous one: 0.33 both MOVED it and gave it a "
            "vllm_config first parameter. A path-only fix calls it with the "
            "groups list as vllm_config and gets a wrong number, not an "
            "exception. Always reach it through pool_bytes_per_block().",
        ),
        _rel(
            "vllm.transformers_utils.tokenizer:MistralTokenizer",
            Candidate("vllm.tokenizers.mistral", "MistralTokenizer"),
            Candidate("vllm.transformers_utils.tokenizer", "MistralTokenizer"),
            note="Broken against 0.31 AND 0.33: vLLM moved the Mistral "
            "tokenizer to vllm.tokenizers.mistral before either. ATOM's "
            "benchmark harness catches the ImportError and re-raises it as "
            "'MistralTokenizer requires vllm package', which is the wrong "
            "diagnosis -- vllm IS installed, the class is simply elsewhere.",
        ),
        _rel(
            "vllm.transformers_utils.gguf_utils:maybe_patch_hf_config_from_gguf",
            Candidate(
                "vllm.transformers_utils.gguf_utils",
                "maybe_patch_hf_config_from_gguf",
            ),
            note="Broken against 0.31 and 0.33: there is no gguf_utils module "
            "at either, and no function of this name anywhere in vLLM. ATOM "
            "v0.1.9 imports it unguarded, so a GGUF checkpoint that falls into "
            "this branch raises ImportError instead of the ValueError the "
            "branch was written to handle. Optional: without it, the config "
            "simply is not GGUF-patched, which is what every non-GGUF "
            "checkpoint already gets.",
            optional=True,
        ),
    ]
)


class VllmSymbolMissing(ImportError):
    """Raised when no candidate location for a symbol resolves."""


def import_first(*candidates: Candidate | tuple, what: str = "", required: bool = True):
    """Return the first candidate symbol that resolves.

    Replaces the ``try: from a import x / except ImportError: from b import x``
    ladders scattered through the plugin, which said nothing about *which* vLLM
    was running when both arms failed.

    With ``required=False`` an unresolvable symbol returns ``None`` instead of
    raising -- for optional capabilities, where the caller stands down and says
    so via :func:`stood_down`.
    """
    norm = [c if isinstance(c, Candidate) else Candidate(*c) for c in candidates]
    tried = []
    for cand in norm:
        try:
            module = importlib.import_module(cand.module)
        except ImportError:
            tried.append(f"{cand.module}:{cand.name} (no such module)")
            continue
        obj = getattr(module, cand.name, None)
        if obj is None:
            tried.append(f"{cand.module}:{cand.name} (module has no such name)")
            continue
        return obj
    if not required:
        return None
    label = what or (norm[0].name if norm else "<nothing>")
    raise VllmSymbolMissing(
        f"ATOM vLLM plugin: cannot resolve {label} against vLLM "
        f"{version_str()}. Tried, in order:\n  "
        + "\n  ".join(tried)
        + "\nIf vLLM moved it again, add the new location to RELOCATIONS in "
        "atom/plugin/vllm/_compat.py; `python -m atom.plugin.vllm.compat_check` "
        "lists every such break at once."
    )


def resolve(key: str, required: bool = True):
    """Resolve a relocated symbol by its ``"module:name"`` v0.1.9 key."""
    rel = RELOCATIONS.get(key)
    if rel is None:
        module, _, name = key.partition(":")
        return import_first(Candidate(module, name), what=key, required=required)
    return import_first(*rel.candidates, what=key, required=required)


# ---------------------------------------------------------------------------
# Modules that are monkeypatched, rather than imported from
# ---------------------------------------------------------------------------

# vLLM 0.33 moved its whole KV-cache planning surface out of
# ``vllm.v1.core.kv_cache_utils`` and into ``vllm.v1.core.kv_cache_planning``:
# get_kv_cache_groups, get_kv_cache_config_from_groups,
# _max_memory_usage_bytes_from_groups, _pool_bytes_per_block,
# get_kv_cache_configs, _auto_fit_max_model_len and may_override_num_blocks --
# every one of the seven ATOM reaches for.
#
# ``kv_cache_utils`` still EXISTS at 0.33, which is what makes this dangerous:
# ``import vllm.v1.core.kv_cache_utils`` succeeds, so an installer guarded by
# try/except ImportError sails straight through and then finds nothing to
# patch. ATOM v0.1.9 has two such installers. One raises AttributeError at
# startup; the other -- `_patch_auto_fit` -- uses ``getattr(module, name,
# None)`` and simply does not install, leaving DeepSeek-V4.1's
# ``--max-model-len -1`` to fit against a pool whose STATE tail has not been
# taken out yet. No error, wrong number.
KV_CACHE_PLANNING_MODULES = (
    "vllm.v1.core.kv_cache_planning",
    "vllm.v1.core.kv_cache_utils",
)


def kv_cache_planning_module(attr: str = "get_kv_cache_configs"):
    """The module that *defines* ``attr``, for installers that patch by module.

    Selected by where the symbol actually is, not by version number, because
    the thing being patched is a module attribute and the only question that
    matters is which module currently owns it.
    """
    for name in KV_CACHE_PLANNING_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(module, attr):
            return module
    raise VllmSymbolMissing(
        f"ATOM vLLM plugin: no module defines {attr!r} against vLLM "
        f"{version_str()}. Tried: {', '.join(KV_CACHE_PLANNING_MODULES)}. Add "
        "the new module to KV_CACHE_PLANNING_MODULES in "
        "atom/plugin/vllm/_compat.py."
    )


# ---------------------------------------------------------------------------
# Arity-normalizing wrappers
# ---------------------------------------------------------------------------


def pool_bytes_per_block(vllm_config, kv_cache_groups) -> int:
    """``_pool_bytes_per_block``, called correctly on either vLLM.

    0.31 takes ``(kv_cache_groups)``; 0.33 takes ``(vllm_config,
    kv_cache_groups)``. Dispatch on the resolved function's real signature
    rather than on the version number, so a backport or a fork that differs
    from its version string still works.
    """
    fn = resolve("vllm.v1.core.kv_cache_utils:_pool_bytes_per_block")
    try:
        takes_config = "vllm_config" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        # Un-introspectable (C function, odd decorator): fall back to the
        # modern signature, since that is what a current vLLM has.
        takes_config = True
    return fn(vllm_config, kv_cache_groups) if takes_config else fn(kv_cache_groups)


# ---------------------------------------------------------------------------
# Patch registry
# ---------------------------------------------------------------------------


@dataclass
class PatchRecord:
    name: str
    installed: bool = False
    reasons: list[str] = field(default_factory=list)
    error: str = ""
    calls: int = 0

    @property
    def status(self) -> str:
        if self.error:
            return "ERROR"
        if self.installed:
            return "installed"
        return "stood down"


_registry: dict[str, PatchRecord] = {}
_lock = threading.Lock()
_current = threading.local()


def _record(name: str) -> PatchRecord:
    with _lock:
        return _registry.setdefault(name, PatchRecord(name=name))


def stood_down(reason: str) -> None:
    """Called from inside a patch that is declining to install.

    Every early ``return`` in an ``apply_*`` function that happens because a
    vLLM target is absent should call this first. The reason is what turns
    "nothing happened" into a diagnosis.
    """
    name = getattr(_current, "name", None)
    if name is None:
        logger.debug("ATOM plugin: stood_down(%r) outside a patch", reason)
        return
    rec = _record(name)
    rec.installed = False
    if reason not in rec.reasons:
        rec.reasons.append(reason)


def already_installed(reason: str = "idempotent re-entry") -> None:
    """Called from a patch's 'I already ran' guard.

    Distinct from :func:`stood_down`: a second call that finds its own marker
    is a success, not a failure, and must not clear the installed flag. Both
    hooks (``register_platform`` and ``register_model``) install the shared
    patches, so this is the common case, not an edge one.
    """
    name = getattr(_current, "name", None)
    if name is None:
        return
    rec = _record(name)
    rec.installed = True
    if reason not in rec.reasons:
        rec.reasons.append(reason)


def patch(name: str):
    """Mark a function as a vLLM patch installer, and account for its outcome.

    A patch counts as installed if it ran to completion without calling
    :func:`stood_down`, without returning ``False``, and without raising.

    ``False`` is read as "did not install on this call" because that is the
    convention ATOM's installers already follow -- ``apply_vllm_v4_prefill_
    not_uniform_patch`` and friends return ``False`` both when their vLLM
    target is missing and when they find their own marker already in place.
    Those two cases are not the same thing, which is why the handlers that
    mean the first one also call :func:`stood_down` with a reason; a bare
    ``False`` after an earlier successful call stays "installed", since that
    is the idempotent re-entry both registration hooks produce by design.

    An exception is recorded and re-raised -- this decorator reports, it does
    not swallow.
    """

    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            rec = _record(name)
            rec.calls += 1
            previous = getattr(_current, "name", None)
            _current.name = name
            was_installed = rec.installed
            rec.installed = True
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                rec.installed = was_installed
                rec.error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                _current.name = previous
            if result is False and not was_installed:
                rec.installed = False
                if not rec.reasons:
                    rec.reasons.append("returned False and gave no reason")
            elif result is False:
                rec.installed = True
            return result

        wrapper._atom_patch_name = name
        return wrapper

    return decorate


def report() -> list[PatchRecord]:
    """Every patch that has been invoked, worst news first."""
    with _lock:
        records = list(_registry.values())
    return sorted(records, key=lambda r: (not r.error, r.installed, r.name))


def registered_names() -> list[str]:
    with _lock:
        return sorted(_registry)


def reset() -> None:
    """Drop all accounting. For tests only."""
    with _lock:
        _registry.clear()

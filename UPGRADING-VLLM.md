# Moving the vLLM plugin to a new vLLM

ATOM's V1 plugin reaches deep into vLLM's private internals: ~320 import sites
across 150 distinct symbols, and 48 monkeypatches of module-level functions and
methods. None of that is covered by a stability contract, so every vLLM minor
release can move something. Before this branch, nothing in the tree checked —
there was no reference to `vllm.__version__` anywhere under `atom/`, and
`pyproject.toml` declared no `vllm` dependency, so pip installed any version
happily.

This is the procedure that replaces reading 29k lines of plugin by hand.

## The two failure modes

**Loud.** A symbol moved and the import raises. Annoying, but it tells you.

**Silent, and the reason this document exists.** The *module* still imports and
the symbol inside it moved. `import vllm.v1.core.kv_cache_utils` succeeds at
0.33 — the module is still there — but all seven functions ATOM patches onto it
left for `kv_cache_planning`. Installers that read their target with
`getattr(module, name, None)` then install nothing, return, and report nothing.
`register.py` records one occasion where exactly this happened: "the dummy-run
marking once covered nothing while the synthetic batches went on taking real
STATE slots."

A third, nastier variant: the symbol moved **and changed arity**.
`_pool_bytes_per_block` gained a `vllm_config` first parameter in 0.33. Fix only
the path and the call passes the groups list where the config belongs, raises
nothing, and returns a wrong number of bytes per block.

## Procedure

### 1. Run the static check

Needs no GPU, no ROCm, no torch, and not even an installed vLLM — it reads both
sides as ASTs and takes under a second.

```
python -m atom.plugin.vllm.compat_check --target /path/to/vllm-checkout
python atom/plugin/vllm/compat_check.py --target /path/to/vllm   # no torch
```

It reports four things:

| section | what it means |
| --- | --- |
| Unresolved imports | an import that resolves nowhere, counting `try/except ImportError` ladders as one group |
| Monkeypatch targets that no longer exist | `import vllm.x as m` succeeds, `m.attr` is gone — the silent one |
| Relocation table | every entry in `_compat.RELOCATIONS`, resolved against this vLLM |
| Arity drift | a relocated symbol whose parameter list no longer matches what `_compat` declares |

Non-zero exit if any of them has a problem.

### 2. Add what it found to the relocation table

One table, in [`atom/plugin/vllm/_compat.py`](atom/plugin/vllm/_compat.py):

```python
_rel(
    "vllm.v1.core.kv_cache_utils:may_override_num_blocks",
    Candidate("vllm.v1.core.kv_cache_planning", "may_override_num_blocks",
              ("vllm_config", "num_blocks")),
    Candidate("vllm.v1.core.kv_cache_utils", "may_override_num_blocks",
              ("vllm_config", "num_blocks")),
    note="why it moved, and anything that bit you",
),
```

The key is where ATOM v0.1.9 originally wrote it; candidates are tried newest
first. Declare the parameter names — that is what makes the arity check work.

For a module-level monkeypatch, extend `KV_CACHE_PLANNING_MODULES` (or add a
similar tuple) rather than hardcoding a module name in the installer.

### 3. Route the call site through the shim

The checker treats a surviving raw import of a relocated symbol as a failure,
deliberately: the table knowing where a symbol went does nothing for code that
still writes the old import literally.

```python
from atom.plugin.vllm._compat import resolve
fn = resolve("vllm.v1.core.kv_cache_utils:may_override_num_blocks")
```

If the arity changed, add a normalizing wrapper beside
`_compat.pool_bytes_per_block` and call that instead. Dispatch on the resolved
function's real signature, not on the version number — a backport or a fork
that differs from its version string still has to work.

Re-run step 1 until it exits zero.

### 4. Run the doctor, on hardware

The static check cannot see a patch that stands down. This can:

```
python -m atom.plugin.vllm.doctor
```

It calls `register_platform()` and `register_model()` exactly as vLLM's entry
points do, then prints one row per patch: installed, stood down with a reason,
or raised. It exits non-zero when a patch did not install **and gave no
reason** — "something is missing and nobody knows what".

A patch that stands down *with* a reason is fine. Many are model-specific and
are expected not to apply.

### 5. Raise the tested ceiling

Only now. Edit `VLLM_TESTED_MAX` in
[`atom/plugin/vllm/vllm-version.env`](atom/plugin/vllm/vllm-version.env), and
`TESTED_MAX` in `_compat.py` to match. `_compat.check_version()` logs once when
the running vLLM is outside the range; it never refuses to run.

## When you add a patch

Decorate it, so it appears in the doctor's report:

```python
@_atom_patch("my_module.apply_my_patch")
def apply_my_patch() -> bool:
    try:
        from vllm.somewhere import Thing
    except ImportError as e:
        _atom_stood_down(f"vllm.somewhere.Thing unavailable: {e}")
        return False
    ...
```

Returning `False` without calling `stood_down` is what the doctor fails on.
That is intentional: a stand-down with no reason is the bug this whole
mechanism exists to make visible.

`tools/add_patch_decorators.py`-style mechanical application is how the
existing 48 were done; the decorator is idempotent and safe to re-apply after a
rebase onto a newer ATOM tag.

## Rebasing this branch onto a newer ATOM

All the new logic is in new files — `_compat.py`, `compat_check.py`,
`doctor.py`, `tests/test_vllm_compat.py`. Edits to pre-existing ATOM files are
one-line substitutions: an import routed through the shim, a decorator added, a
module tuple extended. `git rebase --onto <new-tag> v0.1.9 v0.1.9-vllm0.33`
should be close to clean; re-run step 1 afterwards regardless, since the new
tag may have added import sites of its own.

## Other packages

The checker is not vLLM-specific. ATOM has ~700 AITER call sites and declares
no `aiter` version either:

```
python -m atom.plugin.vllm.compat_check --package aiter --target /path/to/aiter
```

It also runs against any consumer, which is how LMCache v0.5.6 was checked
against vLLM 0.33:

```
python atom/plugin/vllm/compat_check.py \
    --target /path/to/vllm --consumer /path/to/lmcache
```

"""Report, without running anything, which of ATOM's imports a given vLLM
(or AITER) no longer satisfies.

    python -m atom.plugin.vllm.compat_check                      # installed vllm
    python -m atom.plugin.vllm.compat_check --target /src/vllm   # a checkout
    python -m atom.plugin.vllm.compat_check --package aiter
    python -m atom.plugin.vllm.compat_check --json

Why static. ATOM imports ~324 distinct symbols out of vLLM's private internals
and ~700 more out of AITER, and neither dependency is version-constrained in
``pyproject.toml``. Finding out what a new release broke by *running* it needs a
GPU, a model, ROCm, and a serving run that reaches the broken path -- which for
most of these is a specific model family on specific hardware. This reads both
sides as ASTs instead: no import, no torch, no GPU, no ROCm. It therefore runs
in a Dockerfile build step and in CI, and it finishes in under a second.

What it covers:

* every ``from <pkg>... import x`` and ``import <pkg>...`` under ``atom/``,
  resolved against the target's real module layout;
* whether an unresolved import sits inside a ``try:`` (guarded, survivable) or
  not (a crash on that path);
* whether a known relocation in ``_compat.RELOCATIONS`` covers it;
* **arity drift**: for every relocation candidate that declares its parameter
  list, whether the target's actual signature still matches. This is the check
  that catches a symbol which moved *and* changed shape -- the failure that
  reports no error at all, just a wrong number.

What it does NOT cover, and what ``doctor.py`` is for: monkeypatch targets that
still exist but whose *behaviour* changed, and patches that stand down at
runtime. A clean report here means "nothing will fail to import", not "the
plugin works".
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import sys


def _load_relocations():
    """``RELOCATIONS``, with or without an importable ``atom`` package.

    ``atom.plugin.vllm.__init__`` imports ``register``, which imports torch, so
    ``python -m atom.plugin.vllm.compat_check`` needs a torch install. The whole
    point of this checker is to run where there is none -- a CI container, a
    Dockerfile step before the wheel is installed -- so when the package import
    fails, load the sibling ``_compat.py`` straight off disk. It is written to
    have no third-party imports at module scope precisely so that this works.
    """
    try:
        from atom.plugin.vllm._compat import RELOCATIONS as table

        return table
    except ImportError:
        pass
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_compat.py")
    spec = importlib.util.spec_from_file_location("_atom_vllm_compat", path)
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: @dataclass resolves its own module out of
    # sys.modules while processing the class, and raises AttributeError on a
    # module that is not there yet.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.RELOCATIONS


RELOCATIONS = _load_relocations()


# ---------------------------------------------------------------------------
# The target package, read as source
# ---------------------------------------------------------------------------


def _read(path: str) -> str:
    """Source of one file, decoded leniently.

    ``errors="replace"`` because a stray byte in a vendored file is not a
    reason for a compatibility report to fail; ``ast.parse`` will reject it if
    it matters, and the caller treats a SyntaxError as "nothing to see here".
    """
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


class SourceTree:
    """A package on disk, queried without importing it."""

    def __init__(self, package: str, root: str):
        self.package = package
        self.root = root
        if not os.path.isdir(root):
            raise SystemExit(f"compat-check: no {package}/ package at {root}")
        # Per-instance, NOT functools.cache on the method: a cache on a bound
        # method is keyed on `self` and holds every tree ever parsed for the
        # lifetime of the process. These caches matter -- `names()` recurses
        # through `import *` chains -- but they belong to the instance, and
        # should die with it.
        self._ast_cache: dict = {}
        self._names_cache: dict = {}

    @classmethod
    def locate(cls, package: str, hint: str | None) -> SourceTree:
        """A checkout path, a package directory, or the installed package."""
        if hint:
            hint = os.path.abspath(os.path.expanduser(hint))
            # Accept either the repo root (containing vllm/) or the package dir.
            for cand in (os.path.join(hint, package), hint):
                if os.path.isdir(cand) and os.path.basename(cand) == package:
                    return cls(package, cand)
            raise SystemExit(f"compat-check: no {package}/ package under {hint}")
        spec = importlib.util.find_spec(package)
        if spec is None or not spec.submodule_search_locations:
            raise SystemExit(
                f"compat-check: {package} is not installed and no --target was given"
            )
        return cls(package, next(iter(spec.submodule_search_locations)))

    def version(self) -> str:
        """Best-effort version, read from source. Never imports.

        An unbuilt checkout has no ``_version.py`` -- setuptools_scm writes it
        at build time -- so ``version.py`` falls back to the literal "dev". Ask
        git in that case: the version is what the reader of this report needs
        most, and "dev" tells them nothing about which vLLM was checked.
        """
        from_source = self._version_from_source()
        if from_source not in ("dev", "unknown", "0.0.0"):
            return from_source
        from_git = self._version_from_git()
        return from_git or from_source

    def _version_from_git(self) -> str | None:
        import subprocess

        repo = os.path.dirname(self.root)
        try:
            out = subprocess.run(
                ["git", "-C", repo, "describe", "--tags", "--always"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() or None if out.returncode == 0 else None

    def _version_from_source(self) -> str:
        for name in ("version.py", "_version.py", "__init__.py"):
            path = os.path.join(self.root, name)
            if not os.path.isfile(path):
                continue
            try:
                tree = ast.parse(_read(path))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Assign)
                    and any(
                        isinstance(t, ast.Name) and t.id in ("__version__", "VERSION")
                        for t in node.targets
                    )
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                ):
                    return node.value.value
        return "unknown"

    # -- module layout ------------------------------------------------------

    def _path(self, module: str) -> str | None:
        head, _, rest = module.partition(".")
        if head != self.package:
            return None
        base = os.path.join(self.root, *rest.split(".")) if rest else self.root
        if os.path.isdir(base):
            init = os.path.join(base, "__init__.py")
            return init if os.path.isfile(init) else base
        for ext in (".py", ".pyi"):
            if os.path.isfile(base + ext):
                return base + ext
        return None

    def has_module(self, module: str) -> bool:
        return self._path(module) is not None

    def is_package(self, module: str) -> bool:
        _, _, rest = module.partition(".")
        base = os.path.join(self.root, *rest.split(".")) if rest else self.root
        return os.path.isdir(base)

    def _ast(self, module: str):
        if module in self._ast_cache:
            return self._ast_cache[module]
        path = self._path(module)
        tree = None
        if path is not None and path.endswith((".py", ".pyi")):
            try:
                tree = ast.parse(_read(path))
            except SyntaxError:
                tree = None
        self._ast_cache[module] = tree
        return tree

    def names(self, module: str) -> frozenset:
        """Top-level names a module binds, following ``import *`` one hop.

        Walks the whole tree rather than just the module body, so a symbol
        defined inside ``if TYPE_CHECKING:`` or a ``try/except ImportError``
        still counts as present -- which matches what ``getattr`` would find
        at runtime closely enough for this purpose, and errs towards *not*
        reporting a break that is not one.
        """
        if module in self._names_cache:
            return self._names_cache[module]
        # Seeded before the walk: a module in an ``import *`` cycle would
        # otherwise recurse forever.
        self._names_cache[module] = frozenset()
        tree = self._ast(module)
        if tree is None:
            return frozenset()
        out: set[str] = set()
        star_from: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.add(node.name)
            elif isinstance(node, ast.Assign):
                out.update(t.id for t in node.targets if isinstance(t, ast.Name))
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    out.add(node.target.id)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    out.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = module.split(".")
                    pkg = parts if self.is_package(module) else parts[:-1]
                    pkg = pkg[: len(pkg) - (node.level - 1)]
                    base = ".".join(pkg + ([base] if base else []))
                for alias in node.names:
                    if alias.name == "*":
                        star_from.append(base)
                    else:
                        out.add(alias.asname or alias.name)
        for base in star_from:
            if base.startswith(self.package) and base != module:
                out |= set(self.names(base))
        resolved = frozenset(out)
        self._names_cache[module] = resolved
        return resolved

    def resolves(self, module: str, name: str | None) -> str | None:
        if not self.has_module(module):
            return "missing-module"
        if name is None or name in self.names(module):
            return None
        if self.has_module(f"{module}.{name}"):
            return None
        return "missing-symbol"

    def params(self, module: str, name: str) -> tuple[str, ...] | None:
        """Positional parameter names of a top-level function, or ``None``."""
        tree = self._ast(module)
        if tree is None:
            return None
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                node.name == name
            ):
                args = node.args
                return tuple(
                    a.arg for a in (*args.posonlyargs, *args.args) if a.arg != "self"
                )
        return None


# ---------------------------------------------------------------------------
# The consumer side
# ---------------------------------------------------------------------------


def _atom_root() -> str:
    # .../atom/plugin/vllm/compat_check.py -> .../atom
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def scan(consumer_root: str, package: str) -> list[dict]:
    """Every runtime import of ``package`` under ``consumer_root``, grouped.

    Three things make this more than a grep for ``from vllm import``, and all
    three were put in by a false positive this tool produced:

    ``TYPE_CHECKING`` blocks are skipped. They are never executed, so an import
    inside one cannot fail. LMCache's vLLM adapters put
    ``vllm.attention.backends.abstract.AttentionMetadata`` there -- a module
    that exists at neither 0.31 nor 0.33 -- and reporting those three as
    breakage would have been three wrong answers in a five-line report.

    A ``try/except ImportError`` ladder is one group, because its members are
    *alternatives*. ATOM writes the current location in the ``try`` and the
    older one in the handler, so the handler's import is expected to fail on a
    new vLLM and expected never to run there. The group is satisfied when any
    one member resolves.

    A handler that copes satisfies the group by itself. Three shapes of that:
    it defines the name (``def get_kv_cache_shard_count(): ...``), it assigns
    it, or it does neither and simply carries on -- logging and returning
    ``None`` is a complete answer to "this vLLM cannot tell me the layout". A
    handler that *re-raises* is not coping: it is improving the error message,
    and the import behind it is still a break.
    """
    found: list[dict] = []
    for dirpath, dirnames, filenames in os.walk(consumer_root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__")]
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(dirpath, filename)
            try:
                tree = ast.parse(_read(path))
            except SyntaxError:
                continue
            rel = os.path.relpath(path, os.path.dirname(consumer_root))
            found.extend(_scan_module(tree, rel, package))
    return found


class _Ladders:
    """Ladder ids and what each one's handlers supply, assigned during the walk."""

    def __init__(self):
        self.next_id = 0
        self.provides: dict[int, set[str]] = {}
        self.coped: dict[int, bool] = {}

    def new(self, node, package: str) -> int:
        ladder = self.next_id
        self.next_id += 1
        defined: set[str] = set()
        coped = False
        for handler in node.handlers:
            if not _catches_import(handler):
                continue
            for stmt in handler.body:
                if isinstance(
                    stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    defined.add(stmt.name)
                elif isinstance(stmt, ast.Assign):
                    defined.update(
                        t.id for t in stmt.targets if isinstance(t, ast.Name)
                    )
            reraises = any(isinstance(n, ast.Raise) for n in ast.walk(handler))
            imports_target = any(
                _imports_package(n, package) for n in ast.walk(handler)
            )
            if not reraises and not imports_target:
                coped = True
        self.provides[ladder] = defined
        self.coped[ladder] = coped
        return ladder


def _imports_package(node, package: str) -> bool:
    if isinstance(node, ast.ImportFrom):
        return not node.level and _is_target(node.module or "", package)
    if isinstance(node, ast.Import):
        return any(_is_target(a.name, package) for a in node.names)
    return False


def _scan_module(tree, rel: str, package: str) -> list[dict]:
    out: list[dict] = []
    ladders = _Ladders()

    def visit(node, type_checking: bool, ladder: int | None) -> None:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if not type_checking:
                out.extend(_entries(node, rel, package, ladder, ladders))
            return
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            for child in node.body:
                visit(child, True, ladder)
            for child in node.orelse:
                visit(child, type_checking, ladder)
            return
        if isinstance(node, ast.Try) and any(_catches_import(h) for h in node.handlers):
            # An inner ladder inside an outer one keeps the outer group: the
            # outer handler is still the fallback that decides the outcome.
            inner = ladder if ladder is not None else ladders.new(node, package)
            for body in [node.body] + [h.body for h in node.handlers]:
                for child in body:
                    visit(child, type_checking, inner)
            for child in list(node.orelse) + list(node.finalbody):
                visit(child, type_checking, ladder)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, type_checking, ladder)

    visit(tree, False, None)
    return out


def _entries(node, rel: str, package: str, ladder, ladders) -> list[dict]:
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        if node.level or not _is_target(module, package):
            return []
        aliases = list(node.names)
    else:
        aliases = [a for a in node.names if _is_target(a.name, package)]
        module = None
    out = []
    for alias in aliases:
        if module is None:
            entry_module, entry_name = alias.name, None
        else:
            entry_module, entry_name = module, alias.name
        # The name the statement BINDS, not the one it reads. A fallback arm
        # routinely renames -- `from vllm.utils import get_cu_count as
        # num_compute_units` supplies the same capability the try arm asks for
        # under the name `num_compute_units` -- so grouping on the source name
        # splits one ladder in two and reports both halves as broken.
        bound = alias.asname or alias.name.split(".")[0]
        out.append(
            {
                "file": rel,
                "line": node.lineno,
                "module": entry_module,
                "name": entry_name,
                "group": (
                    f"{rel}#ladder{ladder}:{bound}"
                    if ladder is not None
                    else f"{rel}:{node.lineno}:{entry_module}:{entry_name}"
                ),
                "fallback_defined": ladder is not None
                and (
                    bound in ladders.provides.get(ladder, ())
                    or ladders.coped.get(ladder, False)
                ),
            }
        )
    return out


def _is_type_checking(test) -> bool:
    """``if TYPE_CHECKING:`` / ``if typing.TYPE_CHECKING:``, and the ``not`` of
    neither -- a negated test guards runtime code, which is exactly the code
    this tool must keep looking at."""
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def _is_target(module: str, package: str) -> bool:
    return module == package or module.startswith(package + ".")


_IMPORT_ERRORS = {"ImportError", "ModuleNotFoundError", "AttributeError", "Exception"}


def _catches_import(handler) -> bool:
    if handler.type is None:  # bare except
        return True
    names = (
        [handler.type]
        if not isinstance(handler.type, ast.Tuple)
        else list(handler.type.elts)
    )
    return any(isinstance(n, ast.Name) and n.id in _IMPORT_ERRORS for n in names)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyse(tree: SourceTree, consumer_root: str) -> dict:
    imports = scan(consumer_root, tree.package)

    groups: dict[str, list[dict]] = {}
    for imp in imports:
        groups.setdefault(imp["group"], []).append(imp)

    distinct: set[tuple] = {(i["module"], i["name"]) for i in imports}
    unresolved: list[dict] = []
    for members in groups.values():
        verdicts = [(m, tree.resolves(m["module"], m["name"])) for m in dedupe(members)]
        if any(v is None for _, v in verdicts):
            continue  # some alternative in this group resolves
        if any(m.get("fallback_defined") for m in members):
            continue  # the handler supplies the name itself
        # Report the group's primary -- the first-written alternative, which is
        # the one the author expects a current vLLM to satisfy.
        primary, verdict = verdicts[0]
        relocation_key = f"{primary['module']}:{primary['name']}"
        unresolved.append(
            {
                **primary,
                "verdict": verdict,
                "shim": _covered_by_shim(tree, relocation_key),
                "key": relocation_key,
                "alternatives": [f"{m['module']}:{m['name']}" for m, _ in verdicts[1:]],
            }
        )

    drift = _arity_drift(tree)
    relocations = _relocation_status(tree)
    patch_targets = _missing_patch_targets(tree, consumer_root)
    # "covered" is NOT a pass. A relocation entry says where the symbol went;
    # it does nothing for a call site that still writes the old import
    # literally -- and the only reason such an import is here to be found is
    # that the site does not go through the shim. Treating "the table knows"
    # as "the code is fixed" would pass a tree that still raises ImportError
    # on exactly the symbol the table was added for.
    blocking = [u for u in unresolved if u["shim"] != "optional"]
    blocking += [r for r in relocations if r["status"] == "broken"]
    blocking += patch_targets
    return {
        "package": tree.package,
        "target": tree.root,
        "version": tree.version(),
        "distinct_imports": len(distinct),
        "import_sites": len(imports),
        "unresolved": unresolved,
        "relocations": relocations,
        "patch_targets": patch_targets,
        "blocking": blocking,
        "arity_drift": drift,
    }


def _missing_patch_targets(tree: SourceTree, consumer_root: str) -> list[dict]:
    """Monkeypatch targets that no longer exist on the module named.

    The failure import checking is blind to. ``import vllm.v1.core.kv_cache_utils
    as m`` still succeeds at 0.33 -- the module is there -- but every function
    ATOM patches onto it moved to ``kv_cache_planning``. Depending on how the
    installer reads them, that is either an ``AttributeError`` from inside
    registration or, worse, a ``getattr(m, name, None)`` that returns ``None``
    and installs nothing at all while reporting success.

    Only ``import x.y as alias`` / ``alias.attr`` is resolved, because that is
    the shape a module-level monkeypatch takes. An attribute read through a
    name held in a variable (``getattr(module, _ATTR)``) is out of reach of any
    static check and belongs to ``doctor.py``.
    """
    out: list[dict] = []
    for dirpath, dirnames, filenames in os.walk(consumer_root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__")]
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(dirpath, filename)
            try:
                parsed = ast.parse(_read(path))
            except SyntaxError:
                continue
            aliases = {}
            for node in ast.walk(parsed):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if _is_target(alias.name, tree.package):
                            aliases[alias.asname or alias.name.split(".")[0]] = (
                                alias.name
                            )
            if not aliases:
                continue
            rel = os.path.relpath(path, os.path.dirname(consumer_root))
            seen: set[tuple] = set()
            for node in ast.walk(parsed):
                if not isinstance(node, ast.Attribute) or not isinstance(
                    node.value, ast.Name
                ):
                    continue
                module = aliases.get(node.value.id)
                # `_atom_*` attributes are ATOM's own idempotency markers,
                # written onto vLLM modules by these very patches. Expecting
                # vLLM to define them would report every marker as missing.
                if module is None or node.attr.startswith("_atom"):
                    continue
                key = (module, node.attr)
                if key in seen or tree.resolves(module, node.attr) is None:
                    continue
                seen.add(key)
                out.append(
                    {
                        "file": rel,
                        "line": node.lineno,
                        "module": module,
                        "attr": node.attr,
                    }
                )
    return out


def _relocation_status(tree: SourceTree) -> list[dict]:
    """Check every entry in the table, not just the ones still imported.

    This is the half that keeps working as the fix lands. Once a call site is
    rewritten to go through ``_compat.resolve``, it contains no ``from vllm...
    import`` for the scanner to find -- so the import scan goes quiet on
    exactly the symbols most likely to move again. Resolving the table
    directly is what keeps a future relocation of, say,
    ``kv_cache_planning`` visible.
    """
    out: list[dict] = []
    for key, rel in sorted(RELOCATIONS.items()):
        if not key.startswith(tree.package + "."):
            continue
        resolved_at = None
        for cand in rel.candidates:
            if tree.resolves(cand.module, cand.name) is None:
                resolved_at = f"{cand.module}:{cand.name}"
                break
        if resolved_at:
            status = "ok"
        elif getattr(rel, "optional", False):
            status = "absent (optional)"
        else:
            status = "broken"
        out.append(
            {
                "key": key,
                "status": status,
                "resolved_at": resolved_at,
                "candidates": [f"{c.module}:{c.name}" for c in rel.candidates],
                "note": rel.note,
            }
        )
    return out


def dedupe(members: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    out = []
    for m in sorted(members, key=lambda m: m["line"]):
        key = (m["module"], m["name"])
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out


def _covered_by_shim(tree: SourceTree, key: str) -> str:
    """Whether ``_compat.RELOCATIONS`` already knows where this symbol went.

    ``optional`` is a third answer, not a softer "stale": the entry declares
    that resolving nothing is a supported outcome and the call site does
    without. Reporting those as breaks would train the reader to ignore the
    report, which is the one thing it cannot survive.
    """
    rel = RELOCATIONS.get(key)
    if rel is None:
        return "none"
    for cand in rel.candidates:
        if tree.resolves(cand.module, cand.name) is None:
            return "covered"
    return "optional" if getattr(rel, "optional", False) else "stale"


def _arity_drift(tree: SourceTree) -> list[dict]:
    """Relocation candidates whose declared parameters no longer match.

    The check that would have caught ``_pool_bytes_per_block`` gaining a
    ``vllm_config`` parameter in 0.33 -- a change that raises nothing and
    returns a wrong number.
    """
    out: list[dict] = []
    for key, rel in RELOCATIONS.items():
        for cand in rel.candidates:
            if cand.params is None:
                continue
            if tree.resolves(cand.module, cand.name) is not None:
                continue
            actual = tree.params(cand.module, cand.name)
            if actual is None or tuple(actual) == tuple(cand.params):
                continue
            out.append(
                {
                    "key": key,
                    "at": f"{cand.module}:{cand.name}",
                    "declared": list(cand.params),
                    "actual": list(actual),
                }
            )
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def render(result: dict) -> str:
    pkg, version = result["package"], result["version"]
    lines = [
        f"ATOM compat check: {pkg} {version}",
        f"  target     {result['target']}",
        (
            f"  imports    {result['import_sites']} sites, "
            f"{result['distinct_imports']} distinct symbols, "
            f"{len(result['unresolved'])} unresolved, "
            f"{len(result['blocking'])} blocking"
        ),
        "",
    ]
    if result["unresolved"]:
        lines.append("Unresolved imports (no alternative in the group resolves)")
        for u in sorted(result["unresolved"], key=lambda u: (u["file"], u["line"])):
            flags = []
            if u.get("alternatives"):
                flags.append(f"{len(u['alternatives'])} alt also failed")
            if u["shim"] == "covered":
                flags.append("RELOCATION KNOWN - route this site through _compat")
            elif u["shim"] == "optional":
                flags.append("optional")
            elif u["shim"] == "stale":
                flags.append("SHIM STALE")
            tag = f" [{', '.join(flags)}]" if flags else ""
            name = u["name"] or "<module>"
            lines.append(
                f"  {u['verdict']:15} {u['module']}:{name}{tag}\n"
                f"    {u['file']}:{u['line']}"
            )
        lines.append("")
    if result.get("patch_targets"):
        lines.append("Monkeypatch targets that no longer exist")
        for t in result["patch_targets"]:
            lines.append(
                f"  ! {t['module']}.{t['attr']}\n      {t['file']}:{t['line']}"
            )
        lines.append("")
    if result.get("relocations"):
        lines.append("Relocation table")
        for r in result["relocations"]:
            where = r["resolved_at"] or "nowhere"
            mark = "! " if r["status"] == "broken" else "  "
            lines.append(f"  {mark}{r['key']}\n      {r['status']} -> {where}")
        lines.append("")
    if result["arity_drift"]:
        lines.append("Arity drift (resolves, but the signature changed)")
        for d in result["arity_drift"]:
            lines.append(
                f"  {d['at']}\n"
                f"    _compat declares ({', '.join(d['declared'])})\n"
                f"    {pkg} has        ({', '.join(d['actual'])})"
            )
        lines.append("")
    if not result["blocking"] and not result["arity_drift"]:
        lines.append("OK: nothing unresolved that is not guarded or shimmed.")
    else:
        lines.append(
            "FAIL: add the new location to RELOCATIONS in "
            "atom/plugin/vllm/_compat.py, then re-run. See UPGRADING-VLLM.md."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m atom.plugin.vllm.compat_check",
        description="Resolve ATOM's imports of vLLM (or AITER) statically.",
    )
    parser.add_argument("--package", default="vllm", choices=("vllm", "aiter"))
    parser.add_argument(
        "--target",
        default=None,
        help="path to a checkout or package dir; default is the installed one",
    )
    parser.add_argument(
        "--consumer",
        default=None,
        help="path to the atom package; default is the one containing this file",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    tree = SourceTree.locate(args.package, args.target)
    result = analyse(tree, args.consumer or _atom_root())
    if args.json:
        print(json.dumps(result, indent=1, sort_keys=True))
    else:
        print(render(result))
    return 1 if (result["blocking"] or result["arity_drift"]) else 0


if __name__ == "__main__":
    sys.exit(main())

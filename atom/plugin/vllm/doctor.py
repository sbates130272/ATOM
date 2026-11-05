"""Run ATOM's vLLM registration hooks and report what each patch actually did.

    python -m atom.plugin.vllm.doctor
    python -m atom.plugin.vllm.doctor --json

Needs vLLM, torch and a GPU visible -- unlike ``compat_check``, which needs
none of those. The division of labour is deliberate:

    compat_check   can a symbol be found?        static, no GPU, runs in CI
    doctor         did the patch actually go in?  runtime, needs the stack

``compat_check`` cannot see a patch that stands down. ATOM's installers find
their targets with ``getattr(module, name, None)`` and return quietly when the
answer is ``None``; ``register.py`` records one occasion where "the dummy-run
marking once covered nothing while the synthetic batches went on taking real
STATE slots." The whole plugin is ~48 such installers, and before this there
was no way to ask how many of them were in place.

What it does: imports vLLM, calls ``register_platform`` and ``register_model``
exactly as vLLM's entry points would, and prints one row per patch --
installed, stood down with a reason, or raised. Exit status is non-zero if any
patch raised or stood down without a reason, which is the state worth failing a
deployment check on: "something did not install and nobody knows why".

A patch that stands down WITH a reason is reported and tolerated. Many of them
are model-specific by design and are expected not to apply.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys


def run_hooks() -> list[str]:
    """Invoke both registration hooks; return the problems that stopped them."""
    problems = []
    from atom.plugin.vllm import _compat

    _compat.check_version()

    from atom.plugin.vllm.register import register_model, register_platform

    try:
        platform = register_platform()
        print(f"register_platform() -> {platform}")
    except Exception as exc:  # noqa: BLE001 - reported, this is a diagnostic
        problems.append(f"register_platform raised {type(exc).__name__}: {exc}")
        print(f"register_platform() RAISED {type(exc).__name__}: {exc}")

    try:
        register_model()
        print("register_model() -> ok")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"register_model raised {type(exc).__name__}: {exc}")
        print(f"register_model() RAISED {type(exc).__name__}: {exc}")

    return problems


def render(records, version: str) -> str:
    width = max([len(r.name) for r in records], default=10)
    lines = [
        "",
        f"ATOM plugin patch report against vLLM {version}",
        f"  {len(records)} patches invoked",
        "",
    ]
    for rec in records:
        reason = "; ".join(rec.reasons) or rec.error
        suffix = f"  <- {reason}" if reason else ""
        calls = f" (x{rec.calls})" if rec.calls > 1 else ""
        lines.append(f"  {rec.name:<{width}}  {rec.status}{calls}{suffix}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m atom.plugin.vllm.doctor",
        description="Run ATOM's vLLM hooks and report every patch's outcome.",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--verbose", action="store_true", help="show ATOM's own log output"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.ERROR,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from atom.plugin.vllm import _compat

    problems = run_hooks()
    records = _compat.report()
    version = _compat.version_str()

    unexplained = [r for r in records if r.error or (not r.installed and not r.reasons)]

    if args.json:
        print(
            json.dumps(
                {
                    "vllm": version,
                    "problems": problems,
                    "patches": [
                        {
                            "name": r.name,
                            "status": r.status,
                            "calls": r.calls,
                            "reasons": r.reasons,
                            "error": r.error,
                        }
                        for r in records
                    ],
                },
                indent=1,
            )
        )
    else:
        print(render(records, version))
        if problems:
            print("\nHook problems:")
            for problem in problems:
                print(f"  ! {problem}")
        if unexplained:
            print(
                "\nFAIL: "
                f"{len(unexplained)} patch(es) did not install and gave no reason. "
                "A patch whose vLLM target moved stands down silently; that is "
                "what this is for. Run `python -m atom.plugin.vllm.compat_check` "
                "to see whether a symbol relocated."
            )
        elif not problems:
            print("\nOK: every invoked patch installed, or stood down with a reason.")

    return 1 if (problems or unexplained) else 0


if __name__ == "__main__":
    sys.exit(main())

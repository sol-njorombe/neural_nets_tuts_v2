"""Generate the learner stubs from the reference solutions.

    python tools/make_stubs.py           # write lessonXX/<name>.py from solutions/lessonXX/<name>.py
    python tools/make_stubs.py --check   # exit 1 if any stub is out of date (used in CI)

In a solution file, each answer sits between `# BEGIN SOLUTION` and
`# END SOLUTION`. In the stub the block becomes

    raise NotImplementedError("TODO: implement <enclosing function>()")

or, with `# BEGIN SOLUTION stub: <line>`, that line instead.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BEGIN = "# BEGIN SOLUTION"
END = "# END SOLUTION"
DEF = re.compile(r"\s*def (\w+)\(")


def to_stub(source: str, path: pathlib.Path) -> str:
    out: list[str] = []
    in_solution = False
    current_function = "this"
    for number, line in enumerate(source.split("\n"), start=1):
        stripped = line.strip()
        match = DEF.match(line)
        if match and not in_solution:
            current_function = match.group(1)
        if stripped.startswith(BEGIN):
            if in_solution:
                raise SystemExit(f"{path}:{number}: nested BEGIN SOLUTION")
            in_solution = True
            indent = line[: len(line) - len(line.lstrip())]
            custom = stripped[len(BEGIN) :].strip()
            if custom.startswith("stub:"):
                out.append(indent + custom[len("stub:") :].strip())
            else:
                out.append(f'{indent}raise NotImplementedError("TODO: implement {current_function}()")')
        elif stripped.startswith(END):
            if not in_solution:
                raise SystemExit(f"{path}:{number}: END SOLUTION without BEGIN")
            in_solution = False
        elif not in_solution:
            out.append(line)
    if in_solution:
        raise SystemExit(f"{path}: BEGIN SOLUTION never closed")
    return "\n".join(out)


def main() -> int:
    check = "--check" in sys.argv
    stale: list[str] = []
    for solution in sorted((ROOT / "solutions").glob("lesson*/*.py")):
        stub_path = ROOT / solution.parent.name / solution.name
        stub = to_stub(solution.read_text(), solution)
        if stub_path.exists() and stub_path.read_text() == stub:
            continue
        if check:
            stale.append(str(stub_path.relative_to(ROOT)))
        else:
            stub_path.parent.mkdir(parents=True, exist_ok=True)
            stub_path.write_text(stub)
            print("wrote", stub_path.relative_to(ROOT))
    if stale:
        print("Stubs out of date (run python tools/make_stubs.py):", *stale, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

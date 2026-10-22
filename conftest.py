"""Shared pytest setup for the whole course.

    pytest lesson01_neuron      -> tests YOUR code in that lesson
    pytest --solutions          -> tests the reference solutions instead
"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).parent


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--solutions",
        action="store_true",
        help="run the tests against the reference solutions instead of your code",
    )


def pytest_configure(config: pytest.Config) -> None:
    # Each lesson folder holds your code plus its given helper files.
    for folder in sorted(ROOT.glob("lesson*")):
        sys.path.insert(0, str(folder))
    # With --solutions, the reference answers go first on the path, so they win.
    if config.getoption("--solutions"):
        for folder in sorted((ROOT / "solutions").glob("lesson*")):
            sys.path.insert(0, str(folder))


def pytest_report_header(config: pytest.Config) -> str:
    if config.getoption("--solutions"):
        return "Testing: the REFERENCE SOLUTIONS"
    return "Testing: YOUR code"

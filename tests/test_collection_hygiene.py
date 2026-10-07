"""
A bare `pytest` collects the maintained suite in tests/ and nothing else.

Before pytest.ini existed, `pytest` walked the whole repo: it imported the
root scratch script test_repository.py (which ran insert_drift_event
against the configured database at import time) and errored on the stale
src/rag/test_chroma.py. Those scripts are gone; these guards keep new
ones from creeping back. Static checks only.
"""

import ast
import configparser
import fnmatch
import os


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
TESTS_DIR = os.path.join(REPO_ROOT, "tests")

# Scratch-test names. "*_test.py" is not listed: production modules use
# it (src/evaluation/recent_test.py, src/training/build_frozen_test.py),
# so those only count when they actually define tests.
TEST_FILE_PATTERNS = ("test_*.py", "test.py")

# Not code we own, or too large to walk
SKIP_DIRS = {
    ".git", "venv", ".venv", "data", "model", "reports", "mlruns",
    "__pycache__", ".pytest_cache", "node_modules",
}


def test_pytest_ini_points_at_tests_only():

    config = configparser.ConfigParser()
    config.read(os.path.join(REPO_ROOT, "pytest.ini"), encoding="utf-8")

    assert config.get("pytest", "testpaths").split() == ["tests"]


def test_no_test_files_outside_tests_dir():

    strays = []

    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS
            and os.path.join(dirpath, d) != TESTS_DIR
        ]
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            if (
                any(fnmatch.fnmatch(filename, p) for p in TEST_FILE_PATTERNS)
                or (fnmatch.fnmatch(filename, "*_test.py")
                    and _defines_tests(path))
            ):
                strays.append(os.path.relpath(path, REPO_ROOT))

    assert strays == []


def _defines_tests(path):

    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())

    return any(
        (isinstance(node, ast.FunctionDef) and node.name.startswith("test"))
        or (isinstance(node, ast.ClassDef) and node.name.startswith("Test"))
        for node in tree.body
    )


def test_production_modules_named_like_tests_define_no_tests():
    """The two *_test.py production modules stay non-test code."""

    for relative in ("src/evaluation/recent_test.py",
                     "src/training/build_frozen_test.py"):
        assert not _defines_tests(os.path.join(REPO_ROOT, *relative.split("/")))


def test_test_modules_have_no_import_time_calls():
    """
    Importing a test module may define tests, fixtures and constants but
    must not act. conftest.py is exempt: it installs the stub modules that
    keep the suite away from real services.
    """

    offenders = []

    for filename in sorted(os.listdir(TESTS_DIR)):
        if not fnmatch.fnmatch(filename, "test_*.py"):
            continue
        with open(os.path.join(TESTS_DIR, filename), encoding="utf-8") as f:
            tree = ast.parse(f.read())

        for node in tree.body:
            # A bare call statement, or a loop / with block at module level
            bare_call = (
                isinstance(node, ast.Expr)
                and not isinstance(node.value, ast.Constant)
            )
            if bare_call or isinstance(node, (ast.For, ast.While, ast.With)):
                offenders.append(f"{filename}:{node.lineno}")

    assert offenders == []

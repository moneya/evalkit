"""Tests for .env loading.

Credentials in a gitignored file beat a shell `export` that dies with the
window. The rules that actually matter and so are tested hardest:

* the real environment always wins over the file
* nothing ever logs a value
* a malformed line is skipped, not fatal
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from evalkit.env import find_env_file, load_env, parse_env


# -- parsing --------------------------------------------------------------

@pytest.mark.parametrize("line,key,value", [
    ("A=1", "A", "1"),
    ("A = 1", "A", "1"),
    ("  A=1  ", "A", "1"),
    ("export A=1", "A", "1"),
    ("export  A = 1", "A", "1"),
    ('A="quoted"', "A", "quoted"),
    ("A='single'", "A", "single"),
    ('A="has # hash"', "A", "has # hash"),
    ("A=val # trailing comment", "A", "val"),
    ("A=", "A", ""),
    ("A=sk-ant-with=equals=inside", "A", "sk-ant-with=equals=inside"),
    ("A=https://host:8000/v1", "A", "https://host:8000/v1"),
])
def test_parse_shapes(line, key, value):
    assert parse_env(line) == {key: value}


@pytest.mark.parametrize("line", [
    "# comment",
    "",
    "   ",
    "no_equals_sign",
    "=novalue",
    "9STARTS_WITH_DIGIT=x",
])
def test_lines_that_yield_nothing(line):
    assert parse_env(line) == {}


def test_malformed_line_does_not_discard_the_rest():
    """A typo on line 2 must not cost you the key on line 3."""
    parsed = parse_env("A=1\nthis line is broken\nB=2")
    assert parsed == {"A": "1", "B": "2"}


def test_later_duplicate_wins():
    assert parse_env("A=first\nA=second") == {"A": "second"}


# -- precedence -----------------------------------------------------------

def test_real_environment_wins_over_file(tmp_path, monkeypatch):
    """The whole point: `KEY=x evalkit run` and CI secrets beat a stale file."""
    (tmp_path / ".env").write_text("MY_KEY=from-file\n")
    monkeypatch.setenv("MY_KEY", "from-environment")
    applied = load_env(tmp_path / ".env")
    assert os.environ["MY_KEY"] == "from-environment"
    assert "MY_KEY" not in applied


def test_override_is_opt_in(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("MY_KEY=from-file\n")
    monkeypatch.setenv("MY_KEY", "from-environment")
    load_env(tmp_path / ".env", override=True)
    assert os.environ["MY_KEY"] == "from-file"


def test_empty_existing_value_is_treated_as_unset(tmp_path, monkeypatch):
    """An exported-but-blank var should not shadow a real value in the file."""
    (tmp_path / ".env").write_text("MY_KEY=from-file\n")
    monkeypatch.setenv("MY_KEY", "")
    load_env(tmp_path / ".env")
    assert os.environ["MY_KEY"] == "from-file"


def test_file_populates_when_unset(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("BRAND_NEW_VAR=hello\n")
    monkeypatch.delenv("BRAND_NEW_VAR", raising=False)
    applied = load_env(tmp_path / ".env")
    assert os.environ["BRAND_NEW_VAR"] == "hello"
    assert applied == ["BRAND_NEW_VAR"]


# -- discovery ------------------------------------------------------------

def test_found_from_a_subdirectory(tmp_path):
    (tmp_path / ".env").write_text("A=1\n")
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    assert find_env_file(deep) == tmp_path / ".env"


def test_nearest_file_wins(tmp_path):
    (tmp_path / ".env").write_text("A=outer\n")
    inner = tmp_path / "inner"
    inner.mkdir()
    (inner / ".env").write_text("A=inner\n")
    assert find_env_file(inner) == inner / ".env"


def test_missing_file_is_not_an_error(tmp_path):
    assert find_env_file(tmp_path) is None
    assert load_env(tmp_path / "nope.env") == []


def test_unreadable_file_is_not_fatal(tmp_path):
    p = tmp_path / ".env"
    p.write_text("A=1\n")
    p.chmod(0o000)
    try:
        assert load_env(p) == []
    finally:
        p.chmod(0o600)


# -- secrecy --------------------------------------------------------------

def test_load_returns_names_never_values(tmp_path, monkeypatch):
    monkeypatch.delenv("SECRET_TOKEN", raising=False)
    (tmp_path / ".env").write_text("SECRET_TOKEN=sk-super-secret-value\n")
    applied = load_env(tmp_path / ".env")
    assert applied == ["SECRET_TOKEN"]
    assert "sk-super-secret-value" not in repr(applied)


def test_debug_output_prints_names_not_values(tmp_path):
    """EVALKIT_DEBUG exists to answer 'did my key load?' — not to show the key."""
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=sk-leak-me-if-you-can\n")
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    env.update({"EVALKIT_DEBUG": "1", "NO_COLOR": "1", "PATH": os.environ["PATH"]})
    r = subprocess.run(
        [sys.executable, "-m", "evalkit.cli", "providers"],
        cwd=tmp_path, capture_output=True, text=True, env=env,
    )
    combined = r.stdout + r.stderr
    assert "DEEPSEEK_API_KEY" in combined
    assert "sk-leak-me-if-you-can" not in combined


# -- cli integration ------------------------------------------------------

def test_cli_loads_env_from_cwd(tmp_path):
    (tmp_path / ".env").write_text("EVALKIT_PROBE_VAR=loaded\n")
    code = (
        "import os, sys; sys.argv=['evalkit','providers'];"
        "from evalkit.cli import main; main();"
        "print('PROBE=' + os.environ.get('EVALKIT_PROBE_VAR','MISSING'))"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                       capture_output=True, text=True,
                       env={**os.environ, "NO_COLOR": "1"})
    assert "PROBE=loaded" in r.stdout


def test_cli_no_env_file_flag_skips_loading(tmp_path):
    (tmp_path / ".env").write_text("EVALKIT_PROBE_VAR=loaded\n")
    code = (
        "import os, sys; sys.argv=['evalkit','--no-env-file','providers'];"
        "from evalkit.cli import main; main();"
        "print('PROBE=' + os.environ.get('EVALKIT_PROBE_VAR','MISSING'))"
    )
    env = {k: v for k, v in os.environ.items() if k != "EVALKIT_PROBE_VAR"}
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                       capture_output=True, text=True, env={**env, "NO_COLOR": "1"})
    assert "PROBE=MISSING" in r.stdout


def test_explicit_env_file_flag(tmp_path):
    custom = tmp_path / "secrets.env"
    custom.write_text("EVALKIT_PROBE_VAR=from-custom-path\n")
    code = (
        f"import os, sys; sys.argv=['evalkit','--env-file',r'{custom}','providers'];"
        "from evalkit.cli import main; main();"
        "print('PROBE=' + os.environ.get('EVALKIT_PROBE_VAR','MISSING'))"
    )
    env = {k: v for k, v in os.environ.items() if k != "EVALKIT_PROBE_VAR"}
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                       capture_output=True, text=True, env={**env, "NO_COLOR": "1"})
    assert "PROBE=from-custom-path" in r.stdout


# -- repo hygiene ---------------------------------------------------------

def test_env_is_gitignored_but_example_is_not():
    """A committed .env is the failure mode this feature could cause."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    ignore = (root / ".gitignore").read_text()
    assert "\n.env\n" in ignore
    assert "!.env.example" in ignore
    assert (root / ".env.example").is_file()


def test_env_example_contains_no_real_looking_key():
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    text = (root / ".env.example").read_text()
    for line in text.splitlines():
        if line.startswith("#") or "=" not in line:
            continue
        _, _, value = line.partition("=")
        assert not re.match(r"(sk-|nvapi-|gsk_)\S{8,}", value.strip()), line

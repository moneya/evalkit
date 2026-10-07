"""`.env` loading.

Keys belong in a gitignored file, not in a shell `export` that dies with the
window. This is deliberately a ~60-line reader rather than a dependency: an
eval harness should install with as little surface as possible, and the subset
of the format that matters here is small.

Supported, because real `.env` files in the wild contain all of it:

    KEY=value
    KEY = value            # spaces around the separator
    export KEY=value       # pasted from a shell snippet
    KEY="value with #"     # quotes protect a literal hash
    KEY='single'
    # comment
    KEY=                   # explicitly empty

Rules that matter:

* **The real environment always wins.** A value already in `os.environ` is
  never overwritten, so `DEEPSEEK_API_KEY=... evalkit run` and CI secrets still
  take precedence over a stale local file.
* **Search walks upward** from the working directory, so `evalkit` works from a
  subdirectory of the project the way `git` does.
* **Nothing is ever logged.** Callers get the names that were loaded, never the
  values, because the first instinct on a bad key is to print it.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_FILENAME = ".env"
_MAX_PARENTS = 6


def find_env_file(start: Path | str | None = None, filename: str = DEFAULT_FILENAME) -> Path | None:
    """Nearest `filename` at or above `start`. Returns None if there is none."""
    here = Path(start or Path.cwd()).resolve()
    if here.is_file():
        here = here.parent
    for directory in [here, *list(here.parents)[:_MAX_PARENTS]]:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


def parse_env(text: str) -> dict[str, str]:
    """Parse `.env` content. Malformed lines are skipped, not raised on."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or not (key[0].isalpha() or key[0] == "_"):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            # An unquoted trailing comment is a comment; a quoted one is data.
            value = value.split(" #", 1)[0].rstrip()
        out[key] = value
    return out


def load_env(
    path: Path | str | None = None,
    *,
    override: bool = False,
) -> list[str]:
    """Load a `.env` into os.environ. Returns the key NAMES applied, never values.

    Existing environment variables are preserved unless `override=True`, so an
    inline `KEY=... evalkit run` and CI secrets beat a checked-out local file.
    """
    env_path = Path(path) if path else find_env_file()
    if env_path is None or not Path(env_path).is_file():
        return []

    try:
        text = Path(env_path).read_text(encoding="utf-8")
    except OSError:
        return []

    applied: list[str] = []
    for key, value in parse_env(text).items():
        if override or not os.environ.get(key):
            os.environ[key] = value
            applied.append(key)
    return applied


__all__ = ["DEFAULT_FILENAME", "find_env_file", "load_env", "parse_env"]

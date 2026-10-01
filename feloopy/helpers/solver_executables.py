# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""Locate external solver binaries under ``~/feloopy/Solvers``.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

__all__ = [
    "KNOWN_EXTERNAL_SOLVERS",
    "solvers_home",
    "paths_file",
    "solver_paths",
    "register_paths",
    "rebuild_paths",
    "find_in_directory",
    "find_solver_executable",
    "related_dirs",
    "inject_solver_paths",
    "ensure_solver_on_path",
    "missing_solver_message",
    "require_python_solver",
    "auto_download_enabled",
]

#: Executables FelooPy knows how to provision (see solver_provisioning).
KNOWN_EXTERNAL_SOLVERS = (
    "bonmin", "couenne", "ipopt", "cbc", "clp", "glpk", "highs", "scip",
)

#: Canonical solver name -> extra executable spellings it is installed as.
_EXE_ALIASES = {
    "glpk": ("glpsol",),
}

#: How deep the self-healing Solvers-folder scan may descend.
_MAX_DEPTH = 4

#: POSIX library search variables fed with ``.../lib`` directories.
_LIB_VARS = ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH")

#: Environment variable telling libblastrampoline which BLAS/LAPACK
#: library to load on first use (bare filename -> PATH lookup).
_LBT_BACKEND_VAR = "LBT_DEFAULT_LIBS"

_PATHS_HEADER = (
    "# FelooPy solver directories (maintained automatically by `flp setup`).\n"
    "# One directory per line. feloopy injects these into PATH inside the\n"
    "# running Python process only; your system PATH is never changed.\n"
    "# Delete this file to rebuild it from the folders in this directory.\n"
)


def solvers_home() -> Path:
    """The Solvers folder (override with ``FELOOPY_SOLVERS_DIR``)."""
    env = os.environ.get("FELOOPY_SOLVERS_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / "feloopy" / "Solvers"


def paths_file() -> Path:
    """The ``paths.txt`` file that records solver directories."""
    return solvers_home() / "paths.txt"


# ---------------------------------------------------------------------------
# paths.txt: read / write
# ---------------------------------------------------------------------------

def _raw_path_lines() -> List[str]:
    """Every non-comment line of ``paths.txt`` (existing or not)."""
    try:
        text = paths_file().read_text(encoding="utf-8")
    except OSError:
        return []
    return [line.strip() for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")]


def solver_paths() -> List[str]:
    """Directories recorded in ``paths.txt`` (stale entries are dropped)."""
    dirs: List[str] = []
    for line in _raw_path_lines():
        if os.path.isdir(line) and line not in dirs:
            dirs.append(line)
    return dirs


def _write_paths(dirs: Sequence[str]) -> bool:
    target = paths_file()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(_PATHS_HEADER + "".join(d + "\n" for d in dirs),
                       encoding="utf-8")
        os.replace(tmp, target)
        return True
    except OSError:
        return False


def register_paths(dirs: Iterable) -> List[str]:
    """Merge ``dirs`` into ``paths.txt`` (deduplicated, atomic rewrite).

    """
    existing = solver_paths()
    changed = False
    for d in dirs:
        if d is None:
            continue
        d = os.path.abspath(str(d))
        if os.path.isdir(d) and d not in existing:
            existing.append(d)
            changed = True
    if changed:
        _write_paths(existing)
    return existing


def rebuild_paths() -> List[str]:
    """Validate ``paths.txt`` and rescan the Solvers folder for binaries.
    """
    recorded = solver_paths()
    wanted: dict = {}
    for canonical in KNOWN_EXTERNAL_SOLVERS:
        for spelling in _candidate_names(canonical):
            wanted.setdefault(spelling.lower(), canonical)

    additions: List[str] = []
    home = solvers_home()
    if home.is_dir():
        try:
            for dirpath, dirnames, filenames in os.walk(home):
                rel = Path(dirpath).relative_to(home)
                if len(rel.parts) >= _MAX_DEPTH:
                    dirnames[:] = []
                dirnames[:] = [d for d in dirnames if d != ".cache"]
                dirnames.sort()
                if Path(dirpath).name.lower() in ("bin", "lib", "scripts"):
                    additions.append(dirpath)
                for filename in sorted(filenames):
                    if filename.lower() in wanted:
                        additions.extend(related_dirs(os.path.join(dirpath, filename)))
        except OSError:
            pass

    merged = list(recorded)
    for d in additions:
        d = os.path.abspath(d)
        if d not in merged:
            merged.append(d)

    # Rewrite when anything changed or when the file still lists stale
    # entries (raw lines differ from the valid, deduplicated list).
    if merged != recorded or sorted(merged) != sorted(_raw_path_lines()):
        if merged != _raw_path_lines():
            _write_paths(merged)
    return merged


# ---------------------------------------------------------------------------
# Executable lookup
# ---------------------------------------------------------------------------

def _candidate_names(name: str) -> List[str]:
    names = [name]
    names.extend(_EXE_ALIASES.get(name, ()))
    if sys.platform == "win32":
        names = [n + ext for n in names for ext in ("", ".exe", ".bat", ".cmd")]
    return names


def _names_lower(name: str) -> set:
    return {n.lower() for n in _candidate_names(name)}


def _scan(root, names_lower: set, max_depth: int = _MAX_DEPTH) -> Optional[str]:
    """First file under ``root`` whose name matches, else ``None``."""
    root = Path(root)
    if not root.is_dir():
        return None
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            rel = Path(dirpath).relative_to(root)
            if len(rel.parts) >= max_depth:
                dirnames[:] = []
            dirnames.sort()
            for filename in sorted(filenames):
                if filename.lower() in names_lower:
                    return os.path.join(dirpath, filename)
    except OSError:
        return None
    return None


def find_in_directory(directory, name: str, max_depth: int = _MAX_DEPTH) -> Optional[str]:
    """Look for the ``name`` executable anywhere inside ``directory``."""
    if not directory or not name:
        return None
    return _scan(Path(directory), _names_lower(name), max_depth=max_depth)


def _lookup_in_dirs(dirs: Sequence[str], names_lower: set) -> Optional[str]:
    """Direct (non-recursive) match inside any of ``dirs`` — the fast path."""
    for directory in dirs:
        try:
            for filename in sorted(os.listdir(directory)):
                if filename.lower() in names_lower:
                    return os.path.join(directory, filename)
        except OSError:
            continue
    return None


def related_dirs(executable) -> List[str]:
    """The executable's own directory plus sibling ``lib``/``bin`` folders."""
    exe = Path(executable)
    dirs = [exe.parent]
    sibling = exe.parent.parent
    for sub in ("lib", "bin"):
        candidate = sibling / sub
        if candidate.is_dir():
            dirs.append(candidate)
    seen, out = set(), []
    for d in dirs:
        key = os.path.abspath(str(d))
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _env_override(name: str) -> Optional[str]:
    env = os.environ.get("FELOOPY_{}_PATH".format(name.upper()))
    if not env:
        return None
    path = Path(env).expanduser()
    if path.is_dir():
        return _scan(path, _names_lower(name), max_depth=2)
    if path.is_file():
        return str(path)
    return None


def find_solver_executable(name: str) -> Optional[str]:
    """Return the path to the ``name`` executable, or ``None``.

    See the module docstring for the search order.  A binary found by the
    self-healing scan is registered in ``paths.txt`` for next time.
    """
    if not name:
        return None
    name = str(name)
    names_lower = _names_lower(name)

    # 1. explicit per-solver override
    hit = _env_override(name)
    if hit:
        return hit

    # 2. recorded directories (fast path)
    hit = _lookup_in_dirs(solver_paths(), names_lower)
    if hit:
        return hit

    # 3. self-healing scan of the Solvers folder
    hit = _scan(solvers_home(), names_lower)
    if hit:
        register_paths(related_dirs(hit))
        return hit

    # 4. system PATH
    for spelling in _candidate_names(name):
        which = shutil.which(spelling)
        if which:
            return which

    # 5. the active Python environment (venv not on PATH)
    for directory in (Path(sys.prefix) / "bin",
                      Path(sys.prefix) / "Scripts",
                      Path(sys.prefix) / "Library" / "bin"):
        hit = _scan(directory, names_lower, max_depth=1)
        if hit:
            return hit
    return None


# ---------------------------------------------------------------------------
# Process-local PATH injection
# ---------------------------------------------------------------------------

def _inject_var(var: str, dirs: Sequence[str]) -> None:
    """Prepend ``dirs`` to ``var`` in ``os.environ`` (idempotent, no growth).

    Only the current Python process sees this; the user's persistent
    environment is never touched.
    """
    if not dirs:
        return
    current = os.environ.get(var, "")
    parts = [p for p in current.split(os.pathsep) if p]
    wanted = [d for d in dirs if d not in parts]
    if not wanted:
        return
    os.environ[var] = os.pathsep.join(wanted + parts)


def _rehash_pyomo() -> None:
    """Drop Pyomo's cached executable lookups, if Pyomo is already loaded."""
    if "pyomo.common" not in sys.modules:
        return
    try:
        from pyomo.common import Executable
        Executable.rehash()
    except Exception:
        pass


def _inject_lbt_backend(dirs: Sequence[str]) -> None:
    """Set ``LBT_DEFAULT_LIBS`` to an OpenBLAS found in ``dirs``.

    """
    if os.environ.get(_LBT_BACKEND_VAR):
        return
    for directory in dirs:
        try:
            for filename in sorted(os.listdir(directory)):
                if filename.startswith("libopenblas"):
                    os.environ[_LBT_BACKEND_VAR] = filename
                    return
        except OSError:
            continue


def _inject(dirs: Sequence[str]) -> List[str]:
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        return []
    _inject_var("PATH", dirs)
    if sys.platform != "win32":
        lib_dirs = [d for d in dirs if os.path.basename(d) == "lib"]
        for var in _LIB_VARS:
            _inject_var(var, lib_dirs)
    _inject_lbt_backend(dirs)
    _rehash_pyomo()
    return list(dirs)


def inject_solver_paths(names: Optional[Sequence[str]] = None) -> List[str]:
    """Make solver directories visible to this Python process.
    Returns the list of directories that were injected.
    """
    if names:
        dirs: List[str] = []
        for name in names:
            exe = find_solver_executable(name)
            if exe:
                for d in related_dirs(exe):
                    if d not in dirs:
                        dirs.append(d)
        return _inject(dirs)
    return _inject(rebuild_paths())


def ensure_solver_on_path(executable: str) -> None:
    """Inject the directories of an already-found executable (and refresh Pyomo)."""
    _inject(related_dirs(executable))


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

def missing_solver_message(name: str) -> str:
    """Actionable error text for a solver FelooPy could not find.
    """
    from .solver_guides import resolve_commercial

    commercial = resolve_commercial(name)
    if commercial:
        return (
            f"'{name}' is a commercial solver — feloopy cannot download it "
            f"for you.\n"
            f"Install guide (official steps + licensing):  flp setup {commercial}\n"
            f"Once its Python package or binary is installed, feloopy "
            f"picks it up automatically."
        )

    from .solver_provisioning import SETUP_PIP_SOLVERS, resolve_setup_name

    pip_spec = SETUP_PIP_SOLVERS.get(resolve_setup_name(name))
    if pip_spec:
        return (
            f"The '{name}' solver binding (Python package) is not installed.\n"
            f"Install it with: flp setup {name}   (runs: pip install {pip_spec.pip})\n"
            f"or run that pip command yourself, then retry."
        )

    home = solvers_home()
    return (
        f"The '{name}' executable was not found.\n"
        f"Install it with: flp setup {name}\n"
        f"(binaries are downloaded into {home} and recorded in its paths.txt),\n"
        f"or put your own '{name}' binary into that folder (any subfolder),\n"
        f"or point FELOOPY_{name.upper()}_PATH at it."
    )


def require_python_solver(setup_name: str, module: str) -> None:
    """Raise the actionable 'flp setup' message when ``module`` is missing.

    Used by interfaces (rsome, ...) whose solvers are backed by Python
    packages, so a missing binding surfaces as a feloopy hint instead
    of a bare ``ModuleNotFoundError`` from deep inside the interface.
    ``setup_name`` is what ``flp setup`` accepts (pip binding or
    commercial guide name).
    """
    import importlib.util

    try:
        found = importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        found = False
    if not found:
        raise RuntimeError(missing_solver_message(setup_name))


def auto_download_enabled() -> bool:
    """False when first-use binary downloads are opted out."""
    value = os.environ.get("FELOOPY_NO_AUTO_DOWNLOAD", "")
    return value.strip().lower() not in ("1", "true", "yes", "on")

# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""Install open-source solvers
"""

from __future__ import annotations

import hashlib
import json
import os
import platform as _platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .._version import __version__
from .solver_executables import (
    find_in_directory,
    find_solver_executable,
    auto_download_enabled,
    rebuild_paths,
    register_paths,
    related_dirs,
    solvers_home,
)

__all__ = [
    "SetupError",
    "SolverSpec",
    "SETUP_SOLVERS",
    "SETUP_PIP_SOLVERS",
    "SETUP_ALIASES",
    "resolve_setup_name",
    "is_provisionable",
    "provision_solver",
    "ensure_solver",
    "setup_status",
]

_RAW = "https://raw.githubusercontent.com/JuliaBinaryWrappers/{jll}.jl/{ref}/{file}"
_MANIFEST = ".feloopy.json"
_USER_AGENT = "feloopy/{}".format(__version__)

_PINNED_REFS = {"ASL_jll": "ASL-v0.1.3+0"}


def _raw_url(jll: str, file: str) -> str:
    """raw.githubusercontent URL of ``file`` for ``jll`` (pinned tag or main)."""
    ref = urllib.parse.quote(_PINNED_REFS.get(jll, "main"), safe="")
    return _RAW.format(jll=jll, ref=ref, file=file)

#: Exit codes that mean "the loader failed", not "the solver exited".
_LOADER_FAIL_POSIX = (126, 127)

#: Socket silence timeout / absolute per-file download deadline (seconds).
_SOCKET_TIMEOUT = 30.0
_DOWNLOAD_DEADLINE = 1800.0


_STALL_WINDOW = 60.0
_STALL_MIN_BYTES = 16 * 1024


class SetupError(RuntimeError):
    """A solver could not be provisioned (message is user-actionable)."""


@dataclass(frozen=True)
class SolverSpec:
    """One solver ``flp setup`` knows how to install.

    ``kind == "binary"`` entries are downloaded JLL artifacts unpacked
    into ``<Solvers>/<name>/``; ``kind == "pip"`` entries are Python
    bindings installed with pip (``pip`` is the requirement string,
    ``module`` the package checked for availability).
    """

    name: str
    jll: str     # JuliaBinaryWrappers package providing the executable
    exe: str     # executable name expected after installation
    note: str
    kind: str = "binary"   # "binary" (JLL download) or "pip" (pip package)
    pip: str = ""          # requirement for pip installs, e.g. "ecos"
    module: str = ""       # import name checked for pip installs


#: The solvers installable with ``flp setup <name>``.
SETUP_SOLVERS: Dict[str, SolverSpec] = {
    "bonmin": SolverSpec("bonmin", "Bonmin_jll", "bonmin", "MINLP (COIN-OR)"),
    "couenne": SolverSpec("couenne", "Couenne_jll", "couenne", "global MINLP (COIN-OR)"),
    "ipopt": SolverSpec("ipopt", "Ipopt_jll", "ipopt", "NLP (COIN-OR)"),
    "cbc": SolverSpec("cbc", "Cbc_jll", "cbc", "LP/MILP (COIN-OR)"),
    "clp": SolverSpec("clp", "Clp_jll", "clp", "LP (COIN-OR)"),
    "glpk": SolverSpec("glpk", "GLPK_jll", "glpsol", "LP/MILP (GLPK)"),
    "highs": SolverSpec("highs", "HiGHS_jll", "highs", "LP/MILP/QP (HiGHS)"),
    "scip": SolverSpec("scip", "SCIP_jll", "scip", "MIP/NLP (SCIP)"),
}

#: Free solver *bindings* installed with pip — the packages pyomo, pulp,
#: picos, rsome, ortools and feloopy's own interfaces need to reach
#: their solvers.  Pinned requirements mirror the pins in
#: ``clitools/deps.py`` (``==`` for feloopy's core-tested solver bindings
#: like ``highspy==1.15.1``/``pyscipopt==6.2.1``/``ortools==9.15.6755``,
#: ``cylp==0.92.2`` because newer releases dropped Windows wheels;
#: ``>=`` for general packages users may run newer, like ``cvxpy``), so
#: ``flp setup`` never drifts from what feloopy itself offers — keep them
#: in sync (``clitools/deps.py`` in turn mirrors pyproject's ``stock``
#: extra).  An already-installed binding is left alone unless ``--force``
#: is given.
SETUP_PIP_SOLVERS: Dict[str, SolverSpec] = {
    "cvxopt": SolverSpec("cvxopt", "", "", "LP/SOCP via CVXOPT (picos, cvxpy)",
                         "pip", "cvxopt", "cvxopt"),
    "ecos": SolverSpec("ecos", "", "", "SOCP/QP solver (picos, rsome)",
                       "pip", "ecos", "ecos"),
    "osqp": SolverSpec("osqp", "", "", "QP solver (picos)",
                       "pip", "osqp", "osqp"),
    "smcp": SolverSpec("smcp", "", "", "SOCP solver (picos)",
                       "pip", "smcp", "smcp"),
    "pyscipopt": SolverSpec("pyscipopt", "", "", "SCIP binding (picos, feloopy scip)",
                            "pip", "pyscipopt==6.2.1", "pyscipopt"),
    "highspy": SolverSpec("highspy", "", "", "HiGHS binding (pyomo appsi, feloopy highs)",
                          "pip", "highspy==1.15.1", "highspy"),
    "cylp": SolverSpec("cylp", "", "", "CLP binding (rsome cylp, feloopy cylp)",
                       "pip", "cylp==0.92.2", "cylp"),
    "cvxpy": SolverSpec("cvxpy", "", "", "conic interface (rsome cvxpy, feloopy cvxpy)",
                        "pip", "cvxpy>=1.6.4", "cvxpy"),
    "ortools": SolverSpec("ortools", "", "", "bundled solvers (rsome ortools, feloopy ortools)",
                          "pip", "ortools==9.15.6755", "ortools"),
}

#: Spellings interfaces use that map onto a provisioned solver name
#: (``coin`` is PuLP's COIN_CMD which needs our cbc binary, ``glpsol``
#: is the GLPK executable name users often type).
SETUP_ALIASES: Dict[str, str] = {
    "coin": "cbc",
    "coin_cmd": "cbc",
    "glpsol": "glpk",
}


def resolve_setup_name(name: Optional[str]) -> str:
    """Map an interface spelling onto a provisioned setup name."""
    if not name:
        return ""
    return SETUP_ALIASES.get(str(name).strip().lower(), str(name).strip().lower())


def is_provisionable(name: Optional[str]) -> bool:
    """True for free solvers feloopy can install itself (binary or pip)."""
    name = resolve_setup_name(name)
    return name in SETUP_SOLVERS or name in SETUP_PIP_SOLVERS

#: Every install gets the GCC runtime: ``GLPK_jll`` omits it although
#: ``libglpk`` needs ``libgcc_s_seh-1.dll``, and the Fortran-built
#: solvers need libgfortran/libquadmath on machines without one.
_RUNTIME_PKGS = ("CompilerSupportLibraries_jll",)

#: Solvers whose binaries route BLAS/LAPACK through
#: ``libblastrampoline`` (verified by their import tables).  LBT has no
#: backend of its own, so these installs also carry an OpenBLAS.
_BLAS_SOLVERS = frozenset(
    {"bonmin", "couenne", "ipopt", "cbc", "clp", "highs", "scip"}
)
_BLAS_PKGS = ("OpenBLAS32_jll",)

#: Bumped when the provisioning recipe changes (extra packages, pins);
#: older manifests then refresh on the next ``flp setup``.
_SETTINGS_REV = 2

#: In-process caches (parsed Artifacts.toml / Project.toml per package).
_ARTIFACTS_CACHE: Dict[str, list] = {}
_DEPS_CACHE: Dict[str, list] = {}

#: Auto-download attempts that already failed in this process (no retry storm).
_AUTO_FAILED: set = set()


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _http_get(url: str, timeout: float = 60.0, attempts: int = 3) -> bytes:
    last: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 404, 410):
                raise SetupError(
                    "server refused {} (HTTP {}). The package may have been "
                    "renamed or the download is being rate-limited; if you "
                    "use GitHub often, set GITHUB_TOKEN.".format(url, exc.code)
                ) from exc
            last = exc
        except Exception as exc:  # URLError, timeout, ...
            last = exc
        if attempt + 1 < attempts:
            time.sleep(1.5 * (attempt + 1))
    raise SetupError("could not download {}: {}".format(url, last))


# ---------------------------------------------------------------------------
# JLL resolution (Artifacts.toml / Project.toml, no GitHub API)
# ---------------------------------------------------------------------------

def _platform_keys() -> Dict[str, str]:
    """Julia-style platform keys for this machine (for Artifacts.toml)."""
    system = _platform.system()
    machine = _platform.machine().lower()
    if system == "Windows":
        arch = "i686" if machine in ("x86", "i386", "i486", "i586", "i686") else "x86_64"
        return {"os": "windows", "arch": arch}
    if system == "Darwin":
        arch = "aarch64" if machine in ("arm64", "aarch64") else "x86_64"
        return {"os": "macos", "arch": arch}
    if machine in ("aarch64", "arm64"):
        arch = "aarch64"
    elif machine in ("armv6l", "armv7l", "riscv64"):
        arch = machine
    else:
        arch = "x86_64"
    return {"os": "linux", "arch": arch, "libc": _linux_libc()}


def _linux_libc() -> str:
    try:
        value = os.confstr("CS_GNU_LIBC_VERSION") or ""
        if value.lower().startswith("glibc"):
            return "glibc"
    except (AttributeError, OSError, ValueError):
        pass
    return "musl"


def _parse_artifacts(text: str) -> list:
    """Parse the ``[[Name]]`` / ``[[Name.download]]`` blocks of Artifacts.toml."""
    entries: list = []
    current: Optional[dict] = None
    in_download = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[[") and stripped.endswith("]]"):
            target = stripped[2:-2].strip()
            if target.endswith(".download"):
                in_download = True
                continue
            current = {"name": target, "keys": {}, "sha256": None, "url": None}
            entries.append(current)
            in_download = False
            continue
        if current is None:
            continue
        match = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"(.*)"\s*$', stripped)
        if not match:
            continue
        key, value = match.group(1), match.group(2)
        if in_download:
            if key in ("sha256", "url"):
                current[key] = value
        else:
            current["keys"][key] = value
    return entries


def _artifact_entries(jll: str) -> list:
    """All platform artifacts of ``jll`` (its current release, on `main`)."""
    if jll not in _ARTIFACTS_CACHE:
        try:
            raw = _http_get(_raw_url(jll, "Artifacts.toml"))
        except SetupError as exc:
            raise SetupError(
                "could not read release information for {}: {}".format(jll, exc)
            ) from exc
        _ARTIFACTS_CACHE[jll] = _parse_artifacts(raw.decode("utf-8"))
    return _ARTIFACTS_CACHE[jll]


def _pick_artifact(entries: Sequence[dict], want: Dict[str, str]) -> Optional[dict]:
    """Best artifact for the platform keys in ``want`` (never musl/glibc mixups)."""
    best: Optional[dict] = None
    best_score = 0
    for entry in entries:
        keys = entry.get("keys") or {}
        if keys.get("os") != want.get("os") or keys.get("arch") != want.get("arch"):
            continue
        if not entry.get("url"):
            continue
        score = 1
        libc = keys.get("libc")
        if libc:
            score += 4 if libc == want.get("libc") else -4
        if keys.get("cxxstring_abi") == "cxx11":
            score += 2
        gfortran = keys.get("libgfortran_version")
        if gfortran and gfortran.startswith("5"):
            score += 1
        if best is None or score > best_score:
            best, best_score = entry, score
    return best


def _tag_from_url(url: str) -> Optional[str]:
    match = re.search(r"/releases/download/([^/]+)/", url)
    return match.group(1) if match else None


def jll_dependencies(jll: str) -> List[str]:
    """Direct ``*_jll`` dependencies of a package (from its Project.toml)."""
    if jll in _DEPS_CACHE:
        return _DEPS_CACHE[jll]
    raw = _http_get(_raw_url(jll, "Project.toml"))
    match = re.search(r"\[deps\](.*?)(?:\n\[|\Z)", raw.decode("utf-8"), re.S)
    deps = []
    if match:
        deps = [name for name in re.findall(r"^([A-Za-z0-9_]+)\s*=", match.group(1), re.M)
                if name.endswith("_jll")]
    _DEPS_CACHE[jll] = deps
    return deps


def jll_closure(roots: Sequence[str]) -> List[str]:
    """Full transitive ``*_jll`` closure of ``roots`` (roots first).

    A dependency whose Project.toml cannot be read is skipped with a
    warning rather than failing the whole install; the root package is
    required.
    """
    order: List[str] = []
    seen: set = set()
    queue: List[str] = list(roots)
    while queue:
        pkg = queue.pop(0)
        if pkg in seen:
            continue
        seen.add(pkg)
        order.append(pkg)
        try:
            queue.extend(jll_dependencies(pkg))
        except SetupError:
            if pkg in roots:
                raise
            print("feloopy: warning: could not read dependencies of {}; "
                  "continuing without them".format(pkg))
    return order


# ---------------------------------------------------------------------------
# Download / extract / verify
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_dir() -> Path:
    """Verified tarballs are kept here so re-runs and shared dependency
    closures (``flp setup bonmin couenne``) download each file once."""
    return solvers_home() / ".cache"


def _download_to(url: str, target: Path, expected: str) -> None:
    """Stream ``url`` into ``target``, then verify it.

    Resumes a previous partial download when the server supports HTTP
    Range, keeps the partial file across attempts so retries are cheap,
    and aborts when the connection stalls (too little progress per
    window) instead of hanging forever.
    """
    tmp = target.with_name(target.name + ".part")
    start = tmp.stat().st_size if tmp.is_file() else 0
    headers = {"User-Agent": _USER_AGENT}
    if start:
        headers["Range"] = "bytes={}-".format(start)
    request = urllib.request.Request(url, headers=headers)
    started = time.monotonic()
    window_start = started
    window_bytes = 0
    try:
        with urllib.request.urlopen(request, timeout=_SOCKET_TIMEOUT) as response:
            resume = bool(start) and getattr(response, "status", None) == 206
            if start and not resume:
                start = 0
            with tmp.open("ab" if resume else "wb") as handle:
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    window_bytes += len(chunk)
                    now = time.monotonic()
                    if now - window_start >= _STALL_WINDOW:
                        if window_bytes < _STALL_MIN_BYTES:
                            raise SetupError(
                                "connection stalled ({:.0f} KB in the last "
                                "{:.0f}s)".format(window_bytes / 1024.0,
                                                   now - window_start))
                        window_start, window_bytes = now, 0
                    if now - started > _DOWNLOAD_DEADLINE:
                        raise SetupError(
                            "gave up after {}s".format(int(_DOWNLOAD_DEADLINE)))
    except SetupError:
        raise  # partial file stays for the next attempt to resume
    except Exception as exc:
        raise SetupError(str(exc)) from exc
    actual = _sha256(tmp) if tmp.is_file() else ""
    if expected and actual != expected:
        tmp.unlink(missing_ok=True)  # corrupt prefix: start over
        raise SetupError(
            "checksum mismatch for {} (expected {}, got {}); "
            "the download was corrupted or tampered with"
            .format(target.name, expected[:16] + "...", actual[:16] + "...")
        )
    os.replace(tmp, target)


def _fetch_artifact(entry: dict) -> Path:
    """Return the verified artifact for ``entry``, downloading it if needed."""
    url = entry["url"]
    target = _cache_dir() / url.rsplit("/", 1)[-1]
    expected = (entry.get("sha256") or "").lower()
    if target.is_file():
        if not expected or _sha256(target) == expected:
            return target
        target.unlink()  # stale/corrupt cache entry: refetch
    _cache_dir().mkdir(parents=True, exist_ok=True)
    last: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            _download_to(url, target, expected)
            return target
        except SetupError as exc:
            last = exc
            if attempt < 3:
                time.sleep(2.0 * attempt)
    raise SetupError("could not download {}: {}".format(url, last))


def _extract_tar(archive: Path, dest: Path) -> None:
    with tarfile.open(str(archive), "r:gz") as tar:
        try:
            tar.extractall(str(dest), filter="data")  # Python >= 3.10.12/3.11.4
        except TypeError:
            root = os.path.realpath(str(dest))
            for member in tar.getmembers():
                path = os.path.realpath(os.path.join(root, member.name))
                if not path.startswith(root + os.sep):
                    raise SetupError(
                        "refusing to extract unsafe path {} from {}"
                        .format(member.name, archive.name)
                    )
            tar.extractall(str(dest))


def _smoke_test(executable: str) -> None:
    """Run the binary once; a loader failure (missing DLL etc.) raises."""
    try:
        proc = subprocess.run(
            [executable],
            cwd=os.path.dirname(executable) or None,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except OSError as exc:
        raise SetupError("{} could not be executed: {}".format(executable, exc)) from exc
    except subprocess.TimeoutExpired as exc:
        raise SetupError("{} did not respond within 30s".format(executable)) from exc
    returncode = proc.returncode
    failed = (returncode < 0) if sys.platform == "win32" else \
             (returncode in _LOADER_FAIL_POSIX)
    if failed:
        detail = (proc.stderr or proc.stdout or b"").decode("utf-8", "replace").strip()
        raise SetupError(
            "{} failed to load (exit code {}){}"
            .format(executable, returncode, ": " + detail[:400] if detail else "")
        )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _read_manifest(dest: Path) -> Optional[dict]:
    try:
        return json.loads((dest / _MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_manifest(dest: Path, data: dict) -> None:
    try:
        (dest / _MANIFEST).write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass


def _has_backend(dest: Path) -> bool:
    """True when an OpenBLAS backend library sits next to the binaries."""
    for directory in (dest / "bin", dest / "lib", dest):
        if not directory.is_dir():
            continue
        try:
            for entry in directory.iterdir():
                if entry.name.startswith("libopenblas"):
                    return True
        except OSError:
            continue
    return False


# ---------------------------------------------------------------------------
# Provisioning (pip bindings)
# ---------------------------------------------------------------------------

def _pip_probe(spec: SolverSpec) -> tuple:
    """``(installed, version, location)`` for a pip-installed binding."""
    origin = None
    try:
        import importlib.util
        module = importlib.util.find_spec(spec.module)
        origin = getattr(module, "origin", None) or None
    except (ImportError, ValueError, AttributeError):
        origin = None
    installed = bool(origin) and origin != "frozen"
    version = ""
    location = ""
    if installed:
        try:
            from importlib import metadata
            version = metadata.version(spec.name)
        except Exception:
            pass
        location = os.path.dirname(origin) if origin else ""
    return installed, version, location


def _provision_pip(spec: SolverSpec, *, force: bool = False,
                   quiet: bool = False) -> str:
    """Install ``spec``'s binding with pip; returns the package location."""
    def log(*args):
        if not quiet:
            print(*args, flush=True)

    installed, version, location = _pip_probe(spec)
    if installed and not force:
        log("{}: already installed (pip {}) — {}".format(
            spec.name, version or "?", location))
        return location
    log("{}: installing with pip: {}".format(spec.name, spec.pip))
    command = [sys.executable, "-m", "pip", "install",
               "--disable-pip-version-check", spec.pip]
    try:
        proc = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=1800,
        )
    except OSError as exc:
        raise SetupError(
            "could not run pip for {}: {}".format(spec.name, exc)) from exc
    except subprocess.TimeoutExpired as exc:
        raise SetupError(
            "pip install {} timed out after 1800s".format(spec.pip)) from exc
    if proc.returncode != 0:
        tail = (proc.stdout or b"").decode("utf-8", "replace").strip()
        tail = "\n".join(tail.splitlines()[-12:])
        raise SetupError(
            "pip install {} failed{}:\n{}".format(
                spec.pip, "" if not tail else "", tail))
    installed, version, location = _pip_probe(spec)
    if not installed:
        raise SetupError(
            "pip reported success for {} but '{}' cannot be imported; "
            "try it manually: pip install {}".format(
                spec.pip, spec.module, spec.pip))
    log("{}: installed {} -> {}".format(spec.name, version, location))
    return location


# ---------------------------------------------------------------------------
# Provisioning (binaries)
# ---------------------------------------------------------------------------

def provision_solver(
    name: str,
    *,
    force: bool = False,
    quiet: bool = False,
    verify: bool = True,
) -> str:
    """Install (or refresh) ``name`` into ``<Solvers>/<name>/``.

    Returns the path of the installed executable (or the package
    location for pip bindings).  Raises :class:`SetupError` with an
    actionable message on failure.  Nothing is unpacked until every
    archive downloaded and verified, so a failed run never leaves a
    half-installed solver behind.
    """
    name = resolve_setup_name(name)
    spec = SETUP_SOLVERS.get(name) or SETUP_PIP_SOLVERS.get(name)
    if spec is None:
        from .solver_guides import commercial_names
        raise SetupError(
            "unknown solver '{}'; free solvers: {}; commercial (install "
            "guide): {}".format(
                name,
                ", ".join(list(SETUP_SOLVERS) + list(SETUP_PIP_SOLVERS)),
                ", ".join(commercial_names()),
            )
        )
    if spec.kind == "pip":
        return _provision_pip(spec, force=force, quiet=quiet)

    def log(*args):
        if not quiet:
            print(*args, flush=True)

    dest = solvers_home() / name
    want = _platform_keys()
    platform_label = "{}-{}".format(want.get("arch"), want.get("os"))

    extras: List[str] = list(_RUNTIME_PKGS)
    if name in _BLAS_SOLVERS:
        extras += list(_BLAS_PKGS)

    # -- is an up-to-date install already present? ------------------------
    existing_exe = find_in_directory(dest, spec.exe)
    try:
        root_entry = _pick_artifact(_artifact_entries(spec.jll), want)
    except SetupError:
        # Offline (or rate-limited): a working local install still counts.
        if existing_exe and not force:
            log("{}: installed, but could not check for updates ({})"
                .format(name, existing_exe))
            register_paths(related_dirs(existing_exe))
            return existing_exe
        raise
    if root_entry is None:
        raise SetupError(
            "{} has no build for this platform ({})".format(spec.jll, platform_label)
        )
    root_tag = _tag_from_url(root_entry["url"]) or "unknown"
    manifest = _read_manifest(dest)
    if existing_exe and not force:
        fresh = (
            bool(manifest)
            and manifest.get("root_tag") == root_tag
            and manifest.get("rev") == _SETTINGS_REV
            and not (name in _BLAS_SOLVERS and not _has_backend(dest))
        )
        if fresh:
            log("{}: up to date ({}) — {}".format(name, root_tag, existing_exe))
            register_paths(related_dirs(existing_exe))
            return existing_exe
        if manifest is None:
            log("{}: replacing existing files in {}".format(name, dest))
        else:
            log("{}: updating {} -> {}".format(
                name, manifest.get("root_tag"), root_tag))

    # -- resolve the dependency closure ------------------------------------
    log("{}: resolving dependencies for {}...".format(name, platform_label))
    packages = jll_closure([spec.jll] + extras)
    picks: List[tuple] = []
    skipped: List[str] = []
    for pkg in packages:
        entry = _pick_artifact(_artifact_entries(pkg), want)
        if entry is None:
            if pkg == spec.jll:
                raise SetupError(
                    "{} has no build for this platform ({})"
                    .format(spec.jll, platform_label)
                )
            skipped.append(pkg)      # e.g. Xorg_libpciaccess on Windows
            continue
        picks.append((pkg, entry))
    log("{}: {} packages to download{}".format(
        name, len(picks),
        " ({} not built for this platform)".format(len(skipped)) if skipped else ""))

    # -- download (all first, so a failure never leaves a half install) ----
    archives: List[Path] = []
    for pkg, entry in picks:
        target = _cache_dir() / entry["url"].rsplit("/", 1)[-1]
        cached = target.is_file()
        archive = _fetch_artifact(entry)
        size_mb = archive.stat().st_size / (1024.0 * 1024.0)
        log("  {:<28s} {:>7.1f} MB{}".format(
            pkg, size_mb, "  (cached)" if cached else ""))
        archives.append(archive)

    # -- replace the install directory ---------------------------------
    if dest.exists():
        try:
            shutil.rmtree(dest)
        except OSError as exc:
            raise SetupError(
                "cannot replace {}: {} (is a solver still running?)"
                .format(dest, exc)
            ) from exc
    dest.mkdir(parents=True, exist_ok=True)
    log("{}: installing into {}".format(name, dest))
    for archive in archives:
        _extract_tar(archive, dest)

    if sys.platform != "win32":
        for bin_dir in (dest / "bin", dest):
            if not bin_dir.is_dir():
                continue
            for child in bin_dir.iterdir():
                if child.is_file() and not child.suffix:
                    try:
                        child.chmod(child.stat().st_mode | 0o755)
                    except OSError:
                        pass

    # -- locate, smoke-test, record ------------------------------------
    executable = find_in_directory(dest, spec.exe)
    if executable is None:
        listing = sorted(p.name for p in (dest / "bin").iterdir()) \
            if (dest / "bin").is_dir() else []
        raise SetupError(
            "installed {} but the '{}' executable was not found "
            "(bin/ contains: {}). Please report this at "
            "https://github.com/feloopy/feloopy/issues"
            .format(root_tag, spec.exe, ", ".join(listing) or "nothing")
        )
    if verify:
        log("{}: verifying {}...".format(name, executable))
        _smoke_test(executable)

    register_paths(related_dirs(executable))
    rebuild_paths()
    _write_manifest(dest, {
        "name": name,
        "jll": spec.jll,
        "root_tag": root_tag,
        "platform": want,
        "packages": sorted(pkg for pkg, _entry in picks),
        "skipped": skipped,
        "extras": extras,
        "rev": _SETTINGS_REV,
        "exe": os.path.relpath(executable, str(dest)),
        "installed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feloopy": __version__,
    })
    log("{}: installed {} -> {}".format(name, root_tag, executable))
    return executable


def ensure_solver(name: str, quiet: bool = False) -> Optional[str]:
    """Find ``name``'s executable, provisioning it on first explicit use.

    Returns the executable path, or ``None`` when it is unavailable.
    Downloads happen only for :data:`SETUP_SOLVERS` names (free JLL
    binaries — pip bindings and commercial solvers are never installed
    implicitly), only when the binary is missing, and only while
    ``FELOOPY_NO_AUTO_DOWNLOAD`` is not set; a failed attempt is not
    retried within the same process.
    """
    name = resolve_setup_name(name)
    executable = find_solver_executable(name)
    if executable:
        return executable
    if name not in SETUP_SOLVERS or not auto_download_enabled():
        return None
    if name in _AUTO_FAILED:
        return None

    def log(*args):
        if not quiet:
            print(*args, flush=True)

    log("feloopy: '{}' is not installed yet; downloading it into {} "
        "(set FELOOPY_NO_AUTO_DOWNLOAD=1 to disable this)"
        .format(name, solvers_home()))
    try:
        executable = provision_solver(name, quiet=quiet)
    except Exception as exc:
        _AUTO_FAILED.add(name)
        print("feloopy: could not auto-install {}: {}".format(name, exc))
        return None
    return executable


def setup_status() -> List[dict]:
    """Per-solver install status for ``flp setup --list`` and friends.

    Rows cover both kinds: JLL binaries (``kind == "binary"``) and pip
    bindings (``kind == "pip"``); ``installed``/``version``/``exe`` are
    filled in from the manifest resp. the installed package metadata.
    """
    rebuild_paths()
    rows: List[dict] = []
    for name, spec in SETUP_SOLVERS.items():
        executable = find_solver_executable(name)
        manifest = _read_manifest(solvers_home() / name)
        rows.append({
            "name": name,
            "kind": "binary",
            "installed": bool(executable),
            "version": (manifest or {}).get("root_tag"),
            "exe": executable,
            "note": spec.note,
        })
    for name, spec in SETUP_PIP_SOLVERS.items():
        installed, version, location = _pip_probe(spec)
        rows.append({
            "name": name,
            "kind": "pip",
            "installed": installed,
            "version": version,
            "exe": location,
            "note": spec.note,
        })
    return rows

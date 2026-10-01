# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""Julia server for feloopy's JuMP (``interface="jump"``) support.

Provides a single-process Julia runtime that is initialized once and reused
across all solution, model, and result generators.
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

_JUMP_BASE_PACKAGES = ["JuMP", "HiGHS", "Cbc", "GLPK", "Ipopt", "SCIP"]

_RE_VERSION = re.compile(r"^([0-9]+)(?:\.([0-9]+)(?:\.([0-9]+))?)?$")
_JL = None
_STOPPED = False


def feloopy_dirs():
    root = Path.home() / "feloopy"
    return root / "Interpreters" / "Julia", root / "Environments" / "Julia"


def _parse_version(text):
    m = _RE_VERSION.match(str(text).strip())
    if m is None:
        return None
    return tuple(int(g) if g is not None else 0 for g in m.groups())


def _read_juliacall_juliapkg_json():
    spec = importlib.util.find_spec("juliacall")
    if spec is None or spec.origin is None:
        return None
    path = Path(spec.origin).resolve().parent / "juliapkg.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _juliacall_julia_compat():
    data = _read_juliacall_juliapkg_json()
    if data is None:
        return None
    spec = data.get("julia")
    return str(spec) if spec else None


def _caret_range(spec_str):
    s = spec_str.strip().lstrip("^~")
    m = _RE_VERSION.match(s)
    if m is None:
        return None, None
    major, minor, patch = (int(g) if g is not None else 0 for g in m.groups())
    lo = (major, minor, patch)
    if major != 0:
        hi = (major + 1, 0, 0)
    elif minor != 0:
        hi = (0, minor + 1, 0)
    else:
        hi = (0, 0, patch + 1)
    return lo, hi


def _version_in_spec(ver_tuple, spec_str):
    if not spec_str:
        return True
    lo, hi = _caret_range(spec_str)
    if lo is None:
        return True
    return lo <= ver_tuple < hi


def _find_julia_exe(interpreters_dir):
    exe_name = "julia.exe" if os.name == "nt" else "julia"
    found = []
    try:
        entries = os.listdir(interpreters_dir)
    except OSError:
        return []
    for entry in entries:
        ver = _parse_version(entry)
        if ver is None:
            continue
        exe = Path(interpreters_dir) / entry / "bin" / exe_name
        if exe.is_file():
            found.append((ver, str(exe)))
    found.sort(key=lambda item: item[0], reverse=True)
    return found


def find_feloopy_julia(spec=None):
    interpreters_dir, _ = feloopy_dirs()
    if spec is None:
        spec = _juliacall_julia_compat()
    all_found = _find_julia_exe(interpreters_dir)
    if spec:
        return [(v, e) for v, e in all_found if _version_in_spec(v, spec)]
    return all_found


def find_feloopy_julia_env(version_tuple=None):
    _, envs_dir = feloopy_dirs()
    candidates = []
    if version_tuple is not None:
        ver_str = ".".join(str(p) for p in version_tuple)
        candidates.append(Path(envs_dir) / ver_str / "main")
        candidates.append(Path(envs_dir) / ver_str)
    try:
        for entry in sorted(os.listdir(envs_dir)):
            if _parse_version(entry) is None:
                continue
            candidates.append(Path(envs_dir) / entry / "main")
            candidates.append(Path(envs_dir) / entry)
    except OSError:
        pass
    for cand in candidates:
        if (cand / "Project.toml").is_file():
            return str(cand)
    return None


def _env_depot(env_dir):
    if not env_dir:
        return None
    depot = Path(env_dir) / ".julia-depot"
    return str(depot) if depot.is_dir() else None


def configure_feloopy_julia(verbose=False):
    info = {"configured": False, "reason": "no feloopy Julia found"}

    if "juliacall" in sys.modules:
        info["reason"] = "juliacall already imported"
        return info

    if os.getenv("PYTHON_JULIACALL_EXE") or os.getenv("PYTHON_JULIAPKG_EXE"):
        info["reason"] = "user override present"
        return info

    os.environ['PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION'] = 'no'

    found = find_feloopy_julia()
    if not found:
        return info

    ver, exe = found[0]
    env_dir = find_feloopy_julia_env(ver)
    if env_dir is None:
        os.environ["PYTHON_JULIAPKG_EXE"] = exe
        info.update(configured=True, exe=exe, env=None,
                    reason="interpreter only (no project env)")
    else:
        os.environ["PYTHON_JULIACALL_EXE"] = exe
        os.environ["PYTHON_JULIACALL_PROJECT"] = env_dir
        os.environ["PYTHON_JULIAPKG_EXE"] = exe
        os.environ["PYTHON_JULIAPKG_PROJECT"] = env_dir
        depot = _env_depot(env_dir)
        if depot:
            sep = ";" if os.name == "nt" else ":"
            existing = os.getenv("JULIA_DEPOT_PATH", "")
            parts = [p for p in existing.split(sep) if p] if existing else []
            if depot not in parts:
                os.environ["JULIA_DEPOT_PATH"] = sep.join([depot] + parts)
        info.update(configured=True, exe=exe, env=env_dir,
                    version=".".join(str(p) for p in ver),
                    reason="feloopy Julia dirs")
    if verbose:
        print(f"[feloopy] Julia: {info['reason']}: {exe}"
              + (f" (project {env_dir})" if env_dir else ""))
    return info


def _run_julia(exe, args, depot=None, capture=False):
    env = dict(os.environ)
    if depot:
        env["JULIA_DEPOT_PATH"] = depot
    env.setdefault("JULIA_PKG_OFFLINE", "false")
    return subprocess.run([exe] + args, check=True, env=env,
                          capture_output=capture, text=capture)


def _create_project_env(exe, version_str, packages=None, verbose=False):
    _, envs_dir = feloopy_dirs()
    env_dir = Path(envs_dir) / version_str / "main"
    env_dir.mkdir(parents=True, exist_ok=True)
    depot = str(env_dir / ".julia-depot")
    Path(depot).mkdir(parents=True, exist_ok=True)

    data = _read_juliacall_juliapkg_json()
    if data and "packages" in data:
        pin = data["packages"].get("PythonCall", {})
        py_uuid = pin.get("uuid", "6099a3de-0909-46bc-b1f4-468b9a2dfc0d")
        py_ver = pin.get("version", "")
    else:
        py_uuid, py_ver = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d", ""

    add_list = list(packages or _JUMP_BASE_PACKAGES)
    script = (
        "using Pkg; Pkg.activate(%r); "
        "Pkg.add([Pkg.PackageSpec(uuid=%r, version=%r)]); "
        "Pkg.add(%r); Pkg.instantiate(); Pkg.precompile()"
        % (str(env_dir), py_uuid, py_ver, add_list)
    )
    if verbose:
        print(f"[feloopy] Creating Julia env at {env_dir} ...")
    _run_julia(exe, ["--project=" + str(env_dir), "--startup-file=no",
                     "-e", script], depot=depot)
    return str(env_dir)


def ensure_jump_ready(verbose=True, packages=None):
    if os.getenv("PYTHON_JULIACALL_EXE") or os.getenv("PYTHON_JULIAPKG_EXE"):
        os.environ.setdefault('PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION', 'no')
        return {"configured": True, "reason": "user override"}

    info = configure_feloopy_julia(verbose=verbose)
    if info.get("configured"):
        return info

    try:
        from juliapkg.compat import Compat
        from juliapkg.install_julia import best_julia_version, install_julia
    except Exception as exc:
        raise RuntimeError(
            "No feloopy-managed Julia found and juliapkg is unavailable "
            f"to download one: {exc}"
        )

    spec = _juliacall_julia_compat() or "1"
    ver_str, ver_info = best_julia_version(Compat.parse(spec))
    interpreters_dir, _ = feloopy_dirs()
    prefix = str(Path(interpreters_dir) / ver_str)
    if verbose:
        print(f"[feloopy] No local Julia found; downloading Julia {ver_str} "
              f"to {prefix} ...")
    install_julia(ver_info, prefix)
    env_dir = _create_project_env(
        str(Path(prefix) / "bin" / ("julia.exe" if os.name == "nt" else "julia")),
        ver_str, packages=packages, verbose=verbose)
    info = configure_feloopy_julia(verbose=verbose)
    info["downloaded"] = ver_str
    info["env"] = env_dir
    return info


def _jl_atexit_hook():
    """Call the juliacall C-level ``jl_atexit_hook(0)`` to shut down Julia."""
    global _JL, _STOPPED
    if _JL is None or _STOPPED:
        return
    _STOPPED = True
    try:
        import juliacall as _jc
        if "juliacall" in sys.modules and hasattr(_jc, "CONFIG") and _jc.CONFIG.get("inited"):
            lib = _jc.CONFIG.get("lib")
            if lib is not None:
                hook = lib.jl_atexit_hook
                hook.argtypes = [int]
                hook.restype = None
                hook(0)
    except Exception:
        pass
    _JL = None


def _precompile_packages(jl):
    """Precompile JuMP + solver packages so first solve is fast."""
    try:
        jl.seval(
            'import Pkg; '
            'Pkg.status(); '
            'using JuMP, HiGHS, Cbc, GLPK, Ipopt'
        )
    except Exception:
        pass


def get_jl():
    """Return ``juliacall.Main``, initializing Julia exactly once.
    """
    global _JL, _STOPPED
    if _JL is not None:
        return _JL
    _STOPPED = False

    ensure_jump_ready(verbose=False)

    from juliacall import Main as jl

    _precompile_packages(jl)
    _JL = jl
    return _JL


def start_julia():
    """Start the Julia server (or return the existing one).
    """
    return get_jl()


def stop_julia():
    """Stop the running Julia server and release its resources.
    """
    _jl_atexit_hook()


def restart_julia():
    """Stop the current Julia server and start a fresh one.
    """
    stop_julia()
    return start_julia()


def julia_is_running():
    """Return ``True`` if the Julia server is currently active."""
    global _JL, _STOPPED
    return _JL is not None and not _STOPPED


def jl_eval_silent(jl, code):
    """Evaluate Julia code with stdout/stderr redirected to devnull.
    """
    import os as _os
    devnull = _os.open(_os.devnull, _os.O_WRONLY)
    old_stdout = _os.dup(1)
    old_stderr = _os.dup(2)
    _os.dup2(devnull, 1)
    _os.dup2(devnull, 2)
    try:
        jl.seval(code)
    finally:
        _os.dup2(old_stdout, 1)
        _os.dup2(old_stderr, 2)
        _os.close(old_stdout)
        _os.close(old_stderr)
        _os.close(devnull)


def jl_eval_streaming(jl, code):
    """Evaluate Julia code with output streamed progressively to Python stdout.
    """
    jl.seval(code)

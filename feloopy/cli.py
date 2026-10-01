# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""FelooPy's command-line interface (``feloopy`` / ``flp`` / ``fly``)."""

import argparse
import os
import sys

try:
    import tkinter as tk
    from tkinter import filedialog
except Exception:
    pass

from .clitools import *


# ---------------------------------------------------------------------------
# Thin wrappers: argparse always invokes ``func(args)``.
# ---------------------------------------------------------------------------

def _run_setup(args):
    cli_setup(args)


def _run_clean(args):
    clean_project()


def _run_build(args):
    build_project()


def _run_run(args):
    target = args.file
    if not target and os.path.isfile("main.py"):
        target = "main.py"
    if not target:
        sys.exit("error: 'feloopy run' needs a Python file (e.g. 'feloopy run main.py')")
    run_project(target)


def _run_ext(args):
    if not args.extensions:
        sys.exit("error: 'feloopy ext' needs at least one extension id")
    install_vscode_extensions(args.extensions)


def _run_install(args):
    if not args.packages:
        sys.exit("error: 'feloopy install' needs at least one package name "
                 "(run 'feloopy deps' to see what FelooPy knows how to install)")
    resolution = resolve_install(args.packages, latest=getattr(args, "latest", False))
    for name, dep in resolution.matched:
        print("{} -> {}".format(name, ", ".join(dep.requirements)))
    for name, reason in resolution.skipped:
        print("{}: {}".format(name, reason))
    if not resolution.requirements:
        return
    pip_install(resolution.requirements, update=args.update)
    for _name, dep in resolution.matched:
        for interface in dep.interfaces:
            installed, version = probe_interface(interface)
            if installed:
                print("  [ok] interface '{}'{}".format(
                    interface, " ({})".format(version) if version else ""))


def _run_deps(args):
    cli_deps(args)


def _run_uninstall(args):
    if not args.packages:
        sys.exit("error: 'feloopy uninstall' needs at least one package name")
    pip_uninstall(args.packages)


def _run_jlinstall(args):
    if not args.packages:
        sys.exit("error: 'feloopy jlinstall' needs at least one package name")
    julia_install(args.packages)


def _run_jluninstall(args):
    if not args.packages:
        sys.exit("error: 'feloopy jluninstall' needs at least one package name")
    julia_uninstall(args.packages)


def _run_version(args):
    cli_version()


def _run_welcome(args):
    cli_welcome(args)


def _run_update_check(args):
    from .helpers.update_check import check_update
    result = check_update(force=getattr(args, "force", True), verbose=True, notify=False)
    if result.get("error") and not result.get("checked"):
        sys.stderr.write("Update check did not complete: {}\n".format(result["error"]))


def cli_detect(args=None):
    detect_package_manager(verbose=True)


# ---------------------------------------------------------------------------
# Version flags: accepted in any position (``flp -v``, ``flp solvers -v``, ...)
# ---------------------------------------------------------------------------

#: Every spelling that means "print the version" (``--v``/``--ver`` abbreviate
#: to ``--version`` as well).
VERSION_FLAGS = ("-v", "-version", "--version")


def _is_version_flag(arg) -> bool:
    """True for ``-v``, ``-ve`` ... ``-version``, ``--v`` ... ``--version``."""
    if arg in VERSION_FLAGS:
        return True
    if not isinstance(arg, str) or len(arg) < 2 or not arg.startswith("-"):
        return False
    if arg == "--":                       # escape: everything after is data
        return False
    if arg.startswith("--"):
        return "--version".startswith(arg)
    return "-version".startswith(arg)


def _scan_version_flags(argv):
    """Return ``(wanted, first_positional)`` for ``argv``.

    Scanning stops at ``--`` so a literal ``-v`` passed on to a subcommand
    (e.g. ``flp run -- -v``) is never mistaken for a version flag.
    """
    wanted = False
    first_positional = None
    for arg in argv:
        if arg == "--":
            break
        if _is_version_flag(arg):
            wanted = True
        elif not arg.startswith("-") and first_positional is None:
            first_positional = arg
    return wanted, first_positional


def _command_names():
    for action in build_parser()._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    return set()


_EPILOG = """\
examples:
  feloopy -v                                  print the FelooPy version
  feloopy project --name my-problem --dir .   scaffold a new optimization project
  feloopy solvers                            check which solver interfaces are installed
  feloopy solvers --missing --json           machine-readable list of missing solvers
  feloopy deps                               optional dependencies + preferred versions
  feloopy install gurobi cylp                install them, pinned to FelooPy's versions
  feloopy install --latest gurobi            newest releases instead of the pins
  feloopy algorithms -i mealpy -s pso         search mealpy metaheuristics
  feloopy algorithms -c exact --available     exact solvers you can run right now
  feloopy params gurobi gurobi              options for that interface+solver pair
  feloopy info                               summarize the current environment
  feloopy welcome                            banner + setup wizard
  feloopy run main.py                        run a model script
"""


def build_parser():
    """Build and return the ``feloopy`` argument parser (exposed for tests)."""
    parser = argparse.ArgumentParser(
        prog="feloopy",
        description="FelooPy's command-line tool",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", title="commands", description="Valid commands")

    parser.add_argument("-v", "-version", "--version", action="store_true",
                        help="Print the version of FelooPy (any spelling, any position)")

    # -- environment introspection -----------------------------------------
    solvers_parser = subparsers.add_parser(
        "solvers",
        help="Check which solver interfaces are installed",
        description="Probe every solver interface FelooPy can dispatch to (nothing is imported).",
    )
    solvers_parser.add_argument("-c", "--category", choices=sorted(CATEGORY_LABELS),
                                help="Only show this category of interfaces")
    only = solvers_parser.add_mutually_exclusive_group()
    only.add_argument("--installed", action="store_true", help="Only show installed interfaces")
    only.add_argument("--missing", action="store_true", help="Only show missing interfaces")
    solvers_parser.add_argument("--json", action="store_true", help="Print JSON instead of a table")
    solvers_parser.set_defaults(func=cli_solvers)

    deps_parser = subparsers.add_parser(
        "deps",
        help="List optional dependencies and how to install them",
        description="Every optional package FelooPy can drive, its preferred version "
                    "and whether it is installed.  The stock stack comes from "
                    'pip install "feloopy[stock]"; everything else is a plain '
                    "'flp install <name>'.",
    )
    deps_parser.add_argument("--missing", action="store_true",
                             help="Only show dependencies that are not installed")
    deps_parser.add_argument("--json", action="store_true", help="Print JSON instead of a table")
    deps_parser.set_defaults(func=cli_deps)

    algorithms_parser = subparsers.add_parser(
        "algorithms",
        help="List and search the registered algorithms",
        description="Search the algorithm registries: exact, heuristic, multi-objective, "
                    "robust/DRO, constraint programming, and MCDM.",
    )
    algorithms_parser.add_argument("-c", "--category",
                                   choices=[key for key, _attr, _label, _layout in ALGORITHM_CATEGORIES],
                                   help="Only show this category of algorithms")
    algorithms_parser.add_argument("-i", "--interface",
                                   help="Only show algorithms for this interface/backend (e.g. mealpy, pydecision)")
    algorithms_parser.add_argument("-s", "--search", help="Substring filter on algorithm/interface names")
    algorithms_parser.add_argument("--available", action="store_true",
                                   help="Only show algorithms whose backend is installed")
    algorithms_parser.add_argument("--json", action="store_true", help="Print JSON instead of a table")
    algorithms_parser.set_defaults(func=cli_algorithms)

    params_parser = subparsers.add_parser(
        "params",
        help="Show the options passable to flp.search(..., options={})",
        description="Print the options passable to flp.search(interface=..., "
                    "solver=..., options={...}) for one interface + solver "
                    "combination, as a copy-pasteable Python snippet. Both "
                    "arguments are required: which options are accepted "
                    "depends on the pair.",
    )
    params_parser.add_argument("interface", nargs="?", metavar="INTERFACE",
                               help="Interface name, e.g. highs, gurobi, cvxpy, mealpy "
                                    "(required)")
    params_parser.add_argument("solver", nargs="?", metavar="SOLVER",
                               help="Solver/algorithm behind that interface, "
                                    "e.g. highs, cbc, osqp, pso (required too — "
                                    "an interface alone never decides the options)")
    params_parser.add_argument("--json", action="store_true", help="Print JSON instead of a code snippet")
    params_parser.set_defaults(func=cli_params)

    info_parser = subparsers.add_parser(
        "info",
        help="Summarize the current FelooPy environment",
        description="Show FelooPy/Python versions, platform, and installed interfaces.",
    )
    info_parser.add_argument("--json", action="store_true", help="Print JSON instead of a table")
    info_parser.set_defaults(func=cli_info)

    # -- project lifecycle ---------------------------------------------------
    setup_parser = subparsers.add_parser(
        "setup",
        help="Install free solvers, show install guides for commercial ones",
        description="Install open-source solvers into this environment: "
                    "download open-source binaries (with their shared "
                    "libraries) for this platform into ~/feloopy/Solvers and "
                    "record them in its paths.txt (feloopy injects them into "
                    "PATH inside Python only when needed, so your system PATH "
                    "is never modified; binaries you place into the folder "
                    "manually are detected as well), or install free Python "
                    "solver bindings with pip (highspy, pyscipopt, cvxopt, "
                    "ecos, ...) which is how pyomo/pulp/picos/rsome reach "
                    "many solvers. Commercial solver names (gurobi, cplex, "
                    "xpress, mosek, ...) print a compact install guide with "
                    "the vendor's official steps and licensing options.",
    )
    setup_parser.add_argument(
        "solvers", nargs="*", metavar="SOLVER",
        help="free solvers to install (e.g. bonmin couenne cbc osqp), or a "
             "commercial name for its install guide (e.g. gurobi cplex); "
             "omit to show status",
    )
    setup_parser.add_argument(
        "-l", "--list", action="store_true",
        help="Show the install status and exit (default when no solver is given)",
    )
    setup_parser.add_argument(
        "--force", action="store_true",
        help="Reinstall even if the latest release is already installed",
    )
    setup_parser.set_defaults(func=_run_setup)

    project_parser = subparsers.add_parser("project", help="Create or manage a project")
    project_parser.add_argument("--name", required=True, help="Name of the optimization project")
    project_parser.add_argument("--type", help="Specify project type")
    project_parser.add_argument("--dir", dest="directory", default=None,
                                help="Directory to create the project in (skips the folder picker)")
    project_parser.set_defaults(func=cli_project)

    backup_parser = subparsers.add_parser("backup", help="Create a backup of the project")
    backup_parser.add_argument("name", nargs="?", help="Optional name for the backup file")
    backup_parser.set_defaults(func=zip_project)

    recover_parser = subparsers.add_parser("recover", help="Recover project from a backup")
    recover_parser.add_argument("name", nargs="?", help="Name of the backup file to recover")
    recover_parser.set_defaults(func=recover_project)

    subparsers.add_parser("clean", help="Delete caches, logs, and compiled files").set_defaults(func=_run_clean)
    subparsers.add_parser("build", help="Bundle the project (with a venv) into a zip").set_defaults(func=_run_build)

    run_parser = subparsers.add_parser("run", help="Run a Python file (defaults to main.py)")
    run_parser.add_argument("file", nargs="?", help="Python file to run")
    run_parser.set_defaults(func=_run_run)

    # -- package management --------------------------------------------------
    ext_parser = subparsers.add_parser("ext", help="Install VSCode extensions")
    ext_parser.add_argument("extensions", nargs="*", help="Extensions to install")
    ext_parser.set_defaults(func=_run_ext)

    install_parser = subparsers.add_parser(
        "install",
        help="Install packages (FelooPy's optional dependencies by plain name)",
        description="Install Python packages.  Names FelooPy knows ('gurobi', 'cylp', "
                    "'rsome_ro', ...) are resolved to its preferred, tested versions; "
                    "anything else is passed to pip unchanged.",
    )
    install_parser.add_argument("-u", "--update", action="store_true",
                                help="Update installed packages to the latest versions")
    install_parser.add_argument("--latest", action="store_true",
                                help="Install the newest releases instead of FelooPy's preferred (tested) versions")
    install_parser.add_argument("packages", nargs="*",
                                help="Plain dependency names (see 'feloopy deps') or any pip requirement")
    install_parser.set_defaults(func=_run_install)

    uninstall_parser = subparsers.add_parser("uninstall", help="Uninstall Python packages")
    uninstall_parser.add_argument("packages", nargs="*", help="Packages to uninstall")
    uninstall_parser.set_defaults(func=_run_uninstall)

    jlinstall_parser = subparsers.add_parser("jlinstall", help="Install Julia packages")
    jlinstall_parser.add_argument("packages", nargs="*", help="Packages to install")
    jlinstall_parser.set_defaults(func=_run_jlinstall)

    jluninstall_parser = subparsers.add_parser("jluninstall", help="Uninstall Julia packages")
    jluninstall_parser.add_argument("packages", nargs="*", help="Packages to uninstall")
    jluninstall_parser.set_defaults(func=_run_jluninstall)

    # -- misc ----------------------------------------------------------------
    subparsers.add_parser("version", help="Print the version of FelooPy").set_defaults(func=_run_version)
    subparsers.add_parser("detect", help="Detect the system package manager").set_defaults(func=cli_detect)

    welcome_parser = subparsers.add_parser(
        "welcome",
        help="Show the welcome banner and setup wizard",
        description="Print the FelooPy banner and show the menu: a top level "
                    "of groups (Set up / Explore / Build / Maintain / "
                    "Packages / Help), each opening a submenu numbered from "
                    "1 with a 'Back to menu' entry. Levels nest to any "
                    "depth, every level replaces the one before it in place, "
                    "and every entry shows and runs the exact 'flp' line "
                    "behind it. A bare 'feloopy' / 'flp' / 'fly' invocation "
                    "shows this automatically.",
    )
    welcome_parser.set_defaults(func=_run_welcome)

    update_parser = subparsers.add_parser(
        "update-check",
        help="Check PyPI for a newer FelooPy release",
        description="Ask PyPI whether a newer FelooPy release exists. Automatic checks are "
                    "throttled to one request per machine per week; this command always checks.",
    )
    update_parser.add_argument("--offline", dest="force", action="store_false",
                               help="Only report the cached result; do not contact PyPI")
    update_parser.set_defaults(func=_run_update_check)

    return parser


def main(argv=None):
    """Entry point (``feloopy``/``flp``/``fly``).
    """
    argv = list(sys.argv[1:]) if argv is None else list(argv)

    wanted, first_positional = _scan_version_flags(argv)
    if wanted:
        cli_version()
        if first_positional and first_positional != "version":
            if first_positional in _command_names():
                sys.stderr.write(
                    f"note: the '{first_positional}' command was not run "
                    "(the version flag takes precedence).\n"
                )
            else:
                sys.stderr.write(
                    f"note: '{first_positional}' is not a FelooPy command "
                    "(ignored because a version flag was given).\n"
                )
        return

    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "version", False):      # kept for programmatic use
        cli_version()
    elif args.command and getattr(args, "func", None):
        args.func(args)
    else:
        if not argv and maybe_welcome():
            return
        parser.print_help()


if __name__ == "__main__":
    main()

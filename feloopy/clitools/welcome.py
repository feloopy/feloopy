# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""``flp welcome`` — banner and setup wizard for bare CLI runs.
"""

from __future__ import annotations

import os
import sys

from .._version import __version__

__all__ = ["BANNER", "cli_welcome", "maybe_welcome"]

#: The FelooPy logo in block characters (FELOOPY).
BANNER = """\
████ █▀▀▀▀ █    ████ ████ ████ █ █▌
█▄▄▄ █████ █    █  █ █  █ █▄▄█ ███▌
█    █▄▄▄▄ ████ ████ ████ █    ▄▄▄▌"""


# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

def _color_supported(stream):
    try:
        if not stream.isatty():
            return False
    except Exception:
        return False
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM", "") == "dumb":
        return False
    return True


def _print(text=""):
    """``print`` that survives consoles whose encoding lacks block characters."""
    try:
        print(text)
    except UnicodeEncodeError:
        try:
            sys.stdout.reconfigure(encoding="utf-8")     # Python 3.7+
            print(text)
        except Exception:
            encoding = getattr(sys.stdout, "encoding", None) or "ascii"
            print(text.encode(encoding, "backslashreplace").decode(encoding))


def _is_interactive():
    """True when both stdin and stdout are terminals (so prompting is safe)."""
    try:
        return (sys.stdin is not None and sys.stdin.isatty()
                and sys.stdout is not None and sys.stdout.isatty())
    except Exception:
        return False


def _colorama_wrapping():
    """True when colorama is sitting on ``sys.stdout``.

    On Windows ``colorama.init`` either enables VT processing on the
    console handle (so raw escapes are read natively) or, when it cannot,
    translates them into Win32 calls — either route honours cursor-up and
    erase-down, which is why wrapping plus a tty is enough for us.
    """
    try:
        cls = type(sys.stdout)
        return cls.__module__ == "colorama.ansitowin32" and cls.__name__ == "StreamWrapper"
    except Exception:
        return False


def _redraw_supported():
    """True when this terminal can erase lines we have already drawn.

    Menus are redrawn *in place* (erase the previous block, print the new
    one over it) rather than stacked underneath.  That needs cursor-up and
    erase-down escapes, which not every host understands — on Windows they
    only work when colorama is wrapping the stream or the host advertises
    VT processing.  When in doubt we simply append, i.e. behave as before.
    """
    if os.environ.get("FLOOPY_NO_REDRAW") is not None:
        return False
    try:
        if not sys.stdout.isatty():
            return False
    except Exception:
        return False
    if os.environ.get("TERM", "") == "dumb":
        return False
    if sys.platform != "win32":
        return True                        # POSIX terminals speak ANSI natively
    if _colorama_wrapping():
        return True
    return any(os.environ.get(name) for name in
               ("WT_SESSION", "TERM_PROGRAM", "ANSICON", "ConEmuANSI", "VSCODE_PID"))


def _ask(prompt, default=None):
    """Read a line; returns *default* on Enter, ``None`` on EOF/interrupt."""
    suffix = " [{}]".format(default) if default else ""
    try:
        raw = input("{}{}: ".format(prompt, suffix))
    except (EOFError, KeyboardInterrupt):
        try:
            print()
        except Exception:
            pass
        return None
    raw = raw.strip()
    return raw if raw else default


# ---------------------------------------------------------------------------
# Output pieces
# ---------------------------------------------------------------------------

def _print_banner():
    color = _color_supported(sys.stdout)
    for line in BANNER.splitlines():
        _print("\033[36m{}\033[0m".format(line) if color else line)


def _print_header():
    env = None
    try:
        from .inspectors import SOLVER_INTERFACES, probe_interface
        installed = sum(1 for name in SOLVER_INTERFACES if probe_interface(name)[0] is True)
        total = len(SOLVER_INTERFACES)
        env = "{}/{} solver interfaces installed".format(installed, total)
    except Exception:
        pass

    try:
        import platform
        env = "Python {} on {}{}".format(
            platform.python_version(),
            platform.system(),
            " — " + env if env else "",
        )
    except Exception:
        pass

    _print()
    _print("  FelooPy v{}".format(__version__))
    if env:
        _print("  {}".format(env))


def _print_tips():
    _print()
    _print("Next steps:")
    for line in (
        "  feloopy deps                       optional dependencies + preferred versions",
        "  feloopy install gurobi cylp        install them at FelooPy's preferred versions",
        "  feloopy solvers --missing          solver interfaces you can install",
        "  feloopy setup -l                   free-solver install status",
        "  feloopy algorithms --available     algorithms you can run right now",
        "  feloopy params gurobi gurobi       options for that interface+solver pair",
        "  feloopy info                       summarize this environment",
        "  feloopy run main.py                run a model script",
        "  feloopy welcome                    show this wizard again",
    ):
        _print(line)
    _print("Docs: https://github.com/feloopy/feloopy")


# ---------------------------------------------------------------------------
# Wizard steps
# ---------------------------------------------------------------------------

def _run_command(argv):
    """Run an ``flp`` command line through FelooPy's real argument parser.

    Going through the parser (instead of calling the helpers by hand)
    keeps the menu and the command line in lockstep: whatever
    ``flp deps`` / ``flp install`` / ... do at the terminal is exactly
    what the menu runs.  Returns ``True`` when the command completed.
    """
    argv = [str(a) for a in argv]
    try:
        from ..cli import build_parser
    except Exception as exc:
        _print("Cannot load the FelooPy CLI: {}".format(exc))
        return False
    try:
        args = build_parser().parse_args(argv)
    except SystemExit:
        return False               # argparse already printed usage/help
    func = getattr(args, "func", None)
    if func is None:
        _print("Nothing to run: flp {}".format(" ".join(argv)))
        return False
    try:
        func(args)
        return True
    except SystemExit as exc:
        if isinstance(exc.code, str):
            _print(exc.code)
        return False
    except Exception as exc:
        _print("Step failed: {}".format(exc))
        return False


def _cmd(*argv):
    """Menu action that runs a fixed ``flp`` command line."""
    def _step():
        _run_command(list(argv))
    _step.argv = list(argv)        # exposed so the menu can be checked
    return _step


def _ask_cmd(prefix, prompt, default=None, split=True):
    """Menu action that asks for words, then runs ``prefix + words``."""
    def _step():
        raw = _ask(prompt, default)
        if raw is None:
            return
        raw = str(raw).strip()
        if not raw:
            return
        words = [w for w in raw.replace(",", " ").split() if w] if split else [raw]
        if not words:
            return
        _print()
        _run_command(list(prefix) + words)
    _step.argv_prefix = list(prefix)   # exposed so the menu can be checked
    return _step


def _step_solvers():
    """Offer every solver: free ones install, commercial ones print a guide."""
    from ..helpers.solver_guides import COMMERCIAL_GUIDES, commercial_names
    from ..helpers.solver_provisioning import setup_status
    from .inspectors import SOLVER_INTERFACES, _print_table, probe_interface

    _print()
    _print("Free solvers — FelooPy downloads and installs these for you:")
    _print_table(
        ["solver", "type", "status", "version", "note"],
        [[row["name"],
          "pip" if row.get("kind") == "pip" else "binary",
          "installed" if row["installed"] else "missing",
          row["version"] or "",
          row["note"]] for row in setup_status()],
        status_col=2,
    )

    # Commercial solvers have no artifact feloopy can fetch (registration or
    # an installer is involved), so the offer is their official install guide.
    _print()
    _print("Commercial solvers — 'flp setup <name>' prints the vendor's install guide:")
    comm = []
    for name in commercial_names():
        if name in SOLVER_INTERFACES:
            status = "installed" if probe_interface(name)[0] else "missing"
        else:
            status = "guide only"      # a guide, but no feloopy interface
        comm.append([name, status, COMMERCIAL_GUIDES[name]["summary"]])
    _print_table(["solver", "status", "note"], comm, status_col=1)

    _print()
    _print("Both kinds go in the same box: free names install, commercial names print a guide.")
    names = _ask("Solvers to install (space/comma separated; empty to skip)",
                 "cbc highs")
    if not names:
        return
    parts = [p for p in names.replace(",", " ").split() if p]
    if not parts:
        return
    _print()
    if not _run_command(["setup"] + parts):
        _print("Some solvers could not be installed; "
               "re-run 'feloopy setup <name>' to retry.")


def _step_interfaces():
    """Offer to pip-install interfaces — free and commercial alike."""
    from ..helpers.solver_guides import is_commercial
    from .inspectors import SOLVER_INTERFACES, probe_interface
    from .deps import resolve_install
    from .utils import pip_install

    free = {}          # interface key -> probe (pip bindings only)
    commercial = {}
    free_ok = commercial_ok = 0
    exe_keys, exe_ok = [], set()
    not_pip = []
    for key, probe in SOLVER_INTERFACES.items():
        if probe.kind == "builtin":
            continue
        if probe.kind != "module" or not probe.dist:
            # exe-based (bonmin, couenne, picat): never pip-installable, but
            # they still belong in the list — they install through `flp setup`.
            exe_keys.append(key)
            if probe_interface(key)[0]:
                exe_ok.add(key)
            else:
                not_pip.append(key)
            continue
        # An interface is commercial when either its feloopy name or the pip
        # package behind it is one (cplex_cp -> "cplex", gams -> "gamspy", ...).
        paid = is_commercial(key) or is_commercial(probe.dist)
        if probe_interface(key)[0]:
            if paid:
                commercial_ok += 1
            else:
                free_ok += 1
        else:
            (commercial if paid else free)[key] = probe

    pip_installable = dict(free)
    pip_installable.update(commercial)

    _print()
    if not pip_installable:
        _print("All pip-installable interfaces are already installed "
               "({} free, {} commercial).".format(free_ok, commercial_ok))
        if not_pip:
            _print("Not pip-installable (install with 'flp setup <name>'): "
                   + ", ".join(not_pip))
        return

    _print("Interfaces you can install with pip — free and commercial:")
    for heading, group, ok in (("Free", free, free_ok),
                               ("Commercial", commercial, commercial_ok)):
        _print()
        _print("  {} ({} to install, {} installed)".format(heading, len(group), ok))
        for key, probe in group.items():
            _print("    {:<20} {:<22} {}".format(key, probe.dist, probe.note))
        if not group:
            _print("    (all installed)")
    if exe_keys:
        _print()
        _print("  Binaries/exes (via 'flp setup <name>'): {}".format(", ".join(
            "{} ({})".format(k, "installed" if k in exe_ok else "missing")
            for k in exe_keys)))
    _print()
    _print("  Tip: 'flp install <name>' adds one interface, pinned to FelooPy's preferred version.")
    _print("  Tip: 'flp deps' lists every optional dependency and its install status.")

    names = _ask("Interfaces to install (space/comma separated; empty to skip)")
    if not names:
        return
    wanted = [w for w in names.replace(",", " ").split() if w]
    if not wanted:
        return

    unknown = []
    wanted_keys = []
    for name in wanted:
        key = name.lower()
        probe = pip_installable.get(key)
        if probe is None:
            if key in SOLVER_INTERFACES:
                if probe_interface(key)[0]:
                    _print("{} is already installed.".format(key))
                else:
                    _print("{} is not installable with pip (see 'feloopy setup {}').".format(key, key))
            else:
                unknown.append(name)
            continue
        if key not in wanted_keys:
            wanted_keys.append(key)
    if unknown:
        _print("Unknown interface(s): {}".format(", ".join(unknown)))
    if not wanted_keys:
        return

    resolution = resolve_install(wanted_keys)
    _print()
    _print("Installing: {}".format(", ".join(resolution.requirements)))
    pip_install(resolution.requirements)
    _print()
    for key in wanted_keys:
        ok, version = probe_interface(key)
        if ok:
            _print("  [ok] {}{}".format(key, " " + version if version else ""))
        else:
            _print("  [??] {} — import did not succeed yet".format(key))


def _step_params():
    """Ask for an interface + solver pair, then show what that pair accepts.

    The passable ``options={...}`` depend on both halves — ``cvxpy``
    behind ``cbc`` is not ``cvxpy`` behind ``osqp``, and ``mealpy``
    behind ``pso`` is not ``mealpy`` behind ``de`` — so the pair is
    collected rather than guessed.
    """
    from .inspectors import _canon, normalize_interface, solvers_for

    interface = _ask("Interface to show options for", "gurobi")
    if interface is None:
        return
    interface = str(interface).strip()
    if not interface:
        return
    if normalize_interface(interface) is None:
        _print("'{}' is not a known FelooPy interface — run 'flp solvers' "
               "to list them.".format(interface))
        return

    choices = solvers_for(interface)
    if not choices:
        _print("No solver pairs with '{}' — run 'flp solvers' to list "
               "interfaces.".format(interface))
        return
    _print("Solvers for '{}': {}".format(
        interface,
        " ".join(choices[:12]) + (" …" if len(choices) > 12 else "")))
    if len(choices) > 12:
        _print("Full list: flp algorithms -i {}".format(interface))

    default = (interface
               if _canon(interface) in {_canon(c) for c in choices}
               else choices[0])
    solver = _ask("Solver to pair with '{}'".format(interface), default)
    if solver is None:
        return
    solver = str(solver).strip()
    if not solver:
        return
    _print()
    _run_command(["params", interface, solver])


_step_params.argv_prefix = ["params"]   # exposed so the menu can be checked


def _step_project():
    name = _ask("Project name", "my-optimization")
    if not name:
        return
    directory = _ask("Directory", ".") or "."
    _print()
    _run_command(["project", "--name", name, "--dir", directory])


def _step_help():
    """Print FelooPy's full command help (``flp --help``)."""
    try:
        from ..cli import build_parser
        build_parser().print_help()
    except SystemExit:
        return
    except Exception as exc:
        _print("Could not print the help text: {}".format(exc))


#: The menu as data: ``(heading, summary, entries)`` where every entry is
#: ``(key, label, command, target)``.  ``command`` is the exact ``flp``
#: line the entry runs, so the menu doubles as a cheat sheet for it, and
#: ``target`` executes it through the real argument parser.
#:
#: ``target`` is either a callable (a leaf entry) or another
#: ``(heading, summary, entries)`` menu — nesting them builds
#: sub-sub-menus to any depth.  The headings below form the top level;
#: picking one opens that group's submenu.  Every submenu is numbered
#: from 1 and ends with "Back to menu" / "Finish", so a level above is
#: only ever redrawn when you deliberately climb back to it.
_MENU = (
    ("Set up", "install and check solvers & interfaces", (
        ("1", "Install solvers", "flp setup <name>",
         _step_solvers),
        ("2", "Install interfaces", "flp install <name>",
         _step_interfaces),
        ("3", "See which interfaces are ready", "flp solvers",
         _cmd("solvers")),
        ("4", "See which solvers are ready", "flp setup --list",
         _cmd("setup", "--list")),
    )),
    ("Explore", "algorithms, solver options, environment", (
        ("1", "Algorithms you can run right now", "flp algorithms --available",
         _cmd("algorithms", "--available")),
        ("2", "Solver options for flp.search(...)", "flp params <interface> <solver>",
         _step_params),
        ("3", "Summarize this environment", "flp info",
         _cmd("info")),
        ("4", "Detect the system package manager", "flp detect",
         _cmd("detect")),
    )),
    ("Build", "sample project, run a script, next-step tips", (
        ("1", "Scaffold a sample project", "flp project --name ...",
         _step_project),
        ("2", "Run a model script", "flp run main.py",
         _ask_cmd(["run"], "Python file to run", "main.py", split=False)),
        ("3", "Show next-step tips", "",
         _print_tips),
    )),
    ("Maintain", "update, clean, back up, recover, bundle", (
        ("1", "Check for a newer FelooPy release", "flp update-check",
         _cmd("update-check")),
        ("2", "Clean caches, logs and compiled files", "flp clean",
         _cmd("clean")),
        ("3", "Back up the project", "flp backup",
         _cmd("backup")),
        ("4", "Recover the project from a backup", "flp recover",
         _cmd("recover")),
        ("5", "Bundle the project into a zip", "flp build",
         _cmd("build")),
        ("6", "Install VS Code extensions", "flp ext <id>",
         _ask_cmd(["ext"], "VS Code extension ids to install")),
    )),
    ("Packages", "install and remove Python / Julia packages", (
        ("1", "Uninstall Python packages", "flp uninstall <name>",
         _ask_cmd(["uninstall"], "Packages to uninstall")),
        ("2", "Install Julia packages", "flp jlinstall <name>",
         _ask_cmd(["jlinstall"], "Julia packages to install")),
        ("3", "Uninstall Julia packages", "flp jluninstall <name>",
         _ask_cmd(["jluninstall"], "Julia packages to uninstall")),
    )),
    ("Help", "full command help, version", (
        ("1", "Show the full command help", "flp --help",
         _step_help),
        ("2", "Print the FelooPy version", "flp -v",
         _cmd("version")),
    )),
)

#: Spellings that finish the whole wizard.
_DONE = {"s", "skip", "q", "quit", "exit", "done", "0"}

#: Spellings that climb one level (at the top level, which has no parent,
#: they simply redraw it).
_BACK = {"b", "back", "menu", "m", "<"}

#: Returned by one menu level: the user wants the enclosing menu...
_SIG_BACK = object()
#: ...or the wizard is over (Finish, EOF or interrupt).
_SIG_DONE = object()

#: Left-hand column: 4 spaces + "[" + 2-char key + "]" + 2 spaces.
_KEY_COL = 10

#: Lines the menu block currently occupies on screen — its rows plus the
#: ``Choice: `` prompt printed under them — so the next menu can replace
#: it in place instead of being stacked underneath.  0 = nothing to erase.
_block = [0]


def _term_width():
    try:
        import shutil
        return shutil.get_terminal_size((80, 24)).columns
    except Exception:
        return 80


def _erase(count):
    """Clear the last ``count`` lines so a menu can be redrawn in place."""
    if count <= 0 or not _redraw_supported():
        return
    try:
        # Move to the start of the block, then wipe from there downwards.
        sys.stdout.write("\033[{}A\033[0J".format(count))
        sys.stdout.flush()
    except Exception:
        pass


def _drop_block():
    """Forget the on-screen menu block; something else is about to print."""
    _block[0] = 0


def _clear_block():
    """Erase the on-screen menu block; something else is about to print."""
    _erase(_block[0])
    _drop_block()


def _draw(lines):
    """Print a menu block, replacing the one already on screen if we can."""
    _clear_block()
    for line in lines:
        _print(line)
    # Claim the block only when erasing it later is safe: we need a terminal
    # that understands the escapes, and no line may wrap onto a second row
    # (cursor-up counts rows, not logical lines).
    if (_redraw_supported() and lines
            and all(len(line) + 1 <= _term_width() for line in lines)):
        _block[0] = len(lines) + 1         # + the "Choice: " prompt underneath


def _is_node(target):
    """True when a row's target is another menu instead of an action.

    A menu is ``(heading, summary, entries)`` whose entries are 4-tuples;
    an action is a callable, and a row is a 4-tuple, so neither matches.
    """
    if not isinstance(target, (tuple, list)) or len(target) != 3:
        return False
    entries = target[2]
    if not isinstance(entries, (tuple, list)) or not entries:
        return False
    first = entries[0]
    return isinstance(first, (tuple, list)) and len(first) == 4


def _root_node():
    """The top level: one numbered row per group of ``_MENU``.

    Each row's target is the group *node* itself, so the root is just
    another menu and the walk from there down is uniform.
    """
    rows = [(str(index), node[0], node[1], node)
            for index, node in enumerate(_MENU, start=1)]
    return ("What would you like to do?", "", tuple(rows))


def _target_for(node, choice):
    """The row matching ``choice`` in ``node``, or ``None``."""
    _heading, _summary, entries = node
    for key, _label, _command, target in entries:
        if key == choice:
            return target
    return None


def _menu_lines(node, root=False):
    """Render one menu level: heading, rows, then Back (unless root) + Finish.

    Every level renders exactly the same way — that is what lets any
    level be drawn over any other one in place, however deep the tree.
    """
    heading, _summary, entries = node
    width = max(len(label) for _key, label, _command, _target in entries)
    lines = ["", heading if root else "  {}".format(heading)]
    for key, label, command, _target in entries:
        line = "    [{:>2}]  {:<{}}".format(key, label, width)
        if command:
            line = "{:<{w}}  {}".format(line, command, w=_KEY_COL + width + 2)
        lines.append(line)
    lines.append("")
    if not root:
        lines.append("    [ b]  Back to menu")
    lines.append("    [ s]  Finish")
    return lines


def _main_menu_lines():
    """Top level: header, one row per group, then Finish."""
    return _menu_lines(_root_node(), root=True)


def _submenu_lines(heading, entries):
    """A submenu: heading, its rows, then Back and Finish."""
    return _menu_lines((heading, "", tuple(entries)))


def _print_main_menu():
    """Print the top level without touching whatever is already on screen."""
    for line in _main_menu_lines():
        _print(line)


def _print_submenu(heading, entries):
    """Print one group's submenu without touching what is on screen."""
    for line in _submenu_lines(heading, entries):
        _print(line)


def _run_menu(node, root=False):
    """Run one menu level, drawing it in place over the level before it.

    Returns ``_SIG_DONE`` when the whole wizard is over (Finish, EOF or
    interrupt) and ``_SIG_BACK`` when the user wants the enclosing menu.
    The root has no parent, so ``[b]`` there just redraws it.
    """
    while True:
        _draw(_menu_lines(node, root))

        choice = _ask("Choice", "s" if root else "b")
        if choice is None:
            return _SIG_DONE
        choice = str(choice).strip().lower()
        if choice in _DONE:
            return _SIG_DONE
        if choice in _BACK:
            if root:
                continue           # nothing above the top level: redraw
            return _SIG_BACK

        target = _target_for(node, choice)
        if target is None:
            _clear_block()          # keep the complaint, drop the menu
            _print("Unknown choice: {!r}".format(choice))
            continue

        if _is_node(target):
            if _run_menu(target) is _SIG_DONE:
                return _SIG_DONE
            continue                # it went back: redraw this level

        # A leaf entry: clear this level first, so its output takes this
        # spot and the level is redrawn underneath — never twice.
        _clear_block()
        try:
            target()
        except Exception as exc:
            _print("Step failed: {}".format(exc))


def _wizard():
    """Run the menu tree; returns when the user finishes or input ends."""
    _run_menu(_root_node(), root=True)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_welcome(interactive=None):
    """Print the banner and run the setup flow.  Never raises."""
    if interactive is None:
        interactive = _is_interactive()
    try:
        _print()
        _print_banner()
        _print_header()
        if interactive:
            _print()
            _print("Let's get you set up — pick a group to open it "
                   "(b goes back one level, s finishes, "
                   "Enter accepts a default).")
            _wizard()
        else:
            _print_tips()
    except (KeyboardInterrupt, EOFError):
        _print()
    except Exception as exc:
        _print("Welcome wizard error (continuing): {}".format(exc))


def maybe_welcome():
    """Handle a bare ``feloopy`` / ``flp`` / ``fly`` invocation.

    Always shows the welcome screen.  Returns True so the caller skips
    printing the help text.  Never raises.
    """
    try:
        run_welcome()
        return True
    except Exception:
        return False


def cli_welcome(args=None):
    """``feloopy welcome`` — show the banner and setup wizard."""
    try:
        run_welcome()
    except Exception as exc:
        sys.stderr.write("feloopy welcome failed: {}\n".format(exc))

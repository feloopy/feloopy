# Copyright (c) 2022-2025, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import argparse

try:
    import tkinter as tk
    from tkinter import filedialog
except:
    pass

from .clitools import *

def cli_detect():
    detect_package_manager(verbose=True)

def main():
    parser = argparse.ArgumentParser(description="FelooPy's command-line tool")
    subparsers = parser.add_subparsers(dest="command", title="commands", description="Valid commands")


    parser.add_argument("--version", action="store_true", help="Print the version of FelooPy")

    setup_parser = subparsers.add_parser("setup", help="Run setup for FelooPy")

    project_parser = subparsers.add_parser("project", help="Create or manage a project")
    project_parser.add_argument("--name", required=True, help="Name of the optimization project")
    project_parser.add_argument("--type", help="Specify project type")

    backup_parser = subparsers.add_parser("backup", help="Create a backup of the project")
    backup_parser.add_argument("name", nargs="?", help="Optional name for the backup file")
    backup_parser.set_defaults(func=zip_project)

    recover_parser = subparsers.add_parser("recover", help="Recover project from a backup")
    recover_parser.add_argument("name", nargs="?", help="Name of the backup file to recover")
    recover_parser.set_defaults(func=recover_project)

    clean_parser = subparsers.add_parser("clean", help="Clean the project")

    build_parser = subparsers.add_parser("build", help="Build the project")

    run_parser = subparsers.add_parser("run", help="Run a Python file")
    run_parser.add_argument("file", nargs="?", help="Python file to run")

    ext_parser = subparsers.add_parser("ext", help="Install VSCode extensions")
    ext_parser.add_argument("extensions", nargs="*", help="Extensions to install")

    install_parser = subparsers.add_parser("install", help="Install Python packages")
    install_parser.add_argument("-u", "--update", action="store_true", help="Update installed packages to the latest versions")
    install_parser.add_argument("packages", nargs="*", help="Packages to install")

    jlinstall_parser = subparsers.add_parser("jlinstall", help="Install Julia packages")
    jlinstall_parser.add_argument("packages", nargs="*", help="Packages to install")

    jluninstall_parser = subparsers.add_parser("jluninstall", help="Uninstall Julia packages")
    jluninstall_parser.add_argument("packages", nargs="*", help="Packages to uninstall")

    uninstall_parser = subparsers.add_parser("uninstall", help="Uninstall Python packages")
    uninstall_parser.add_argument("packages", nargs="*", help="Packages to uninstall")

    args = parser.parse_args()

    if args.version:
        cli_version()
    elif args.command:
        if hasattr(args, "func"):
            args.func(args)
        else:
            print(f"Executing command: {args.command}")
    else:
        parser.print_help()

if __name__ == "__main__":
   main()
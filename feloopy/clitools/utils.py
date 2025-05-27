# Copyright (c) 2022-2025, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import os
import argparse
import getpass
try:
    import tkinter as tk
    from tkinter import filedialog
except:
    pass

import nbformat
import shutil
import zipfile
import subprocess
import sys
import urllib.request
from tqdm import tqdm
from datetime import datetime
from .._version import __version__

__version__ = f"v{__version__}"

def download_and_extract(url, output_folder, filename):
   print(f"Downloading and unpacking {filename}")
   with urllib.request.urlopen(url) as url:
       f = open(filename, 'wb')
       total_size = int(url.info().get("Content-Length", 0))
       chunk_size = 1024 # 1 KB
       pbar = tqdm(total=total_size, unit='iB', unit_scale=True)
       for data in iter(lambda: url.read(chunk_size), b''):
           f.write(data)
           pbar.update(len(data))
       f.close()
       pbar.close()

   if "zip" in filename:
    shutil.unpack_archive(filename, output_folder)
    os.remove(filename)
   elif "tar" in filename:
    shutil.unpack_archive(filename, output_folder, format='gztar')
    os.remove(filename)

def ask_for_directory():
 root = tk.Tk()
 root.withdraw()
 return filedialog.askdirectory()

def run_setup_file():
 try:
     install_dir = ask_for_directory()
 except Exception:
     install_dir = input("Enter the directory where you want to install the solvers: ")

 os.chdir(install_dir)

 os.makedirs("solvers", exist_ok=True)

 versions = {
     "cbc": ("2.10.12", "https://github.com/coin-or/Cbc/releases/download/releases%2F{version}/Cbc-releases.{version}-x86_64-w64-mingw64.zip"),
     "highs": ("1.7.2", "https://github.com/JuliaBinaryWrappers/HiGHSstatic_jll.jl/releases/download/HiGHSstatic-v{version}%2B0/HiGHSstatic.v{version}.x86_64-w64-mingw32.tar.gz"),
     "ipopt": ("3.14.16", "https://github.com/coin-or/Ipopt/releases/download/releases%2F{version}/Ipopt-{version}-win64-msvs2019-md.zip"),
     "glpk": ("4.65", "https://sourceforge.net/projects/winglpk/files/winglpk/GLPK-{version}/winglpk-{version}.zip/download"),
     "bonmin": ("1.4.0", "https://www.coin-or.org/download/binary/Bonmin/Bonmin-{version}-win32-msvc9.zip"),
     "couenne": ("0.3.2", "https://www.coin-or.org/download/binary/Couenne/Couenne-{version}-win32-msvc9.zip"),
     "scip": ("9.1.1", "https://github.com/scipopt/scip/releases/download/v{version_no_dots}/SCIPOptSuite-{version}-win64-VS15.exe"),
     "git": ("2.47.0", "https://github.com/git-for-windows/git/releases/download/v{version}.windows.1/Git-{version}-64-bit.exe"),
 }

 for solver, (version, url) in versions.items():
    url = url.format(version=version, version_no_dots=version.replace(".", ""))
    output_folder = os.path.join("solvers", solver + "-windows")
    if "zip" in url.lower():
        filename = os.path.join("solvers", solver + "-windows.zip")
    if "tar" in url.lower():
        filename = os.path.join("solvers", solver + "-windows.tar.gz")
    if "exe" in url.lower():
        filename = os.path.join("solvers", solver + "-windows.exe")
    try:
        download_and_extract(url, output_folder, filename)
    except:
        pass
def create_optimization_project(project_name, directory=".", project_type=None):

   project_dir = os.path.join(directory, project_name)

   os.makedirs(project_dir, exist_ok=True)

   subdirectories = ["data", "modules", "results"]

   data_dir = os.path.join(project_dir, "data")
   for subdir in ["raw", "processed", "final"]:
       subdir_path = os.path.join(data_dir, subdir)
       os.makedirs(subdir_path, exist_ok=True)

   data_dir = os.path.join(project_dir, "results")
   for subdir in ["tables", "figures", "texts"]:
       subdir_path = os.path.join(data_dir, subdir)
       os.makedirs(subdir_path, exist_ok=True)

   for subdirectory in subdirectories:
       subdirectory_path = os.path.join(project_dir, subdirectory)
       os.makedirs(subdirectory_path, exist_ok=True)

   modules_init_path = os.path.join(project_dir, "modules", "__init__.py")
   with open(modules_init_path, "w") as modules_init_file:
       modules_init_file.write("# Initialization file for the modules directory")

   main_file_path = os.path.join(project_dir, "main.py")
   with open(main_file_path, "w") as main_file:
       main_file.write(f"""
# name: {project_name}
# author: {get_user_name()}
# feloopy version: {__version__}
# date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
""")

   ipynb_file_path = os.path.join(project_dir, f"debug.ipynb")
   with open(ipynb_file_path, "w") as ipynb_file:
       nb = nbformat.v4.new_notebook()
       nb['metadata'] = {
           'name:': project_name,
           'author:': get_user_name(),
           'feloopy version:': __version__,
           'date:': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
       }
       nbformat.write(nb, ipynb_file)

   log_file_path = os.path.join(project_dir, "log.txt")
   with open(log_file_path, "w") as log_file:
       log_file.write(f"""
# log for {project_name}
# author: {get_user_name()}
# date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

# Notes, to-dos, and other information can be added here.
""")

   if project_type:
       project_type_dir = os.path.join(project_dir, '.' + project_type)
       os.makedirs(project_type_dir, exist_ok=True)

   print(f"Optimization project '{project_name}' created at: {project_dir}")

def get_user_name():
    return getpass.getuser()

def run_project(file_path):
    try:
        subprocess.run(["python", file_path], check=True)
    except subprocess.CalledProcessError as e:
        try:
            subprocess.run([sys.executable, file_path], check=True)
        except:
            print(f"Error running the project: {e}")

def julia_install(packages):
    from juliacall import Main as jl
    for package in packages:
        try:
            jl.seval(f'import Pkg; Pkg.add("{package}")')
            print(f"Successfully installed {package} with Julia Pkg")
        except Exception as e:
            print(f"Error installing {package} with Julia Pkg: {e}")

def julia_uninstall(packages):
    from juliacall import Main as jl
    for package in packages:
        try:
            jl.seval(f'import Pkg; Pkg.rm("{package}")')
            print(f"Successfully uninstalled {package} with Julia Pkg")
        except Exception as e:
            print(f"Error uninstalling {package} with Julia Pkg: {e}")

def pip_install(packages, update=False):
   for package in packages:
       try:
           command = [sys.executable, '-m', 'pip', 'install', '--verbose', package]
           if update:
               command.insert(4, '--upgrade')
           subprocess.run(command, check=True)
           print(f"Successfully {'updated' if update else 'installed'} {package} with pip")
       except subprocess.CalledProcessError as pip_error:
           print(f"Error {'updating' if update else 'installing'} {package} with pip: {pip_error}")
           try:
               command = ['conda', 'install', '--yes', package]
               if update:
                  command.append('--update-all')
               subprocess.run(command, check=True)
               print(f"Successfully {'updated' if update else 'installed'} {package} with conda")
           except (subprocess.CalledProcessError, FileNotFoundError) as conda_error:
               print(f"Error {'updating' if update else 'installing'} {package} with conda: {conda_error}")
               try:
                  command = [sys.executable, '-m', 'pipx', 'install', '--verbose', package]
                  if update:
                      command.insert(4, '--upgrade')
                  subprocess.run(command, check=True)
                  print(f"Successfully {'updated' if update else 'installed'} {package} with pipx")
               except subprocess.CalledProcessError as pipx_error:
                  print(f"Error {'updating' if update else 'installing'} {package} with pipx: {pipx_error}")

def pip_uninstall(packages):
    for package in packages:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', package], check=True)
            print(f"Successfully uninstalled {package} with pip")
        except subprocess.CalledProcessError as pip_uninstall_error:
            print(f"Error uninstalling {package} with pip: {pip_uninstall_error}")
            try:
                subprocess.run(['conda', 'remove', '--yes', package], check=True)
                print(f"Successfully uninstalled {package} with conda")
            except (subprocess.CalledProcessError, FileNotFoundError) as conda_uninstall_error:
                print(f"Error uninstalling {package} with conda: {conda_uninstall_error}")
                try:
                    subprocess.run([sys.executable, '-m', 'pipx', 'uninstall', package], check=True)
                    print(f"Successfully uninstalled {package} with pipx")
                except subprocess.CalledProcessError as pipx_uninstall_error:
                    print(f"Error uninstalling {package} with pipx: {pipx_uninstall_error}")

def get_current_date():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")

def select_directory():
    try:
        root = tk.Tk()
        root.withdraw()
        directory = filedialog.askdirectory(title="Select Project Directory")
        root.destroy()
        return directory
    except:
        print("Error: Unable to use graphical file dialog. Please enter the directory manually.")
        return input("Enter the project directory: ")

def cli_version():
    print(f"FelooPy ({__version__})")

def cli_project(args):
    directory = select_directory()
    if directory:
        create_optimization_project(args.name, directory, args.type)

def zip_project(args=None):
    backup_dir = os.path.join(".", "backups")
    os.makedirs(backup_dir, exist_ok=True)

    if args and getattr(args, "name", None):
        zip_file_name = f"{args.name}.zip"
    else:
        current_datetime = get_current_datetime()
        zip_file_name = f"bkp-{current_datetime}.zip"

    zip_file_path = os.path.join(backup_dir, zip_file_name)

    all_files = [
        os.path.relpath(os.path.join(dp, f), start=".")
        for dp, dn, filenames in os.walk(".")
        for f in filenames + dn
        if "backups" not in os.path.join(dp, f)
    ]

    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        for file in all_files:
            zipf.write(file, arcname=file)

    print(f"Project (excluding 'backups' directory) zipped and saved at: {zip_file_path}")

def debug_print(message):
    print(f"[DEBUG] {message}")

def extract_src(zip_file_path):
    try:
        zip_file_path = os.path.abspath(zip_file_path)
        debug_print(f"Attempting to open: {zip_file_path}")

        if not os.path.exists(zip_file_path):
            raise FileNotFoundError(f"The file '{zip_file_path}' does not exist.")

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall('.')
        debug_print(f"Successfully extracted '{zip_file_path}' to the current directory.")

    except Exception as e:
        print(f"Error extracting '{zip_file_path}': {e}", file=sys.stderr)
        sys.exit(1)

def recover_project(args):
    try:
        backups_dir = os.path.abspath("./backups")
        debug_print(f"Looking for backups in: {backups_dir}")

        if not os.path.exists(backups_dir):
            raise FileNotFoundError(f"The backups directory '{backups_dir}' does not exist.")

        backup_files = [f for f in os.listdir(backups_dir) if f.endswith('.zip')]
        if not backup_files:
            raise FileNotFoundError("No backup files found in the backups directory.")

        print("Recent project backups:")
        for i, file in enumerate(backup_files, start=1):
            print(f"{i} | {file}")

        if hasattr(args, 'name') and args.name:
            selected_backup = args.name
            if ".zip" not in selected_backup:
                selected_backup+=".zip"
        else:
            selection = input("Enter the number of the backup to recover: ")
            try:
                selection = int(selection)
                if selection < 1 or selection > len(backup_files):
                    raise ValueError("Invalid selection.")
            except ValueError as e:
                print(f"Invalid input: {e}", file=sys.stderr)
                sys.exit(1)
            selected_backup = backup_files[selection - 1]

        backup_path = os.path.join(backups_dir, selected_backup)
        debug_print(f"Selected backup: {backup_path}")

        if not os.path.exists(backup_path):
            raise FileNotFoundError(f"The backup file '{backup_path}' does not exist.")

        extract_src(backup_path)

    except Exception as e:
        print(f"Error recovering project: {e}", file=sys.stderr)
        sys.exit(1)

def get_recent_src_files():
    backup_dir = os.path.join(".", "backups")
    src_files = [f for f in os.listdir(backup_dir) if f.startswith("bkp-") and f.endswith(".zip")]
    src_files.sort(reverse=True)
    return src_files[:10]

def delete_all_except_backup():
    for item in os.listdir("."):
        if os.path.isdir(item) and item != "backups":
            shutil.rmtree(item)
        elif os.path.isfile(item) and item != "backups":
            os.remove(item)

def extract_src(src_file):
    backup_dir = os.path.join(".", "backups")
    zip_file_path = os.path.join(backup_dir, src_file)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(".")

def get_current_datetime():
    from datetime import datetime
    return datetime.now().strftime("on-%Y-%m-%d-at-%H-%M-%S")


def build_project():

    backup_dir = os.path.join(".", "backups")
    os.makedirs(backup_dir, exist_ok=True)

    venv_name = input("Enter the name of the virtual environment (venv): ")

    if not venv_name:
        print("Error: Virtual environment (venv) name is required.")
        return

    venv_dir = os.path.abspath(os.path.join('..', venv_name))

    if not os.path.exists(venv_dir):
        print(f"Error: Virtual environment (venv) directory '{venv_dir}' not found.")
        return

    project_venv_dir = os.path.join(".", venv_name)

    shutil.copytree(venv_dir, project_venv_dir, symlinks=True, ignore=shutil.ignore_patterns('*.pyc', '__pycache__'))

    current_datetime = get_current_datetime()
    zip_file_name = f"build-{current_datetime}.zip"
    zip_file_path = os.path.join(backup_dir, zip_file_name)

    all_files = [os.path.relpath(os.path.join(dp, f), start=".") for dp, dn, filenames in os.walk(".") for f in filenames + dn if "backups" not in os.path.join(dp, f)]

    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        for file in all_files:
            zipf.write(file, arcname=file)

    build_dir_name = f"build"
    build_dir_path = os.path.join(".", build_dir_name)
    os.makedirs(build_dir_path, exist_ok=True)

    build_zip_file_path = os.path.join(build_dir_path, zip_file_name)
    shutil.move(zip_file_path, build_zip_file_path)

    print(f"Project including '{venv_name}' directory built and saved at: {build_zip_file_path}")

def install_vscode_extensions(extensions):
    print(f"Installing VS Code extensions: {extensions}")
    try:
        subprocess.run(['code', '--install-extension', *extensions], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error installing VS Code extensions: {e}")

def clean_project():

    extensions_to_delete = ['.pyc', '.pyo', '.pyd', '.py~', '.log', '.zip']
    directories_to_delete = ['__pycache__']

    for root, dirs, files in os.walk("."):
        for file in files:
            if any(file.endswith(ext) for ext in extensions_to_delete) and "backups" not in root:
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

        for directory in dirs:
            if directory in directories_to_delete and "backups" not in root:
                dir_path = os.path.join(root, directory)
                shutil.rmtree(dir_path)
                print(f"Deleted: {dir_path}")

    print("Project cleaned successfully.")

def get_installed_dependencies():
    result = subprocess.run(['pip', 'freeze'], stdout=subprocess.PIPE, text=True)
    print(result.stdout)
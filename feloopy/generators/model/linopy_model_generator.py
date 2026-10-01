# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import os
import sys

def _import_linopy_quietly():
    stderr_fd = sys.stderr.fileno()
    stdout_fd = sys.stdout.fileno()
    saved_stderr_fd = os.dup(stderr_fd)
    saved_stdout_fd = os.dup(stdout_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, stderr_fd)
        os.dup2(devnull, stdout_fd)
        from linopy import Model as LINOPYMODEL
    finally:
        os.dup2(saved_stderr_fd, stderr_fd)
        os.dup2(saved_stdout_fd, stdout_fd)
        os.close(saved_stderr_fd)
        os.close(saved_stdout_fd)
        os.close(devnull)
    return LINOPYMODEL

LINOPYMODEL = _import_linopy_quietly()

def generate_model(features):

    return LINOPYMODEL()

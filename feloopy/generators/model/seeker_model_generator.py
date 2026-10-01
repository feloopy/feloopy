# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

import os


def generate_demo_model(features):
    try:
        import seekerdemo as skr
        return skr.Env("license.sio")
    except (ImportError, Exception) as e:
        raise RuntimeError(
            "Could not load seekerdemo. "
            "Install it with: flp install insideopt-demo\n"
            f"Original error: {e}"
        )


def generate_model(features):
    try:
        import seeker as skr
    except ImportError as e:
        raise RuntimeError(
            "Could not load seeker. "
            "Install it with: flp install insideopt\n"
            f"Original error: {e}"
        )

    solver_options = features.get('solver_options', {})
    license_path = solver_options.get('license', None)

    if license_path is None:
        license_path = os.environ.get('SEEKER_LICENSE', None)

    if license_path is None:
        raise RuntimeError(
            "No seeker license file specified. "
            "Provide a license via solver_options={'license': '/path/to/license.sio'} "
            "or set the SEEKER_LICENSE environment variable."
        )

    if not os.path.isfile(license_path):
        raise RuntimeError(f"License file not found: '{license_path}'")

    try:
        return skr.Env(license_path)
    except Exception as e:
        raise RuntimeError(
            f"Could not create seeker Env with license '{license_path}'.\n"
            f"Original error: {e}"
        )

# Copyright (c) 2022-2025, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

def generate_demo_model(features):
    import seekerdemo as skr
    return skr.Env("license.sio")

def generate_model(features):
    import seeker as skr
    return skr.Env("license.sio")

# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

from .ortools_proxy import OrtoolsModelProxy


def generate_model(features):
    return OrtoolsModelProxy()

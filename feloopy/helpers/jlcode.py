# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.

"""Julia/JuMP code-injection hooks used by the jump solution generator.

Snippets are staged under ``features['jlcode_*']`` keys and concatenated
by ``generators.solution.jump_solution_generator`` in solve order.
"""

import os

__all__ = ["JLCodeHandler"]


class JLCodeHandler:
    """Mixin: append Julia code snippets onto model ``features``."""

    def _append_jlcode(self, key, code):
        if key not in self.features:
            self.features[key] = f"\n{code}"
        else:
            self.features[key] += f"\n{code}"

    def jlcode_preamble(self, code):
        """Code emitted before the rest of the generated Julia model."""
        self._append_jlcode("jlcode_preamble", code)

    def jlcode_before_variables(self, code):
        """Code emitted before variable declarations."""
        self._append_jlcode("jlcode_before_variables", code)

    def jlcode_before_constraints(self, code):
        """Code emitted before constraint declarations."""
        self._append_jlcode("jlcode_before_constraints", code)

    def jlcode_before_objectives(self, code):
        """Code emitted before objective declarations."""
        self._append_jlcode("jlcode_before_objectives", code)

    def jlcode_before_solve(self, code):
        """Code emitted before the solve call."""
        self._append_jlcode("jlcode_before_solve", code)

    def jlcode_after_solve(self, code):
        """Code emitted after the solve call."""
        self._append_jlcode("jlcode_after_solve", code)

    def jlcode_data(self, data):
        """Serialize ``data`` to JSON and stage the Julia loader snippet."""
        import json
        import numpy as np
        from ..helpers._lazy import pl

        if isinstance(data, dict):
            data_dict = data.copy()
        else:
            data_dict = data.data.copy()

        def convert_value(value):
            if isinstance(value, pl.DataFrame):
                return value.to_dict(as_series=False)
            elif isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, set):
                return list(value)
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            elif isinstance(value, (list, tuple)):
                return [convert_value(v) for v in value]
            else:
                return value

        converted_dict = {k: convert_value(v) for k, v in data_dict.items()}

        file_path = './__pycache__/data.json'
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(converted_dict, f)

        code = """
using JSON
using DataFrames

data = JSON.parsefile("./__pycache__/data.json")

function convert_value(value)
    if typeof(value) == Dict
        if all(x -> typeof(x) == Vector{Any}, values(value))
            try
                return DataFrame(value)
            catch
                return Dict(k => convert_value(v) for (k, v) in value)
            end
        else
            return Dict(k => convert_value(v) for (k, v) in value)
        end
    elseif typeof(value) == Vector{Any}
        return [convert_value(v) for v in value]
    else
        return value  # Base case
    end
end

converted_data = Dict(k => convert_value(v) for (k, v) in data)

for item in converted_data
    key = Symbol(item[1])
    value = item[2]
    @eval global $key = $value
end
"""
        self.features["jlcode_data"] = code

# Copyright (c) 2022-2026, Keivan Tafakkori. All rights reserved.
# See the file LICENSE file for licensing details.


def generate_model(features):

    model_name = features.get('model_name', 'feloopy_model')
    env = features.get('env', None)
    log = features.get('log', True)

    if not log:
        try:
            import os, sys, contextlib, io
            import gurobipy as gurobi_interface
            try:
                old_stdout_fd = os.dup(sys.stdout.fileno())
                old_stderr_fd = os.dup(sys.stderr.fileno())
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, sys.stdout.fileno())
                os.dup2(devnull, sys.stderr.fileno())
                os.close(devnull)
                try:
                    silent_env = gurobi_interface.Env(empty=True)
                    silent_env.setParam('OutputFlag', 0)
                    try:
                        silent_env.setParam('LogToConsole', 0)
                    except Exception:
                        pass
                    silent_env.start()
                    use_env = env if env is not None else silent_env
                    model_object = gurobi_interface.Model(name=model_name, env=use_env)
                    if env is None:
                        model_object._feloopy_silent_env = silent_env
                    else:
                        silent_env.dispose()
                finally:
                    os.dup2(old_stdout_fd, sys.stdout.fileno())
                    os.dup2(old_stderr_fd, sys.stderr.fileno())
                    os.close(old_stdout_fd)
                    os.close(old_stderr_fd)
                return model_object
            except Exception:
                pass
            try:
                silent_env = gurobi_interface.Env(empty=True)
                silent_env.setParam('OutputFlag', 0)
                silent_env.start()
                use_env = env if env is not None else silent_env
                model_object = gurobi_interface.Model(name=model_name, env=use_env)
                if env is None:
                    model_object._feloopy_silent_env = silent_env
                else:
                    silent_env.dispose()
                return model_object
            except Exception:
                pass
        except Exception:
            pass
    import gurobipy as gurobi_interface
    if env is not None:
        model_object = gurobi_interface.Model(name=model_name, env=env)
    else:
        model_object = gurobi_interface.Model(model_name)

    model_params = features.get('model_params', {})
    for key, value in model_params.items():
        model_object.setParam(key, value)

    return model_object

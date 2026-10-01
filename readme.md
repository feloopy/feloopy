<div align="center">

  <p>
    <a href="https://feloopy.github.io" target="_blank">
      <picture>
        <source media="(prefers-color-scheme: light)" srcset="https://github.com/feloopy/feloopy/raw/main/feloopy/assets/logo-black.svg">
        <source media="(prefers-color-scheme: dark)" srcset="https://github.com/feloopy/feloopy/raw/main/feloopy/assets/logo-white.svg">
        <img alt="FelooPy's logo." src="https://github.com/feloopy/feloopy/raw/main/feloopy/assets/logo-black.svg" width="100" height="auto">
      </picture>
    </a>
  </p>

</div>


<div align="center">

  <h1>FelooPy</h1>

  <strong>Efficient & Feature-Rich Decision Engine</strong>
</div>

<div align="center" style="margin-bottom: 2px;">

![PyPI](https://img.shields.io/pypi/v/feloopy?color=red&label=version&labelColor=red)
![Release Date](https://img.shields.io/github/release-date/feloopy/feloopy?label=release&color=red&labelColor=red)
[![Downloads](https://static.pepy.tech/personalized-badge/feloopy?period=total&units=international_system&left_color=red&right_color=red&left_text=downloads)](https://pepy.tech/project/feloopy)
![License](https://img.shields.io/static/v1?label=license&message=MIT&color=red&labelColor=red)

</div>

---

FelooPy is a plug-and-play operations research workflow for mathematical optimization, simulation, and decision analysis. It integrates solvers, algorithms, and visualization into a unified environment.

## Installation

```bash
pip install feloopy
```

Add the stock solver stack (Pyomo, HiGHS, SCIP, Uno, OR-Tools, CVXPY,
pyDecision, Mealpy, PyMoo):

```bash
pip install "feloopy[stock]"
```

Everything else FelooPy can drive - commercial bindings, extra modelling
layers, optional data features - is a plain name away, pinned to FelooPy's
preferred (tested) versions:

```bash
flp install gurobi cylp rsome      # resolved names, preferred versions
flp install --latest gurobi        # newest releases instead of the pins
flp deps                           # what is available, and what is installed
flp solvers                        # which interfaces are ready to use
```

### Update notifications

FelooPy checks PyPI for newer releases the same way pip does: in the background,
without ever delaying `import feloopy`, at most **once per machine every 7 days**
(jittered, with exponential backoff when offline, and a cross-process lock, so it
never generates meaningful traffic). When a newer stable release exists you get
one line on stderr, once per version:

```text
FelooPy 0.5.0 is available (you have 0.4.0) -> python -m pip install --upgrade feloopy
```

```bash
feloopy update-check        # check now (respects nothing but the network)
feloopy update-check --offline   # report the cached result only
```

```python
import feloopy
info = feloopy.check_update(force=True, verbose=True)   # never raises
```

Disable the automatic check with `FELOOPY_DISABLE_UPDATE_CHECK=1` (or
`FELOOPY_UPDATE_CHECK=0`); set `FELOOPY_UPDATE_CHECK_INTERVAL_DAYS` to tune the
interval (clamped to >= 1 hour). Only `https://pypi.org/pypi/feloopy/json` is
contacted over TLS, and state is stored as JSON in `~/feloopy/Caches/API/`.

## Quick Start

```python
import feloopy as flp

def example(m):
    x = m.bvar(name="x")
    y = m.pvar(name="y", bound=[0, 1])
    m.con(x + y <= 1, name="c1")
    m.con(x - y >= 1, name="c2")
    m.maximize(x + y)
    return m

flp.search(example).report()
```

## Documentation

Full documentation is available at [feloopy.github.io/docs](https://feloopy.github.io/docs).

## Citation

If you use FelooPy in your work or research, please cite:

**Article**

```bibtex
@article{feloopy2026,
  title   = {FelooPy: The Plug-and-Play OR Workflow You've Been Looking For},
  journal = {OR/MS Tomorrow},
  publisher = {INFORMS},
  url     = {https://www.informs.org/Publications/OR-MS-Tomorrow/FelooPy-The-Plug-and-Play-OR-Workflow-You-ve-Been-Looking-For}
}
```

**Software**

```bibtex
@software{feloopy2026software,
  author       = {Keivan Tafakkori},
  title        = {FelooPy: Efficient and feature-rich integrated decision environment},
  version      = {0.4.0},
  date         = {2026-09-19},
  url          = {https://github.com/feloopy/feloopy},
  license      = {MIT}
}
```

## Contributing

See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License. See [license.txt](license.txt) for details.

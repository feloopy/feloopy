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

  <strong>Efficient & Feature-Rich Integrated Decision Environment</strong>
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

Or the stock solver stack:

```bash
pip install "feloopy[stock]"
```

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

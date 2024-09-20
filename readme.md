<div align="center">
  <p>
    <a href="https://feloopy.github.io" target="_blank">
      <picture>
        <source media="(prefers-color-scheme: light)" srcset="https://github.com/feloopy/feloopy/raw/main/repo/assets/feloopy-logo-name-light.png">
        <source media="(prefers-color-scheme: dark)" srcset="https://github.com/feloopy/feloopy/raw/main/repo/assets/feloopy-logo-name-dark.png">
        <img alt="FelooPy's logo." src="https://github.com/feloopy/feloopy/raw/main/repo/assets/feloopy-logo-name-light.png" width="300" height="auto">
      </picture>
    </a>
  </p>
</div>

<p align="center">
  <strong>Efficient & Feature-Rich Integrated Decision Environment</strong>
</p>


<div align="center" style="margin-bottom: 2px;">

![Version](https://img.shields.io/static/v1?label=Version&message=v0.3.5&color=orange&labelColor=blue)
![Release Date](https://img.shields.io/github/release-date/feloopy/feloopy?label=release&color=orange&labelColor=blue)
[![Downloads](https://static.pepy.tech/personalized-badge/feloopy?period=total&units=international_system&left_color=blue&right_color=orange&left_text=downloads)](https://pepy.tech/project/feloopy)
![License](https://img.shields.io/static/v1?label=license&message=MIT&color=orange&labelColor=blue)

</div>

<br>

### Quick Intro

FelooPy (/fɛlupaɪ/) is a user-friendly tool for coding, modeling, and solving decision problems, helping you focus on analysis. FelooPy offers and supports a wide range of mathematical and statistical models and algorithms for decision-making.

<br>

### Quick Features

- **Linear & Non-Linear Programming**: Exact algorithms for LP/NLP.
- **Integer & Mixed-Integer Programming**: Exact algorithms for IP/MIP.
- **General Purpose Programming**: Heuristic algorithms for various problems.
- **Constraint Programming**: Techniques for constraint satisfaction.
- **Multi-Objective Decision-Making**: Optimizing multiple objectives (MODM/MCDM).
- **Multi-Attribute Decision-Making**: Evaluating opinions on alternatives using multiple attributes (MADM/MCDM).

<br>

### Quick Installation

You can install `feloopy` inside a Python>=3.10.x virtual environment:

```bash
pip install -U "feloopy[stock]==0.3.5"
```

For supporting the developer, testing the latest version, and reporting bugs or contributing to the codebase, use:

```bash
pip install "feloopy[stock] @ git+https://github.com/feloopy/feloopy.git"
```

<br>

### Quick Testing

Here is an example to test FelooPy's functionality:

```python
import feloopy as flp

def example(m):
    x = m.bvar(name="x")
    y = m.pvar(name="y", bound=[0, 1])
    m.con(x + y <= 1, name="c1")
    m.con(x - y >= 1, name="c2")
    m.obj(x + y)
    return m

flp.search(example,directions=["max"]).clean_report()
```

<br>


### Quick Citation

To cite or give credit to FelooPy in publications, projects, presentations, web pages, blog posts, etc. please use one of the following entries, based on the used version:

<br>

#### APA 6th/7th Edition

Tafakkori, K. (2024). *Efficient and feature-rich integrated decision environment* [Python Library]. Retrieved from [https://github.com/feloopy/feloopy](https://github.com/feloopy/feloopy) (Original work published April 2024).

#### LaTeX/BiBTeX

```
@software{ktafakkori2024Apr,
author       = {Keivan Tafakkori},
title        = {{FelooPy: Efficient and feature-rich integrated decision environment}},
year         = {2024},
month        = apr,
publisher    = {GitHub},
url          = {https://github.com/feloopy/feloopy/}
}
```

<br>
<br>

<details>
<summary>Previous citations</summary>



#### Versions before 0.3.5

##### APA 6th/7th Edition

Tafakkori, K. (2022). FelooPy: An integrated optimization environment for AutoOR in Python [Python Library]. Retrieved from https://github.com/ktafakkori/feloopy (Original work published September 2022).

##### LaTeX/BiBTeX
```
@software{ktafakkori2022Sep,
author       = {Keivan Tafakkori},
title        = { {FelooPy: An integrated optimization environment for AutoOR in Python} },
year         = {2022},
month        = sep,
publisher    = {GitHub},
url          = {https://github.com/ktafakkori/feloopy/}
}
```


</details>
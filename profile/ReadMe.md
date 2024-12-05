
Profiling of datasets uses interpreter 3.12 from
[profile/.venv/](https://github.com/jhanley634/ml-2025/tree/main/profile/.venv).

The [ydata](https://pypi.org/project/ydata-profiling)
profiling package is not yet compatible with cPython interpreter 3.13.
This is being tracked in
[bug 1675](https://github.com/ydataai/ydata-profiling/issues/1675).
Till it is closed, this project will continue to do profiling
using the older interpreter in a separate virtual environment.

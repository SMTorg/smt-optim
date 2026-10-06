"""
Load and validate optimization benchmark results.

The results of a benchmark are read from a directory tree with one ``doe.csv`` file per optimization run::

    root_dir/
        algorithm_0/
            problem_0/
                instance_0/doe.csv
                instance_1/doe.csv
            problem_1/
                instance_0/doe.csv
        algorithm_1/
            ...

:func:`load_data` returns the results as a nested dictionary together with a :class:`ValidationReport`
summarizing data-quality issues. The dictionary can be passed to the functions of
:mod:`smt_optim.postprocessing.profiles`.
"""

import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# columns needed to load a file; the profiles additionally require "fidelity"
REQUIRED_COLUMNS = {"iter", "budget", "f0", "rscv"}


@dataclass
class ValidationReport:
    """
    Data-quality issues found while loading optimization results.

    Attributes
    ----------
    missing_columns : dict
        ``(algo, problem, instance) -> set of str``. Required columns absent from the file. These files are
        excluded from the loaded data.
    missing_combinations : dict
        ``algo -> set of (problem, instance)``. Combinations covered by other algorithms but not by `algo`.
    inconsistent_start : dict
        ``(problem, instance) -> {algo: (f0, rscv)}``. Combinations for which the initial point (``iter == 0``)
        differs between algorithms. Both files are kept in the loaded data.

    Notes
    -----
    The report is valid (`is_valid`) when all three dictionaries are empty.
    """

    missing_columns: dict = field(default_factory=dict)       # (algo, problem, instance) -> missing column names
    missing_combinations: dict = field(default_factory=dict)  # algo -> {(problem, instance), ...} not covered by it
    inconsistent_start: dict = field(default_factory=dict)    # (problem, instance) -> {algo: (f0, rscv)}

    @property
    def is_valid(self) -> bool:
        """bool: True if no issue was found."""
        return not (self.missing_columns or self.missing_combinations or self.inconsistent_start)

    def summary(self) -> str:
        """
        Human-readable description of the issues.

        Returns
        -------
        str
            One section per kind of issue, or ``"No issues found."``.
        """
        lines = []

        if self.missing_columns:
            lines.append("Files missing required columns:")
            for (algo, problem, instance), cols in self.missing_columns.items():
                lines.append(f"  - {algo}/{problem}/{instance}: missing {sorted(cols)}")

        if self.missing_combinations:
            lines.append("Algorithms not benchmarked on all problem/instance combinations:")
            for algo, combos in self.missing_combinations.items():
                for problem, instance in sorted(combos):
                    lines.append(f"  - {algo}: missing {problem}/{instance}")

        if self.inconsistent_start:
            lines.append("Inconsistent initial point (f0/rscv at iter 0) across algorithms:")
            for (problem, instance), algo_values in self.inconsistent_start.items():
                lines.append(f"  - {problem}/{instance}:")
                for algo, (f0, rscv) in algo_values.items():
                    lines.append(f"      {algo}: f0={f0}, rscv={rscv}")

        return "\n".join(lines) if lines else "No issues found."

    def warn_if_invalid(self) -> None:
        """Emit a ``UserWarning`` describing the issues, if any."""
        if not self.is_valid:
            warnings.warn(
                "Data validation issues found while loading optimization results:\n" + self.summary()
            )

    def raise_if_invalid(self) -> None:
        """
        Raise an error describing the issues, if any.

        Raises
        ------
        ValueError
            If the report is not valid.
        """
        if not self.is_valid:
            raise ValueError(
                "Data validation issues found while loading optimization results:\n" + self.summary()
            )


def _check_initial_points(initial_points: dict, atol: float) -> dict:
    """
    Flag (problem, instance) combinations whose initial point differs between algorithms.

    Parameters
    ----------
    initial_points : dict
        ``(problem, instance) -> {algo: (f0, rscv)}`` gathered at ``iter == 0``.
    atol : float
        Absolute tolerance used to compare ``f0`` and ``rscv``.

    Returns
    -------
    dict
        The entries of `initial_points` that are inconsistent.
    """

    inconsistent = {}
    for (problem, instance), algo_values in initial_points.items():
        if len(algo_values) < 2:
            continue

        f0_values = np.array([f0 for f0, _ in algo_values.values()])
        rscv_values = np.array([rscv for _, rscv in algo_values.values()])

        if not np.allclose(f0_values, f0_values[0], atol=atol) or not np.allclose(
            rscv_values, rscv_values[0], atol=atol
        ):
            inconsistent[(problem, instance)] = algo_values

    return inconsistent


def _check_coverage(data: dict) -> dict:
    """
    Flag (problem, instance) combinations missing for an algorithm.

    The reference is the union of the combinations found for all algorithms.

    Parameters
    ----------
    data : dict
        Nested dictionary ``data[algo][problem][instance]``.

    Returns
    -------
    dict
        ``algo -> set of (problem, instance)`` for the algorithms with missing combinations.
    """

    all_combinations = {
        (problem, instance)
        for algo_data in data.values()
        for problem, p_data in algo_data.items()
        for instance in p_data
    }

    missing_combinations = {}
    for algo, algo_data in data.items():
        algo_combinations = {(problem, instance) for problem, p_data in algo_data.items() for instance in p_data}
        missing = all_combinations - algo_combinations
        if missing:
            missing_combinations[algo] = missing

    return missing_combinations


def load_data(root_dir, delimiter=",", atol=1e-8, strict=False):
    """
    Load optimization results and validate them.

    Every ``doe.csv`` found under ``root_dir/<algorithm>/<problem>/<instance>/`` is read into a
    :class:`pandas.DataFrame`. Directories whose name starts with ``"_"`` at the algorithm level are ignored.

    Parameters
    ----------
    root_dir : str or os.PathLike
        Root of the results tree.
    delimiter : str, default=","
        Field delimiter of the ``doe.csv`` files.
    atol : float, default=1e-8
        Absolute tolerance used to compare the initial points of different algorithms.
    strict : bool, default=False
        If True, raise an error as soon as an issue is found instead of emitting a warning.

    Returns
    -------
    data : dict
        ``data[algorithm][problem][instance] -> pandas.DataFrame``.
    report : ValidationReport
        Data-quality issues found while loading.

    Raises
    ------
    FileNotFoundError
        If `root_dir` does not exist.
    ValueError
        If `strict` is True and the data is not valid.

    Warns
    -----
    UserWarning
        If `strict` is False and the data is not valid.

    See Also
    --------
    ValidationReport

    Notes
    -----
    The following checks are performed:

    - every file contains the required columns (``iter``, ``budget``, ``f0`` and ``rscv``). Files with missing
      columns are excluded from `data` and listed in ``report.missing_columns``.
    - all algorithms were benchmarked on the same (problem, instance) combinations
      (``report.missing_combinations``).
    - for a given (problem, instance), all algorithms start from the same initial point: ``f0`` and ``rscv``
      at ``iter == 0`` must match within `atol` (``report.inconsistent_start``).

    The last two issues are reported but the corresponding data is kept.

    The functions of :mod:`smt_optim.postprocessing.profiles` additionally require a ``fidelity`` column.

    Examples
    --------
    >>> data, report = load_data("data/unconstrained")
    >>> print(report.summary())
    No issues found.
    >>> data["expected_improvement"]["Sphere_2d"]["instance_0"].head()
    """

    data = {}
    report = ValidationReport()

    # (problem, instance) -> {algo: (f0, rscv)}, gathered from each file's iter == 0 row
    initial_points = {}

    for algorithm in sorted(os.listdir(root_dir)):
        algo_path = os.path.join(root_dir, algorithm)
        if not os.path.isdir(algo_path) or algorithm.startswith("_"):
            continue

        data[algorithm] = {}

        for problem in sorted(os.listdir(algo_path)):
            problem_path = os.path.join(algo_path, problem)
            if not os.path.isdir(problem_path):
                continue

            data[algorithm][problem] = {}

            for instance in sorted(os.listdir(problem_path)):
                instance_path = os.path.join(problem_path, instance)
                doe_path = os.path.join(instance_path, "doe.csv")
                if not os.path.isfile(doe_path):
                    continue

                instance_df = pd.read_csv(doe_path, delimiter=delimiter)

                missing_cols = REQUIRED_COLUMNS - set(instance_df.columns)
                if missing_cols:
                    report.missing_columns[(algorithm, problem, instance)] = missing_cols
                    continue  # data is unreliable without the required columns

                data[algorithm][problem][instance] = instance_df

                init_row = instance_df.loc[instance_df["iter"] == 0]
                if not init_row.empty:
                    initial_points.setdefault((problem, instance), {})[algorithm] = (
                        init_row["f0"].iloc[0],
                        init_row["rscv"].iloc[0],
                    )

            if not data[algorithm][problem]:
                del data[algorithm][problem]

    report.missing_combinations = _check_coverage(data)
    report.inconsistent_start = _check_initial_points(initial_points, atol)

    if strict:
        report.raise_if_invalid()
    else:
        report.warn_if_invalid()

    return data, report


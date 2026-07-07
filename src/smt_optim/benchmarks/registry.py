import inspect
import warnings
import copy

from smt_optim.benchmarks.base import BenchmarkProblem

from .misc import original
from .misc import gano

# from .misc import avt
# from .misc import modified_avt
# from .misc import edge_cases
from .sfu import many_local_minima, bowl_shaped
from .gotestproblems import constrained
from .avt311 import avt311
from .misc import mixvar_branin

from .misc import mf_colville
from .misc import mf_borehole
from .misc import misc2
from .misc import weldedbeam_variants

from .misc import zdt


available = {}


def _register_from_module(module):
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, BenchmarkProblem) and obj is not BenchmarkProblem:
            available[obj.__name__] = obj()


_register_from_module(original)
_register_from_module(gano)
# _register_from_module(avt)
# _register_from_module(modified_avt)
# _register_from_module(edge_cases)

_register_from_module(many_local_minima)
_register_from_module(bowl_shaped)
_register_from_module(constrained)
_register_from_module(avt311)
_register_from_module(mixvar_branin)

_register_from_module(mf_colville)
_register_from_module(mf_borehole)
_register_from_module(misc2)
_register_from_module(weldedbeam_variants)

_register_from_module(zdt)



# def list_problems(**criteria):
#
#     results = []
#
#     for prob in available.values():
#         if all(getattr(prob, k) == v for k, v in criteria.items()):
#             results.append(prob)
#
#     return results


def list_problems(
    n: list[int] = None,
    num_obj: list[int] = [1, 1],
    num_dim: list[int] = None,
    num_cstr: list[int] = None,
    num_fidelity: list[int] = None,
    tags: list[str] = None,
) -> list[BenchmarkProblem]:
    """
    Retrieves all benchmark problems matching the specified filtering criteria.

    Parameters
    ----------
    n : list[int], optional
        Deprecated alias for ``num_dim``.
        A two-element list ``[min_num_dim, max_num_dim]`` specifying
        the inclusive range for the number of design variables.

        .. warning::
            ``n`` is deprecated and will be removed in a future release.
            Use ``num_dim`` instead.

    num_obj : list[int], optional
        A two-element list ``[min_num_obj, max_num_obj]`` specifying
        the inclusive range for the number of objectives.
        If ``None``, no filtering is applied on the number of objectives.

    num_dim : list[int], optional
        A two-element list ``[min_num_dim, max_num_dim]`` specifying
        the inclusive range for the number of design variables.
        If ``None``, no filtering is applied on the number of dimensions.

    num_cstr : list[int], optional
        A two-element list ``[min_num_cstr, max_num_cstr]`` specifying
        the inclusive range for the number of constraints.
        If ``None``, no filtering is applied on the number of constraints.

    num_fidelity : list[int], optional
        A two-element list ``[min_num_fidelity, max_num_fidelity]``
        specifying the inclusive range for the number of fidelity levels.
        If ``None``, no filtering is applied on the number of fidelities.

    tags : list[str], optional
        A list of tags used to filter benchmark problems.
        A problem is returned if it contains all specified tags.
        If ``None``, no tag filtering is applied.

    Returns
    -------
    list[BenchmarkProblem]
        A list of benchmark problem instances matching the specified
        filtering criteria.

    Examples
    --------
    Retrieve all single-objective, mono-fidelity problems, with no constraints:

    >>> problems = list_problems(num_obj=[1, 1], num_cstr=[0, 0], num_fidelity=[1, 1])

    Retrieve all single-objective, multi-fidelity problems:

    >>> problems = list_problems(num_fidelity=[2, np.inf])

    Retrieve all dimension variable benchmark problems:

    >>> problems = list_problems(tags=["n_variable"])

    """

    results = []

    if n is not None:
        warnings.deprecated(
            "`n` is deprecated and will be removed in a future release. Use `num_dim` instead."
        )
        num_dim = n

    for prob in available.values():
        try:
            if num_dim is not None:
                if prob.num_dim < num_dim[0] or prob.num_dim > num_dim[1]:
                    continue
            if num_cstr is not None:
                if prob.num_cstr < num_cstr[0] or prob.num_cstr > num_cstr[1]:
                    continue
            if num_obj is not None:
                if prob.num_obj < num_obj[0] or prob.num_obj > num_obj[1]:
                    continue
            if num_fidelity is not None:
                if (
                    prob.num_fidelity < num_fidelity[0]
                    or prob.num_fidelity > num_fidelity[1]
                ):
                    continue

            if tags is not None:
                if not set(tags).issubset(set(prob.tags)):
                    continue

            results.append(copy.deepcopy(prob))

        except AttributeError:
            continue

    return results


def get_problem(name: str) -> BenchmarkProblem:
    """
    Retrieves a single BenchmarkProblem object by its unique name.

    Parameters
    ----------
    name : str
        The name of the problem to retrieve.
    Returns
    -------
    result : BenchmarkProblem or None
        The retrieved BenchmarkProblem object, or None if no matching problem is found.
    """
    problem = available.get(name)
    return copy.deepcopy(problem) if problem is not None else None

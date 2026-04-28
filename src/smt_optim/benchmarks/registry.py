import inspect
from smt_optim.benchmarks.base import BenchmarkProblem

from .misc import original
from .misc import gano
from .misc import avt
from .misc import modified_avt
from .misc import edge_cases
from .sfu import many_local_minima
from .avt311 import avt311
from .misc import mixvar_branin

available = {}

def _register_from_module(module):
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, BenchmarkProblem) and obj is not BenchmarkProblem:
            available[obj.__name__] = obj()

_register_from_module(original)
_register_from_module(gano)
_register_from_module(avt)
_register_from_module(modified_avt)
_register_from_module(edge_cases)

_register_from_module(many_local_minima)
_register_from_module(avt311)
_register_from_module(mixvar_branin)


# def list_problems(**criteria):
#
#     results = []
#
#     for prob in available.values():
#         if all(getattr(prob, k) == v for k, v in criteria.items()):
#             results.append(prob)
#
#     return results

def list_problems(n: list[int] = None, tags: list[str] = None) -> list[BenchmarkProblem]:
    """
    Retrieves a list of BenchmarkProblem objects that match the specified problem features.

    Parameters
    ----------
    n : Optional[list[int]]
        A list containing minimum and maximum problem dimensions (inclusive).
        If `None`, no dimension filtering is applied.
    tags : Optional[list[str]]
        A list of problem tags to filter by. If `None`, no tag filtering is applied.

    Returns
    -------
    results : list[BenchmarkProblem]
        A list of BenchmarkProblem objects that match the specified features.
    """

    results = []

    for prob in available.values():
        try:
            if n is not None:
                if prob.num_dim < n[0] or prob.num_dim > n[1]:
                    continue
            if tags is not None:
                if not set(tags).issubset(set(prob.tags)):
                    continue

            results.append(prob)

        except:
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
    return available.get(name)
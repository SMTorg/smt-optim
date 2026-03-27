import inspect
from smt_optim.benchmarks.base import BenchmarkProblem

from .misc import original
from .misc import gano
from .misc import avt
from .misc import modified_avt
from .misc import edge_cases
from .sfu import many_local_minima
from .avt311 import avt311

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


def get_problem(name):
    return available.get(name)
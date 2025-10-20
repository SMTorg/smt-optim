import inspect
from .base import BenchmarkProblem

from . import original
from . import gano
from . import avt
from . import modified_avt

available = {}

def _register_from_module(module):
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, BenchmarkProblem) and obj is not BenchmarkProblem:
            available[obj.__name__] = obj()

_register_from_module(original)
_register_from_module(gano)
_register_from_module(avt)
_register_from_module(modified_avt)

def list_problems(**criteria):

    results = []

    for prob in available.values():
        if all(getattr(prob, k) == v for k, v in criteria.items()):
            results.append(prob)

    return results


def get_problem(name):
    return available.get(name)
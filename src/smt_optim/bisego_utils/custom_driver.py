from smt_optim.core import Driver
from smt_optim.utils.stop_criteria import check_stop_criteria
from functools import partial

class CustomStopDriver(Driver):
    def __init__(self, problem, config, strategy, strategy_kwargs=None, stop_conditions=None):
        """
        :param stop_conditions: List of items. Each item can be:
            1. A simple callable: func(state, config)
            2. A tuple: (func, {"arg_name": value})
        """
        s_kwargs = strategy_kwargs if strategy_kwargs is not None else {}
        super().__init__(problem, config, strategy, strategy_kwargs=s_kwargs)
        
        # Normalize stop conditions into (callable, dict) pairs
        self.stop_conditions = []
        if stop_conditions!=None:
            for item in stop_conditions:
                if isinstance(item, tuple):
                    self.stop_conditions.append(item)
                else:
                    self.stop_conditions.append((item, {}))

    def _should_continue(self) -> bool:
        # Default criteria
        if not check_stop_criteria(self.state, self.config):
            return False
        
        # Custom criteria with extra arguments
        for condition, extra_args in self.stop_conditions:
            # We unpack the dictionary into the function call
            if not condition(self.state, self.config, **extra_args):
                return False
        
        return True

    def optimize(self):
        self.start_optim()
        while self._should_continue():
            self.iteration(self.state)
        return self.state


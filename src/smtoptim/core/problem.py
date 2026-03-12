

class Problem:
    def __init__(self, obj_configs: list, design_space, cstr_configs: list = [], costs: list[float] | None = None) -> None:


        self._validate_cstr(cstr_configs)

        self.num_dim = design_space.shape[0]
        self.num_obj = 0
        self.num_cstr = 0
        self.num_fidelity = len(obj_configs[0].objective) if isinstance(obj_configs[0].objective, list) else 1

        self.design_space = design_space

        if costs is None:
            if self.num_fidelity == 1:
                costs = [1]
            else:
                raise Exception("Costs must be provided for multi-fidelity problems.")

        self.costs = costs

        self.obj_configs = []
        self.obj_funcs = []

        self.cstr_configs = []
        self.cstr_funcs = []

        for idx, config in enumerate(obj_configs):
            self._validate_obj(config)
            self.obj_configs.append(config)
            self.obj_funcs.append(config.objective)
            self.num_obj += 1

        for idx, config in enumerate(cstr_configs):
            self._validate_cstr(config)
            self.cstr_configs.append(config)
            self.cstr_funcs.append(config.constraint)
            self.num_cstr += 1

    def __repr__(self):

        data = f"num_dim =       {self.num_dim}\n"\
               f"num_obj =       {self.num_obj}\n"\
               f"num_cstr =      {self.num_cstr}\n"\
               f"num_fidelity =  {self.num_fidelity}\n"\
               f"design_space =  {self.design_space}\n"

        return data


    def _validate_obj(self, config):
        pass


    def _validate_cstr(self, config):
        pass


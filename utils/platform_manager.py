import inspect
from collections.abc import Sequence

class ModuleManager:
    """Implement a manager class to add the new module properly.
       The module can be added as either class or function type.
    """

    def __init__(self, name=None):
        self._modules_dict = dict()
        self._name = name

    def __len__(self):
        return len(self._modules_dict)

    def __repr__(self):
        name_str = self._name if self._name else self.__class__.__name__
        return "{}:{}".format(name_str, list(self._modules_dict.keys()))

    def __getitem__(self, item):
        if item not in self._modules_dict.keys():
            raise KeyError("{} does not exist in availabe l {}".format(
                item, self))
        return self._modules_dict[item]

    @property
    def modules_dict(self):
        return self._modules_dict

    @property
    def name(self):
        return self._name

    def _add_single_module(self, module):
        """Add a single module into the corresponding manager.
        """

        # Currently only support class or function type
        if not (inspect.isclass(module) or inspect.isfunction(module)):
            raise TypeError(
                "Expect class/function type, but received {}".format(
                    type(module)))

        # Obtain the internal name of the module
        module_name = module.__name__

        # Check whether the module was added already
        if module_name in self._modules_dict.keys():
            raise KeyError("{} exists already!".format(module_name))
        else:
            # Take the internal name of the module as its key
            self._modules_dict[module_name] = module

    def add_module(self, modules):
        """Add module(s) into the corresponding manager.
        """

        # Check whether the type is a sequence
        if isinstance(modules, Sequence):
            for module in modules:
                self._add_single_module(module)
        else:
            module = modules
            self._add_single_module(module)

        return modules


DATASETS = ModuleManager("dataset")
MODELS = ModuleManager("models")
LOSSES = ModuleManager("losses")


    
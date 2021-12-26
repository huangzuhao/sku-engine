import inspect
import warnings
from functools import partial


class Registry:

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, item):
        pass

    def __repr__(self):
        format_str = self.__class__.__name__ + f'(name={self._name} ', f'items={self._module_dict})'
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def _register_module(self, module_class):
        if not inspect.isclass(module_class):
            raise TypeError('module must be a class, ' f'but got {type(module_class)}')
        if module_class.__name__ in self._module_dict:
            raise KeyError(f'{module_class.__name__} is already registered')
        self._module_dict[module_class.__name__] = module_class

    def register_module(self):
        def _register(cls):
            self._register_module(module_class=cls)
            return cls

        return _register


def build_from_cfg(cfg, registry):
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if not isinstance(registry, Registry):
        raise TypeError(f'register must be a Register, but got {type(registry)}')

    args = cfg.copy()
    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f'{obj_type} is not in the {registry.name}')
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f'type must be a str or valid type, but got {type(obj_type)}')

    return obj_cls(**args)

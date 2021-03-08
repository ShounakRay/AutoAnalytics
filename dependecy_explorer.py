# @Author: Shounak Ray <Ray>
# @Date:   08-Mar-2021 14:03:50:501  GMT-0700
# @Email:  rijshouray@gmail.com
# @Filename: dependecy_explorer.py
# @Last modified by:   Ray
# @Last modified time: 08-Mar-2021 16:03:86:864  GMT-0700
# @License: [Private IP]

# NOTE: This is useless now, turns out there a package `pydeps` that does what I'm trying to do!

import os
from types import ModuleType

import h2o

# getattr(h2o, 'estimators')

MAX_DEPTH = 10
depth = 0


def recursive_search(_module, depth=0):
    depth += 1

    if(depth > MAX_DEPTH):
        depth = 0
        return 'MAXIMUM'

    print(_module)
    # These are all the sub modules in the modules
    sub_modules = set([mod for mod in [getattr(_module, sub_mod) for sub_mod in dir(_module)
                                       if '_' not in sub_mod] if isinstance(mod, ModuleType)])

    print(sub_modules)
    if len(sub_modules) == 1:  # You reached the edge of the branch
        return None

    return_modules = {}
    subs = []
    for s_mod in sub_modules:
        subs.append(recursive_search(s_mod))

    return_modules[_module] = subs
    return {_module: return_modules}


_module = h2o.automl
[mod for mod in [getattr(_module, sub_mod) for sub_mod in dir(_module)
                 if '_' not in sub_mod] if isinstance(mod, ModuleType)]

recursive_search(h2o)
# EOF

# EOF

import importlib
import sys

_module = importlib.import_module('pmv_crn.baselines.runner')
if __name__ == '__main__':
    _module.main()
else:
    sys.modules[__name__] = _module

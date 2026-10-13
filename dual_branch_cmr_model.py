import importlib
import sys

_module = importlib.import_module('pmv_crn.models.api')
sys.modules[__name__] = _module

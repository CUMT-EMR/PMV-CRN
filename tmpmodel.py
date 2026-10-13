import importlib
import sys

_module = importlib.import_module('pmv_crn.training.engine')
if __name__ == '__main__':
    args = _module.build_argparser().parse_args()
    if args.auto_search:
        _module.auto_search(args)
    else:
        _module.train(args)
else:
    sys.modules[__name__] = _module

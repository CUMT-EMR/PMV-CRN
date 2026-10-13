import importlib
import sys

_module = importlib.import_module('pmv_crn.training.balanced')
if __name__ == '__main__':
    from pmv_crn.training import engine
    args = engine.build_argparser().parse_args()
    if args.auto_search:
        engine.auto_search(args)
    else:
        engine.train(args)
else:
    sys.modules[__name__] = _module

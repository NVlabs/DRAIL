import inspect

def convert_to_args(use_cls, parser):
    sig = inspect.signature(use_cls)
    for k, p in sig.parameters.items():
        d = p.default
        if isinstance(d, inspect._empty):
            continue

        if d is None:
            continue

        if not (isinstance(d, float) or isinstance(d, int) or isinstance(d, str)):
            continue

        k = k.replace('_', '-')
        print(f"+ '--{k}' with default parameter '{d}'")
        parser.add_argument('--' + k, type=type(d), default=d)

def convert_kwargs(args, use_cls):
    dargs = vars(args)
    sig = inspect.signature(use_cls)
    ret = {}
    for k, p in sig.parameters.items():
        if k in dargs:
            ret[k] = dargs[k]
    return ret





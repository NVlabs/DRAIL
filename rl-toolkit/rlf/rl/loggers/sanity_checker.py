from collections import defaultdict
import torch.nn as nn

def print_tensor(x):
    if len(x.shape) > 1:
        return str(x.view(-1, x.shape[-1])[0, :5])
    else:
        return str(x[:5])


class SanityChecker:
    def __init__(self, should_check, is_verbose, stop_key, stop_iters):
        self.is_verbose = is_verbose
        self.should_check = should_check
        self.log_key_calls = defaultdict(lambda:0)
        self.stop_key = stop_key
        self.stop_iters = stop_iters

    def check(self, log_key, **kwargs):
        if not self.should_check:
            return
        if self.is_verbose:
            print('---')
            for k,v in kwargs.items():
                print(self.get_str(k,v))

        self.log_key_calls[log_key] += 1
        if self.log_key_calls[self.stop_key] >= self.stop_iters:
            raise ValueError('Sanity stopped. Program done.')

    def check_rnd_state(self, key):
        if not self.should_check:
            return
        weight = nn.Linear(3,2).weight
        print(f"{key}:Rnd", weight.view(-1).detach()[0].item())


    def get_str(self, k,v, indent=""):
        s = f"{indent}{k}: "
        if isinstance(v, dict):
            for x,y in v.items():
                s += "\n"
                s += self.get_str(x,y, "   ")
        elif isinstance(v, nn.Module):
            params = list(v.parameters())
            sample_spots = [0, -1, -5, 3]
            for x in sample_spots:
                s += f"\n{indent}   {x}:" + print_tensor(params[x])
        else:
            s += f"{v}"
        return s

sanity_checker = None
def get_sanity_checker():
    global sanity_checker
    assert sanity_checker is not None
    return sanity_checker

def set_sanity_checker(args):
    global sanity_checker
    cmd = args.sanity_cmd
    if len(cmd) == 0:
        cmd = ':'
    stop_key, stop_iters = cmd.split(':')
    if stop_iters == '':
        stop_iters = 1
    else:
        stop_iters = int(stop_iters)

    sanity_checker = SanityChecker(args.sanity, args.sanity_verbose, stop_key,
            stop_iters)

def set_sanity_checker_simple():
    global sanity_checker
    sanity_checker = SanityChecker(True, True, "", 1000000000)


def check(*args, **kwargs):
    get_sanity_checker().check(*args, **kwargs)

def c(v):
    get_sanity_checker().check("tmp", v=v)


def check_rand_state(key=""):
    get_sanity_checker().check_rnd_state(key)

import sys
sys.path.insert(0, './')
import argparse
import yaml
from rlf.exp_mgr.wb_data_mgr import get_run_ids_from_report
import wandb
from rlf.exp_mgr import config_mgr
from rlf.rl.utils import CacheHelper
import rlf.rl.utils as rutils
from rlf.run_settings import RunSettings
from collections import defaultdict
import os.path as osp
import os
import numpy as np


def convert_to_prefix(run_names, info):
    wb_proj_name = config_mgr.get_prop('proj_name')
    wb_entity = config_mgr.get_prop('wb_entity')
    api = wandb.Api()
    prefixes = []
    for section, run_name in run_names:
        run = api.run(f"{wb_entity}/{wb_proj_name}/{run_name}")
        prefix = run.config['prefix']
        prefixes.append((prefix, section, run.config['env_name'],
            {'section': section, 'prefix': prefix, **info}))
    return prefixes


def eval_from_file(plot_cfg_path, load_dir, get_run_settings, args):
    with open(plot_cfg_path) as f:
        eval_settings = yaml.load(f)
        config_mgr.init(eval_settings['config_yaml'])
        eval_key = eval_settings['eval_key']
        scale_factor = eval_settings['scale_factor']
        rename_sections = eval_settings['rename_sections']
        wb_proj_name = config_mgr.get_prop('proj_name')
        wb_entity = config_mgr.get_prop('wb_entity')
        api = wandb.Api()
        all_run_names = []
        for eval_section in eval_settings['eval_sections']:
            report_name = eval_section['report_name']
            eval_sections = eval_section['eval_sections']
            cacher = CacheHelper(report_name, eval_sections)
            if cacher.exists() and not eval_section['force_reload']:
                run_names = cacher.load()
            else:
                run_ids = get_run_ids_from_report(wb_entity, wb_proj_name, report_name, eval_sections,
                        api)
                run_names = convert_to_prefix(run_ids, {'report_name': report_name})
                cacher.save(run_names)
            all_run_names.extend(run_names)

    full_load_name = osp.join(load_dir, 'data/trained_models')
    full_log_name = osp.join(load_dir, 'data/log')
    method_names = defaultdict(list)
    for name, method_name, env_name, info in all_run_names:
        model_dir = osp.join(full_load_name, env_name, name)
        cmd_path = osp.join(full_log_name, env_name, name)

        if not osp.exists(model_dir):
            raise ValueError(f"Model {model_dir} does not exist", info)

        if not osp.exists(cmd_path):
            raise ValueError(f"Model {cmd_path} does not exist")

        model_nums = [int(x.split('_')[1].split('.')[0]) for x in os.listdir(model_dir) if 'model_' in x]
        if len(model_nums) == 0:
            raise ValueError(f"Model {model_dir} is empty", info)

        max_idx = max(model_nums)
        use_model = osp.join(model_dir, f"model_{max_idx}.pt")

        with open(osp.join(cmd_path,'cmd.txt'), 'r') as f:
            cmd = f.read()

        method_names[method_name].append((use_model, cmd, env_name, info))

    env_results = defaultdict(lambda: defaultdict(list))
    NUM_PROCS = 20

    total_count = sum([len(x) for x in method_names.values()])

    done_count = 0
    for method_name, runs in method_names.items():
        for use_model, cmd, env_name, info in runs:
            print(f"({done_count}/{total_count})")
            done_count += 1
            cache_result = CacheHelper(f"result_{method_name}_{use_model.replace('/', '_')}_{args.num_eval}", cmd)
            if cache_result.exists() and not args.override:
                eval_result = cache_result.load()
            else:
                if args.table_only and not args.override:
                    break
                cmd = cmd.split(' ')[2:]
                cmd.append('--no-wb')
                cmd.append('--eval-only')
                cmd.extend(['--cuda', 'False'])
                cmd.extend(['--num-render', '0'])
                cmd.extend(['--eval-num-processes', str(NUM_PROCS)])
                cmd.extend(["--num-eval", f"{args.num_eval // NUM_PROCS}"])
                cmd.extend(["--load-file", use_model])
                run_settings = get_run_settings(cmd)
                run_settings.setup()
                eval_result = run_settings.eval_result
                cache_result.save(eval_result)
            store_num = eval_result[args.get_key]
            env_results[info['report_name']][method_name].append(store_num)
            rutils.pstart_sep()
            print(f"Result for {use_model}: {store_num}")
            rutils.pend_sep()
    print(generate_eval_table(env_results, scale_factor, rename_sections))

def generate_eval_table(env_results, scale_factor, rename_sections):
    methods = None
    disp_str = ""
    disp_str += "\\begin{table}[]\n"
    max_env_size  = max([len(x) for x in env_results])
    for env_name in env_results:
        if methods is None:
            methods = list(env_results[env_name].keys())
            disp_str += "\\begin{tabular}{l|" + str(''.join(['c' for _ in range(len(methods))])) + "}\n"
            disp_str += " & " + (" & ".join(methods)) + "\\\\ \\hline \n"
        disp_str += rename_sections[env_name].ljust(max_env_size, ' ')
        vals = [scale_factor * np.array(env_results[env_name][method_name]) for method_name in methods]
        disp_str += " & "
        def convert_vals(x):
            ret_str = "$%.2f \\pm %.2f$" % (np.mean(x), np.std(x))
            return ret_str.ljust(5 + 5 + 5 + 2, ' ')

        disp_str += " & ".join([convert_vals(x) for x in vals])
        disp_str += " \\\\ \n"
    disp_str += "\\end{tabular}\n"
    disp_str += "\\end{table}"
    return disp_str

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-cfg', type=str, required=True)
    parser.add_argument('--load-dir', type=str, required=True)
    parser.add_argument('--num-eval', type=int, required=True)
    parser.add_argument('--get-key', type=str, required=True)
    parser.add_argument('--override', action='store_true')
    parser.add_argument('--table-only', action='store_true')
    return parser

def full_auto_eval(get_run_settings):
    parser = get_arg_parser()
    args = parser.parse_args()
    eval_from_file(args.eval_cfg, args.load_dir, get_run_settings, args)


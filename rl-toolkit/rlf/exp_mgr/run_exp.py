import argparse
import os
import os.path as osp
import random
import string
import subprocess
import sys
import time
import uuid

import libtmux
from rlf.exp_mgr import config_mgr
from rlf.exp_mgr.wb_data_mgr import get_run_params


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sess-id",
        type=int,
        default=-1,
        help="tmux session id to connect to. If unspec will run in current window",
    )
    parser.add_argument(
        "--sess-name", default=None, type=str, help="tmux session name to connect to"
    )
    parser.add_argument(
        "--cmd", type=str, required=True, help="list of commands to run"
    )
    parser.add_argument("--seed", type=str, default=None)
    parser.add_argument(
        "--run-single",
        action="store_true",
        help="""
            If true, will run all commands in a single pane sequentially.
    """,
    )
    parser.add_argument(
        "--cd",
        default="-1",
        type=str,
        help="""
            String of CUDA_VISIBLE_DEVICES. A value of "-1" will not set
            CUDA_VISIBLE_DEVICES at all.
            """,
    )
    parser.add_argument("--cfg", type=str, default="./config.yaml")
    parser.add_argument(
        "--debug",
        type=int,
        default=None,
        help="""
            Index of the command to run.
            """,
    )
    parser.add_argument(
        "--cmd-format",
        type=str,
        default="reg",
        help="""
            Options are [reg, nodash]
            """,
    )
    # MULTIPROC OPTIONS
    parser.add_argument("--mp-offset", type=int, default=0)
    parser.add_argument("--pt-proc", type=int, default=-1)

    # SLURM OPTIONS
    parser.add_argument("--comment", type=str, default=None)
    parser.add_argument("--overcap", action="store_true")
    parser.add_argument("--slurm-no-batch", action="store_true")
    parser.add_argument(
        "--skip-env",
        action="store_true",
        help="""
            If true, will not export any environment variables from config.yaml
            """,
    )
    parser.add_argument("--skip-add", action="store_true")
    parser.add_argument(
        "--speed",
        action="store_true",
        help="""
            SLURM optimized for maximum CPU usage.
            """,
    )
    parser.add_argument(
        "--st", type=str, default=None, help="Slum parition type [long, short]"
    )
    parser.add_argument(
        "--time",
        type=str,
        default=None,
        help="""
            Slurm time limit. "10:00" is 10 minutes.
            """,
    )
    parser.add_argument(
        "--c",
        type=str,
        default="7",
        help="""
            Number of cpus for SLURM job
            """,
    )
    parser.add_argument(
        "--g",
        type=str,
        default="1",
        help="""
            Number of gpus for SLURM job
            """,
    )
    parser.add_argument(
        "--ntasks",
        type=str,
        default="1",
        help="""
            Number of processes for SLURM job
            """,
    )

    return parser


def add_on_args(spec_args):
    spec_args = ['"' + x + '"' if " " in x else x for x in spec_args]
    return " ".join(spec_args)


def get_cmds(cmd_path, spec_args, args):
    try:
        open_cmd = osp.join(cmd_path + ".cmd")
        print("opening", open_cmd)
        with open(open_cmd) as f:
            cmds = f.readlines()
    except:
        base_cmd = cmd_path.split("/")[-1]
        if len(base_cmd) == 8:
            # This could be a W&B ID.
            run = get_run_params(base_cmd)
            if run is not None:
                # The added part should already be in the command
                args.skip_add = True
                use_args = " ".join(run["args"])
                full_cmd = f"python {run['codePath']} {use_args} "

                for k, v in config_mgr.get_prop("eval_replace").items():
                    full_cmd = transform_k(full_cmd, k + " ", v)

                eval_add = config_mgr.get_prop("eval_add")
                if "%s" in eval_add:
                    ckpt_dir = osp.join(
                        config_mgr.get_prop("base_data_dir"),
                        "checkpoints",
                        run["full_name"],
                    )
                    eval_add = eval_add % ckpt_dir
                full_cmd += eval_add
                cmds = [full_cmd]
                return cmds

        raise ValueError(f"Command at {cmd_path} does not exist")

    # pay attention to comments
    cmds = list(filter(lambda x: not (x.startswith("#") or x == "\n"), cmds))
    cmds = [cmd.rstrip() + " " for cmd in cmds]

    # Check if any commands are references to other commands
    all_ref_cmds = []
    for i, cmd in enumerate(cmds):
        if cmd.startswith("R:"):
            cmd_parts = cmd.split(":")[1].split(" ")
            ref_cmd_loc = cmd_parts[0]
            full_ref_cmd_loc = osp.join(config_mgr.get_prop("cmds_loc"), ref_cmd_loc)
            ref_cmds = get_cmds(
                full_ref_cmd_loc.rstrip(), [*cmd_parts[1:], *spec_args], args
            )
            all_ref_cmds.extend(ref_cmds)
    cmds = list(filter(lambda x: not x.startswith("R:"), cmds))

    cmds = [cmd + add_on_args(spec_args) for cmd in cmds]

    cmds.extend(all_ref_cmds)
    return cmds


def get_tmux_window(sess_name, sess_id):
    server = libtmux.Server()

    if sess_name is None:
        tmp = server.list_sessions()
        sess = server.get_by_id("$%i" % sess_id)
    else:
        sess = server.find_where({"session_name": sess_name})
    if sess is None:
        raise ValueError("invalid session id")

    return sess.new_window(attach=False, window_name="auto_proc")


def transform_k(s, use_split, replace_s):
    prefix_parts = s.split(use_split)

    before_prefix = use_split.join(prefix_parts[:-1])
    prefix = prefix_parts[-1]
    parts = prefix.split(" ")
    prefix = parts[0]
    after_prefix = " ".join(parts[1:])

    return before_prefix + f"{use_split} {replace_s} " + after_prefix


def transform_prefix(s, common_id):
    if "--prefix" in s:
        use_split = "--prefix "
    elif "PREFIX" in s:
        use_split = "PREFIX "
    else:
        return s

    prefix_parts = s.split(use_split)
    before_prefix = use_split.join(prefix_parts[:-1])
    prefix = prefix_parts[-1]
    parts = prefix.split(" ")
    prefix = parts[0]
    after_prefix = " ".join(parts[1:])
    if prefix != "debug":
        return before_prefix + f"{use_split} {prefix}-{common_id} " + after_prefix
    else:
        return s


def execute_command_file(cmd_path, add_args_str, cd, sess_name, sess_id, seed, args):
    cmds = get_cmds(cmd_path, add_args_str, args)

    n_seeds = 1
    if args.cmd_format == "reg":
        cmd_format = "--"
        spacer = " "
    elif args.cmd_format == "nodash":
        cmd_format = ""
        spacer = "="
    else:
        raise ValueError(f"{args.cmd_format} does not match anything")

    if seed is not None and len(seed.split(",")) > 1:
        seeds = seed.split(",")
        common_id = "".join(random.sample(string.ascii_uppercase + string.digits, k=2))

        cmds = [transform_prefix(cmd, common_id) for cmd in cmds]
        cmds = [
            cmd + f" {cmd_format}seed{spacer}{seed}" for cmd in cmds for seed in seeds
        ]
        n_seeds = len(seeds)
    elif seed is not None:
        cmds = [x + f" {cmd_format}seed{spacer}{seed}" for x in cmds]
    add_on = ""

    if (len(cmds) // n_seeds) > 1:
        # Make sure all the commands share the last part of the prefix so they can
        # find each other. The name is long because its really bad if a job
        # finds the wrong other job.
        common_id = "".join(
            random.sample(
                string.ascii_uppercase + string.ascii_lowercase + string.digits, k=6
            )
        )
        cmds = [transform_prefix(cmd, common_id) for cmd in cmds]

    if args.pt_proc != -1:
        pt_dist_str = f"MULTI_PROC_OFFSET={args.mp_offset} python -u -m torch.distributed.launch --use_env --nproc_per_node {args.pt_proc} "

        def make_dist_cmd(x):
            parts = x.split(" ")
            runf = None
            for i, part in enumerate(parts):
                if ".py" in part:
                    runf = i
                    break

            if runf is None:
                raise ValueError("Could not split command")

            rest = " ".join(parts[runf:])
            return pt_dist_str + rest

        cmds[0] = make_dist_cmd(cmds[0])

    if args.debug is not None:
        print("IN DEBUG MODE")
        cmds = [cmds[args.debug]]

    env_vars = " ".join(config_mgr.get_prop("add_env_vars", []))
    if len(env_vars) != 0:
        env_vars += " "

    if sess_id == -1:
        if len(cmds) == 1:
            exec_cmd = cmds[0]
            if cd != "-1":
                exec_cmd = "CUDA_VISIBLE_DEVICES=" + cd + " " + exec_cmd + " " + add_on
            else:
                exec_cmd = exec_cmd + " " + add_on
            if not args.skip_env:
                exec_cmd = env_vars + exec_cmd
            print("executing ", exec_cmd)
            os.system(exec_cmd)
        else:
            raise ValueError("Running multiple jobs. You must specify tmux session id")
    else:

        def as_list(x):
            if isinstance(x, int):
                return [x for _ in cmds]
            x = x.split("|")
            if len(x) == 1:
                x = [x[0] for _ in cmds]
            return x

        cd = as_list(cd)
        ntasks = as_list(args.ntasks)
        g = as_list(args.g)
        c = as_list(args.c)

        DELIM = " ; "

        if args.run_single:
            cmds = DELIM.join(cmds)
            cmds = [cmds]

        for cmd_idx, cmd in enumerate(cmds):
            new_window = get_tmux_window(sess_name, sess_id)
            cmd += " " + add_on
            print("running full command %s\n" % cmd)

            # Send the keys to run the command
            conda_env = config_mgr.get_prop("conda_env")
            if args.st is None:
                if not args.skip_env:
                    tmp_cmds = cmd.split(DELIM)
                    tmp_cmds = [env_vars + x for x in tmp_cmds]
                    cmd = DELIM.join(tmp_cmds)
                last_pane = new_window.attached_pane
                last_pane.send_keys(cmd, enter=False)
                pane = new_window.split_window(attach=False)
                pane.set_height(height=50)
                pane.send_keys("source deactivate")

                pane.send_keys("source activate " + conda_env)
                pane.enter()
                if cd[cmd_idx] != "-1":
                    pane.send_keys("export CUDA_VISIBLE_DEVICES=" + cd[cmd_idx])
                    pane.enter()
                pane.send_keys(cmd)
                pane.enter()
            else:
                # Make command into a SLURM command
                base_data_dir = config_mgr.get_prop("base_data_dir")
                python_path = osp.join(
                    osp.expanduser("~"), "miniconda3", "envs", conda_env, "bin"
                )
                runs_dir = "data/log/runs"
                if not osp.exists(runs_dir):
                    os.makedirs(runs_dir)

                parts = cmd.split(" ")
                prefix = None
                for i, x in enumerate(parts):
                    if x == "--prefix" or x == "PREFIX":
                        prefix = parts[i + 1].replace('"', "")
                        break

                new_log_dir = osp.join(base_data_dir, "log")
                new_vids_dir = osp.join(base_data_dir, "vids")
                new_save_dir = osp.join(base_data_dir, "trained_models")
                ident = str(uuid.uuid4())[:8]
                log_file = osp.join(runs_dir, ident) + ".log"

                last_pane = new_window.attached_pane
                last_pane.send_keys(f"tail -f {log_file}", enter=False)
                pane = new_window.split_window(attach=False)
                pane.set_height(height=10)

                new_dirs = []
                cmd_args = cmd.split(" ")
                if not args.skip_add:
                    for k, v in config_mgr.get_prop("change_cmds").items():
                        if k in cmd_args:
                            continue
                        new_dirs.append(k + " " + osp.join(base_data_dir, v))
                    cmd += " " + (" ".join(new_dirs))

                if not args.slurm_no_batch:
                    run_file, run_name = generate_hab_run_file(
                        log_file,
                        ident,
                        python_path,
                        cmd,
                        prefix,
                        args.st,
                        ntasks[cmd_idx],
                        g[cmd_idx],
                        c[cmd_idx],
                        args.overcap,
                        args,
                    )
                    print(f"Running file at {run_file}")
                    pane.send_keys(f"sbatch {run_file}")
                    time.sleep(2)
                    pane.send_keys(f"scancel {run_name}", enter=False)
                else:
                    srun_settings = (
                        f"--gres=gpu:{args.g} "
                        + f"-p {args.st} "
                        + f"-c {args.c} "
                        + f"-J {prefix}_{ident} "
                        + f"-o {log_file}"
                    )

                    # This assumes the command begins with "python ..."
                    cmd = f"srun {srun_settings} {python_path}/{cmd}"
                    pane.send_keys(cmd)

        print("everything should be running...")


def generate_hab_run_file(
    log_file, ident, python_path, cmd, prefix, st, ntasks, g, c, use_overcap, args
):
    ignore_nodes_s = ",".join(config_mgr.get_prop("slurm_ignore_nodes", []))
    if len(ignore_nodes_s) != 0:
        ignore_nodes_s = "#SBATCH -x " + ignore_nodes_s

    add_options = [ignore_nodes_s]
    if use_overcap:
        add_options.append("#SBATCH --account=overcap")
    if args.time is not None:
        add_options.append(f"#SBATCH --time={args.time}")
    if args.comment is not None:
        add_options.append(f'#SBATCH --comment="{args.comment}"')
    add_options = "\n".join(add_options)

    pre_python_txt = ""
    python_parts = cmd.split("python")
    has_python = False
    if len(python_parts) > 1:
        pre_python_txt = python_parts[0]
        cmd = "python" + python_parts[1]
        has_python = True

    cpu_options = "#SBATCH --cpus-per-task %i" % int(c)
    if args.speed:
        cpu_options = "#SBATCH --overcommit\n"
        cpu_options += "#SBATCH --cpu-freq=performance\n"
        cpu_options += (
            "#SBATCH -c $(((${SLURM_CPUS_PER_TASK} * ${SLURM_TASKS_PER_NODE})))"
        )

    if has_python:
        run_cmd = python_path + "/" + cmd
        requeue_s = "#SBATCH --requeue"
    else:
        run_cmd = cmd
        requeue_s = ""

    fcontents = """#!/bin/bash
#SBATCH --job-name=%s
#SBATCH --output=%s
#SBATCH --gres gpu:%i
%s
#SBATCH --nodes 1
#SBATCH --signal=USR1@600
#SBATCH --ntasks-per-node %i
%s
#SBATCH -p %s
%s

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export MULTI_PROC_OFFSET=%i

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

set -x
srun %s"""
    if prefix is not None:
        job_name = prefix + "_" + ident
    else:
        job_name = ident
    log_file_loc = "/".join(log_file.split("/")[:-1])
    fcontents = fcontents % (
        job_name,
        log_file,
        int(g),
        cpu_options,
        int(ntasks),
        requeue_s,
        st,
        add_options,
        args.mp_offset,
        run_cmd,
    )
    job_file = osp.join(log_file_loc, job_name + ".sh")
    with open(job_file, "w") as f:
        f.write(fcontents)
    return job_file, job_name


def full_execute_command_file():
    parser = get_arg_parser()
    print(os.getcwd())
    args, rest = parser.parse_known_args()
    config_mgr.init(args.cfg)

    cmd_path = osp.join(config_mgr.get_prop("cmds_loc"), args.cmd)
    execute_command_file(
        cmd_path, rest, args.cd, args.sess_name, args.sess_id, args.seed, args
    )


def kill_current_window():
    """
    Kills the current tmux window. Helpful for managing tmux windows. If not
    current attached to a tmux window, does nothing.
    """
    # From https://superuser.com/a/1188041
    run_cmd = """tty=$(tty)
for s in $(tmux list-sessions -F '#{session_name}' 2>/dev/null); do
    tmux list-panes -F '#{pane_tty} #{session_name}' -t "$s"
done | grep "$tty" | awk '{print $2}'"""
    curr_sess_id = subprocess.check_output(run_cmd, shell=True).decode("utf-8").strip()
    print("Got session id", curr_sess_id)
    if curr_sess_id == "":
        return
    curr_sess_id = int(curr_sess_id)

    server = libtmux.Server()
    session = server.get_by_id("$%i" % curr_sess_id)

    session.attached_window.kill_window()


if __name__ == "__main__":
    full_execute_command_file()

import argparse


def str2bool(v):
    return v.lower() == "true"


def get_default_parser():
    parser = argparse.ArgumentParser(description="RL", conflict_handler="resolve")
    add_args(parser)
    return parser


def add_args(parser):
    #############################
    # INTERVALS
    #############################
    parser.add_argument("--sanity", action="store_true", default=False)
    parser.add_argument("--sanity-verbose", action="store_true", default=False)
    parser.add_argument(
        "--sanity-cmd",
        type=str,
        default="",
        help="""
            Example format "after_rollout:5" stops after 5 calls to after_rollout
            """,
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="log interval, one log per n updates (default: 1)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=-1,
        help="save interval, one save per n updates (default: 100)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=-1,
        help="eval interval, one eval per n updates (default: None)",
    )

    #############################
    # RUN CONFIG
    #############################
    parser.add_argument(
        "--env-log-dir",
        default="/tmp/gym",
        help="directory to save agent logs (default: /tmp/gym)",
    )

    parser.add_argument(
        "--log-dir",
        default="./data/log",
        help="directory to save agent logs (default: /tmp/gym)",
    )

    parser.add_argument(
        "--sync",
        action="store_true",
        default=False,
        help="Whether to sync with properties specified in config",
    )

    parser.add_argument(
        "--save-dir",
        default="./data/trained_models/",
        help="directory to save agent trained models (default: ./data/trained_models/)",
    )

    parser.add_argument("--prefix", default="debug", help="Run identifier")

    #############################
    # RL LOOP
    #############################
    parser.add_argument(
        "--num-env-steps",
        type=float,
        default=5e6,
        help="number of environment steps to train (default: 1e8)",
    )

    parser.add_argument(
        "--num-processes",
        type=int,
        default=32,
        help="how many training CPU processes to use (default: 32)",
    )

    
    parser.add_argument(
        "--env-name",
        help="environment to train on (default: PongNoFrameskip-v4)",
        required=True,
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor for rewards (default: 0.99)",
    )

    parser.add_argument(
        "--use-proper-time-limits",
        action="store_true",
        default=False,
        help="compute returns taking into account time limits",
    )

    parser.add_argument(
        "--num-steps",
        type=int,
        default=128,
        help="number of forward steps in A2C/PPO (old default: 128)",
    )

    parser.add_argument(
        "--seed", type=int, default=31, help="random seed (default: 31)"
    )
    parser.add_argument(
        "--detect-nan",
        type=str2bool,
        default=False,
        help="""
            If true, will use `torch.autograd.set_detect_anomaly`.
            """,
    )

    #############################
    # EVAL / RETRAIN
    #############################
    parser.add_argument(
        "--eval-only",
        action="store_true",
        default=False,
        help="""
            Enables rendering text to the observation from `mod_render_frames`
            of the env interface or the alg_env_settings.
            """,
    )
    parser.add_argument("--render-metric", action="store_true", default=False)
    parser.add_argument(
        "--render-succ-fails",
        action="store_true",
        default=False,
        help="""
            Renders successes and failures separately. When true,
            eval-num-processes must be 1.
            """,
    )
    parser.add_argument("--num-eval", type=int, default=5)
    parser.add_argument(
        "--log-smooth-len",
        type=int,
        default=100,
        help="Smoothing window for all logged statistics",
    )
    parser.add_argument(
        "--num-render", type=int, default=0, help="None places no limit"
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="Resume training"
    )

    parser.add_argument(
        "--load-file", default="", help=".pt weights file for resuming or evaluating"
    )

    parser.add_argument(
        "--eval-num-processes",
        type=int,
        default=None,
        help="# of evaluation processes. When None use the same # as in non-evaluation",
    )
    parser.add_argument("--vid-fps", type=float, default=30.0)
    parser.add_argument("--vid-dir", type=str, default="./data/vids")

    #############################
    # POLICY
    #############################
    parser.add_argument(
        "--cuda", type=str2bool, default=True, help="disables CUDA training"
    )
    parser.add_argument("--gradient-clip", type=str2bool, default=False)
    parser.add_argument("--gradient-clip-period", type=float, default=0.2)

    #############################
    # DISCRIMINATOR
    #############################
    parser.add_argument("--lr-schedule", type=str2bool, default=False)

    #############################
    # IMITATION LEARNING
    #############################
    parser.add_argument(
        "--eval-save",
        action="store_true",
        default=False,
        help="Save the trajectories from the evaluation",
    )
    parser.add_argument("--traj-dir", type=str, default="./data/traj")

    #############################
    # ENV
    #############################
    parser.add_argument("--normalize-env", type=str2bool, default=True)
    parser.add_argument("--clip-actions", type=str2bool, default=False)
    parser.add_argument("--frame-stack", type=str2bool, default=False)
    parser.add_argument("--policy-ob-key", type=str, default="observation")
    parser.add_argument("--force-multi-proc", type=str2bool, default=False)
    parser.add_argument(
        "--context-mode",
        type=str,
        default="fork",
        help="""
            The mode used for multi-processing. Valid options are "fork",
            "spawn", "forkserver". "fork" will not work if you are doing
            rendering in MuJoCo.
            """,
    )
    parser.add_argument("--transpose-frame", type=str2bool, default=True)
    parser.add_argument(
        "--warp-frame",
        type=str2bool,
        default=False,
        help="""
            Resizes to 84x84 and makes grayscale
            """,
    )
    parser.add_argument("--time-limit", type=float, default=None)

    #############################
    # RAY
    #############################
    parser.add_argument("--ray", action="store_true")
    parser.add_argument("--ray-config", type=str, default="{}")
    parser.add_argument("--ray-cpus", type=float, default=1)
    parser.add_argument("--ray-gpus", type=float, default=0)
    parser.add_argument("--ray-nsamples", type=int, default=1)
    parser.add_argument(
        "--ray-debug",
        action="store_true",
        help="""
            Turns on internal non-ray logging. Also enables `local_mode` for
            Ray.
            """,
    )

    #############################
    # MISC
    #############################
    parser.add_argument(
        "--use-gae",
        type=str2bool,
        default=True,
        help="use generalized advantage estimation",
    )

    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="gae lambda parameter (default: 0.95)",
    )

import sys
sys.path.insert(0, './')
from rlf.exp_mgr import config_mgr
import argparse
import os

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./config.yaml')
    parser.add_argument('--sync-dirs', type=str, default=None)
    return parser

if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    config_mgr.init(args.cfg)

    proj_name = config_mgr.get_prop('proj_name')
    sync_host = config_mgr.get_prop('sync_host')
    sync_user = config_mgr.get_prop('sync_user')
    sync_port = config_mgr.get_prop('sync_port')

    cmds = []
    for sync_dir in args.sync_dirs.split(','):
        parent_sync_dir = '/'.join(sync_dir.split('/')[:-1])
        cmds.extend([
                "ssh -i ~/.ssh/id_open_rsa/id -p {} {}@{} 'mkdir -p ~/{}_backup/{}'".format(
                    sync_port, sync_user, sync_host, proj_name, sync_dir),
                'rsync -avuzhr -e "ssh -i ~/.ssh/id_open_rsa/id -p {}" {} {}@{}:~/{}_backup/{}'.format(
                    sync_port, sync_dir, sync_user, sync_host, proj_name,
                    parent_sync_dir),
                ])

    os.system("\n".join(cmds))

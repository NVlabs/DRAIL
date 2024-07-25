import torch
import os.path as osp
import os


class Checkpointer(object):
    def __init__(self, args):
        self.save_state = {}
        self.load_state = {}
        self.save_dir = args.save_dir
        self.prefix = args.prefix
        self.env_name = args.env_name
        self.load_file = args.load_file
        self.is_loaded = False
        self.args = args

        self.model_dir_name = osp.join(self.save_dir, self.env_name, self.prefix)

        if self.load_file != '':
            self.load()

    def load(self):
        self.load_state = torch.load(self.load_file)
        print('-' * 30)
        print('Loaded model from %s' % self.load_file)
        print('-' * 30)
        self.is_loaded = True

    def should_load(self):
        return self.is_loaded

    def should_save(self):
        return self.save_dir != '' and self.args.save_interval != -1

    def save_key(self, key_name, val):
        self.save_state[key_name] = val

    def has_load_key(self, key_name):
        return (key_name in self.load_state)

    def get_key(self, key_name):
        return self.load_state[key_name]

    def get_load_state(self):
        return self.load_state

    def get_save_path(self):
        return self.model_dir_name

    def flush(self, num_updates):
        if not self.should_save():
            return

        if not osp.exists(self.model_dir_name):
            os.makedirs(self.model_dir_name)
        save_path = osp.join(self.model_dir_name, 'model_%i.pt' % num_updates)

        torch.save(self.save_state, save_path)
        print('-' * 30)
        print('Saved model to %s' % save_path)
        print('-' * 30)

        self.save_state = {}

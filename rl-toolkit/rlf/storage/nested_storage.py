from rlf.storage.base_storage import BaseStorage

class NestedStorage(BaseStorage):
    def __init__(self, child_dict, main_key):
        super().__init__()
        self.child_dict = child_dict
        self.main_key = main_key

    def set_traj_done_callback(self, on_traj_done_fn):
        for _,v in self.child_dict.items():
            v.set_traj_done_callback(on_traj_done_fn)

    def init_storage(self, obs):
        for _,v in self.child_dict.items():
            v.init_storage(obs)

    def get_obs(self, step):
        return self.child_dict[self.main_key].get_obs(step)

    def get_hidden_state(self, step):
        return self.child_dict[self.main_key].get_hidden_state(step)

    def get_masks(self, step):
        return self.child_dict[self.main_key].get_masks(step)

    def insert(self, obs, next_obs, reward, done, info, ac_info):
        for _,v in self.child_dict.items():
            v.insert(obs, next_obs, reward, done, info, ac_info)

    def after_update(self):
        for _,v in self.child_dict.items():
            v.after_update()

    def to(self, device):
        for _,v in self.child_dict.items():
            v.to(device)

    def add_info_key(self, key_name, data_size):
        for _,v in self.child_dict.items():
            v.add_info_key(key_name, data_size)

    def get_extract_info_keys(self):
        return self.child_dict[self.main_key].get_extract_info_keys()

    def get_add_info(self, key):
        return self.child_dict[self.main_key].get_add_info(key)

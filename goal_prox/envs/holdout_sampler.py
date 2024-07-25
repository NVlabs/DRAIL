import numpy as np
from collections import defaultdict
from itertools import combinations

class HoldoutSampler:
    def __init__(self, low, high, n_bins, rnd_gen):
        low = np.array(low)
        high = np.array(high)
        step = (high - low) / n_bins
        self.n_bins = n_bins

        all_start_x = np.arange(low[0], high[0], step[0])
        all_start_y = np.arange(low[1], high[1], step[1])
        self.rnd_gen = rnd_gen

        self.regions = []
        for start_x in all_start_x:
            for start_y in all_start_y:
                self.regions.append((
                    (start_x, start_y),
                    (start_x + step[0], start_y + step[1])))

    def sample(self, allowed_frac, rng):
        get_count = int(self.n_bins * allowed_frac)
        if self.rnd_gen:
            allowed = [(x,y)
                    for x in range(self.n_bins)
                    for y in range(self.n_bins)]
            np.random.RandomState(int(10 * allowed_frac)).shuffle(allowed)
            allowed = allowed[:int((self.n_bins ** 2) * allowed_frac)]
        else:
            allowed = [(x,y)
                    for x in range(self.n_bins)
                    for y in range(self.n_bins)
                    if (x % get_count) == (y % get_count)]
        if len(allowed) == 0:
            region = self.regions[0]
        else:
            x_idx, y_idx = allowed[rng.randint(len(allowed))]
            sel_idx = (x_idx * self.n_bins) + y_idx
            region = self.regions[sel_idx]

        sample = rng.uniform(region[0], region[1])
        return sample

class LineHoldoutSampler:
    def __init__(self, low, high):
        n_bins = 4
        low = np.array(low)
        high = np.array(high)
        step = (high - low) / n_bins
        self.n_bins = n_bins

        all_start_y = np.arange(low[1], high[1], step[1])

        self.regions = []
        for start_y in all_start_y:
            self.regions.append((
                (low[0], start_y),
                (high[0], start_y + step[1])))
    def sample(self, allowed_frac, rng):
        # For 8 bins
        #if allowed_frac == 0.25:
        #    allowed = [2, 5]
        #elif allowed_frac == 0.5:
        #    allowed = [0, 2, 4, 6]
        #elif allowed_frac == 0.75:
        #    allowed = [0, 1, 3, 4, 6, 7]

        # For 4 bins
        if allowed_frac == 0.25:
            allowed = [0]
        elif allowed_frac == 0.5:
            allowed = [0, 2]
        elif allowed_frac == 0.75:
            allowed = [0, 1, 2]

        region = self.regions[rng.randint(len(allowed))]
        sample = rng.uniform(region[0], region[1])
        return sample



if __name__ == '__main__':
    # Unit test to make sure this sampler works.
    sampler = HoldoutSampler([0, 0], [4, 4], 4)
    idxs = defaultdict(int)
    state = np.random.RandomState()
    for i in range(1000):
        x, sel_idx = sampler.sample(0.5, state)
        idxs[sel_idx] += 1
    for k, v in idxs.items():
        print(f"{k}: {v}")

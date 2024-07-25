import numpy as np
from itertools import combinations

class OldHoldoutSampler:
    def __init__(self, low, high, n_bins):
        low = np.array(low)
        high = np.array(high)
        step = (high - low) / n_bins
        self.n_bins = n_bins

        all_start_x = np.arange(low[0], high[0], step[0])
        all_start_y = np.arange(low[1], high[1], step[1])

        self.regions = []
        for start_x in all_start_x:
            for start_y in all_start_y:
                self.regions.append((
                    (start_x, start_y),
                    (start_x + step[0], start_y + step[1])))

    def sample(self, allowed_frac, rng):
        get_count = int(self.n_bins * allowed_frac)

        row_selections = list(combinations(range(self.n_bins), get_count))

        x_idx = rng.randint(self.n_bins)
        y_idx = rng.choice(row_selections[x_idx % len(row_selections)])
        sel_idx = (x_idx * self.n_bins) + y_idx
        region = self.regions[sel_idx]

        sample = rng.uniform(region[0], region[1])
        return sample

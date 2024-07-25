# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys
import os
import argparse
import gdown


# These datasets are shared by the authors of https://github.com/clvrai/goal_prox_il
DEMOS = {
    "ant": [("50", "1ST9_V_ddV4mdbhNnidx3r7BNabHki33m")],
    "hand": [("10000", "1NsZ8FrTIyVvxEiAyTRDtHyzcKfCZNlZu")],
    "maze2d": [
        ("25", "1Il1SWb0nX8RT796izf-YkqvYyb3ls8yO"),
        ("50", "1xfrhsFQEY__pCYe-6xYmPSkPdyrw-reD"),
        ("75", "1A4F3eammJaLWiV2HxAh9lKiij4dDgj6h"),
        ("100", "1Eocidtv_BUwmXQlVgF17rkRerxOX-mvM"),
    ],
    "pick": [
        ("partial3", "1xrAw_ic0DOjfBSl6P6btP4oVGsXFmKNB"),
    ],
    "push": [
        ("partial2", "1kV48YTLdYO3SYN8OQk6KNWa12aCjqJcB"),
    ],
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="../expert_datasets")
    args = parser.parse_args()

    DIR = args.dir

    tasks = ["ant", "hand", "maze2d", "pick", "push"]

    os.makedirs(DIR, exist_ok=True)

    for task in tasks:
        for postfix, id in DEMOS[task]:
            url = "https://drive.google.com/uc?id=" + id
            target_path = "%s/%s_%s.pt" % (DIR, task, postfix)
            if os.path.exists(target_path):
                print("%s is already downloaded." % target_path)
            else:
                print("Downloading demo (%s_%s) from %s" % (task, postfix, url))
                gdown.download(url, target_path)

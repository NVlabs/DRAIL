"""
Utilities for manipulating images, rendering images, and rendering videos.
"""
import os
import os.path as osp
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rlf.rl.utils as rutils

try:
    import wandb
except:
    pass


def append_text_to_image(
    image: np.ndarray, lines: List[str], from_bottom: bool = False
) -> np.ndarray:
    """
    Args:
        image: The NxMx3 frame to add the text to.
        lines: The list of strings (new line separated) to add to the image.
    Returns:
        image: (np.array): The modified image with the text appended.
    """
    h, w, c = image.shape
    font_size = 0.5
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    blank_image = np.zeros(image.shape, dtype=np.uint8)

    if from_bottom:
        y = image.shape[0]
    else:
        y = 0
    for line in lines:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        if from_bottom:
            y -= textsize[1] + 10
        else:
            y += textsize[1] + 10
        x = 10
        cv2.putText(
            blank_image,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    final = image + blank_image
    return final


def save_agent_obs(frames, imdim, vid_dir, name):
    use_dir = osp.join(vid_dir, name + "_frames")
    if not osp.exists(use_dir):
        os.makedirs(use_dir)

    if imdim != 1:
        raise ValueError("Only gray scale is supported right now")

    for i in range(frames.shape[0]):
        for frame_j in range(frames.shape[1]):
            fname = f"{i}_{frame_j}.jpg"
            frame = frames[i, frame_j].cpu().numpy()
            cv2.imwrite(osp.join(use_dir, fname), frame)

    print(f"Wrote observation sequence to {use_dir}")


def save_mp4(frames, vid_dir, name, fps=60.0, no_frame_drop=False, should_print=True):
    frames = np.array(frames)
    if len(frames[0].shape) == 4:
        new_frames = frames[0]
        for i in range(len(frames) - 1):
            new_frames = np.concatenate([new_frames, frames[i + 1]])
        frames = new_frames

    if not osp.exists(vid_dir):
        os.makedirs(vid_dir)

    vid_file = osp.join(vid_dir, name + ".mp4")
    if osp.exists(vid_file):
        os.remove(vid_file)

    w, h = frames[0].shape[:-1]
    videodims = (h, w)
    #fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    fourcc = cv2.VideoWriter_fourcc("a", "v", "c", "1")
    video = cv2.VideoWriter(vid_file, fourcc, fps, videodims)
    for frame in frames:
        frame = frame[..., 0:3][..., ::-1]
        video.write(frame)
    video.release()
    if should_print:
        print(f"Rendered to {vid_file}")


def plot_traj_data(
    pred: np.ndarray,
    real: np.ndarray,
    save_path: str,
    full_save_name: str,
    y_axis_name: str,
    step: int,
    no_wb: bool,
    title: str = "",
):
    """
    Plots each state dimension of a trajectory comparing a predicted and real trajectory.
    :param pred: Shape [H, D] for a trajectory of length H and state dimension D.
    D plots will be created.
    :param real: Shape [H, D].
    """

    per_state_mse = np.mean((pred - real) ** 2, axis=0)
    per_state_sqrt_mse = np.sqrt(per_state_mse)

    H, state_dim = real.shape
    for state_i in range(state_dim):
        use_full_save_name = full_save_name % state_i
        use_save_path = save_path % state_i
        plt.plot(np.arange(H), real[:, state_i], label="Real")
        plt.plot(np.arange(H), pred[:, state_i], label="Pred")
        plt.grid(b=True, which="major", color="lightgray", linestyle="--")
        plt.xlabel("t")
        plt.ylabel(y_axis_name % state_i)

        if isinstance(title, list):
            use_title = title[state_i]
        else:
            use_title = title

        if len(use_title) != 0:
            use_title += "\n"
        use_title += "MSE %.4f, SQRT MSE %.4f" % (
            per_state_mse[state_i],
            per_state_sqrt_mse[state_i],
        )
        plt.title(use_title)
        plt.legend()

        rutils.plt_save(use_save_path)
        if not no_wb:
            wandb.log(
                {use_full_save_name: [wandb.Image(use_save_path)]},
                step=step,
            )

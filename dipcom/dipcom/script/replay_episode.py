#!/usr/bin/env python

# The MIT License (MIT)
#
# Copyright (c) 2025 OMRON SINIC X
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Author: Malek Aburub, Cristian C. Beltran-Hernandez

import signal
import sys
from dipcom.common.utils.utils import get_env_action
import hydra
from omegaconf import DictConfig, OmegaConf

from pathlib import Path
import timeit
import numpy as np
import time
import tqdm

import robosuite as suite
from robosuite.utils import transform_utils as T
from dipcom.common.utils.teleop_utils import get_controller_config
from dipcom.common.utils.ft_visualizer import FTVisualizer

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    env.close()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


@hydra.main(config_path="../../configs", config_name="single_wipe.yaml", version_base=None)
def main(cfg: DictConfig):
    # command line parameters
    dataset_dir = Path(cfg.dataset.dataset.dir)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist")

    dataset = LeRobotDataset(
        cfg.dataset.dataset.repo_id,
        root=cfg.dataset.dataset.dir,
        video_backend="pyav",
    )

    robosuite_config = OmegaConf.to_container(cfg.robosuite, resolve=True)
    robosuite_config['controller_configs'] = get_controller_config(cfg, robosuite_config)

    sleep_time = (1.0/cfg.dataset.dataset.fps)
    camera_names = [cam_name for cam_name in cfg.dataset.cameras]
    camera_name = camera_names[0]

    camera_height = cfg.dataset.camera_height
    camera_width = cfg.dataset.camera_width

    np.random.seed(cfg.dataset.dataset.seed)

    # Create environment
    global env
    env = suite.make(
        **robosuite_config,
        has_renderer=True,
        has_offscreen_renderer=True,
        render_camera=camera_name,
        ignore_done=True,
        use_camera_obs=True,
        camera_names=camera_names,
        camera_heights=camera_height,
        camera_widths=camera_width,
    )

    # Setup printing options for numbers
    np.set_printoptions(linewidth=np.inf)
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    # render one time first so lerobot can be imported without breaking
    env.render()

    episode_idx = cfg.dataset.dataset.episode_idx
    if episode_idx > 0:
        for _ in range(episode_idx):
            env.reset()

    ft_visualizer = None
    if cfg.teleop.plot_ft:
        # init f/t visualizer
        ft_visualizer = FTVisualizer(
            maxlen=cfg.teleop.ft_viz_maxlen,
            include_stiffness="stiffness" in cfg.actions,
            arm=cfg.teleop.arm,
            figure_size=(5, 3),
            force_ylim=(-10, 100),
        )
        ft_visualizer.clear()

    ep_start = dataset.episode_data_index["from"][episode_idx]
    ep_end = dataset.episode_data_index["to"][episode_idx]

    obs = env.reset()
    for t in tqdm.tqdm(range(ep_start, ep_end), desc=f"Episode {episode_idx}"):
        start_time = timeit.default_timer()

        action = get_env_action(env, dataset[t], cfg.dataset.replay_actions)

        obs, reward, done, info = env.step(action)
        env.render()

        if ft_visualizer:
            forces = np.concatenate([obs[f'robot{i}_eef_force_torque'] for i in range(len(env.robots))])
            print(f"forces: {forces}")
            ft_visualizer.add_data(t-ep_start, forces=forces)

            if t % 10 == 0:
                ft_visualizer.render_now()

        computation_time = timeit.default_timer() - start_time
        # sleep about the same as the control frequency
        if computation_time < sleep_time:
            time.sleep(sleep_time - computation_time)
    env.close()
    if ft_visualizer:
        ft_visualizer.close()


if __name__ == "__main__":

    main()

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
# Author: Cristian C. Beltran-Hernandez, Malek Aburub

"""Training & evaluating script for Diffusion in robosuite.

Usage:
        python3 imitate_robosuite.py --rollout_dir

        --dataset-dir: Directory to save the data. Default set to /root/osx-ur/act/dataset/robosuite_reach_human.
        --save: Whether you want to save the data or not.
        --ft: Whether to save F/T readings or not.
        --eval: Whether to evaluate the policy or not. This loads the best validation checkpoint.
        --replay: Whether to replay the demo or not.
        --cartesian: Whether to control cartesian pose of joint angles.
"""


from pathlib import Path
import numpy as np
import os
import hydra
from enum import Enum
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf
import timeit
import time

from dipcom.common.datasets.utils import format_observations, dataset_to_policy_features, tensors_to_numpy
from dipcom.common.utils.utils import get_env_action, load_policy
from dipcom.common.utils.teleop_utils import get_controller_config
from dipcom.common.utils.ft_visualizer import FTVisualizer

from dipcom.common.utils import math_utils

import torch
from torchvision import transforms
import robosuite as suite

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
from dipcom.common.policies.types import FeatureType


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_stiffness(stiffness_representation, actions, arm):
    offset = 0 if arm == "right" else int(np.ceil(len(actions)/2))
    if stiffness_representation == 'cholesky':
        # Convert to SPD and return just the first value
        s = math_utils.cholesky_vector_to_spd(actions[offset: offset + 6])
        return s[0, 0]
    else:
        # Return just the first value
        return actions[0]


@hydra.main(config_path="../../configs", config_name="single_wipe.yaml")
def main(cfg: DictConfig):

    set_seed(0)

    policy_config = OmegaConf.to_container(cfg.policy)

    robosuite_config = OmegaConf.to_container(cfg.robosuite)
    robosuite_config['controller_configs'] = get_controller_config(cfg, robosuite_config)

    control_frequency = int(cfg.dataset.dataset.fps)
    sleep_time = (1.0/control_frequency)
    camera_names = [cam_name for cam_name in cfg.dataset.cameras]
    num_cam = len(camera_names)

    camera_name = "closeview"

    camera_height = cfg.dataset.cameras[camera_name].height
    camera_width = cfg.dataset.cameras[camera_name].width

    ckpt_dir = Path(cfg.eval.base.load_ckpt)  # keep it for now

    onscreen_render = True
    policy_config = policy_config
    max_timesteps = cfg.dataset.dataset.episode_len
    include_stiffness = policy_config["include_stiffness"]
    plot_ft = True
    skip_frame = 5

    global env
    env = suite.make(
        **robosuite_config,
        has_renderer=onscreen_render,
        has_offscreen_renderer=True,
        render_camera=camera_name,
        ignore_done=True,
        use_camera_obs=True,
        camera_names=camera_names,
        camera_heights=camera_height,
        camera_widths=camera_width,
        reward_shaping=True,
    )

    # render one time first so lerobot can be imported without breaking the fucking thing
    obs = env.reset()
    env.render()

    policy, base_cfg, features = load_policy(Path(cfg.eval.base.load_ckpt))  # keep features loaded from here for now
    policy.cuda()
    # policy.eval()
    action_keys = [key for key, ft in features.items() if ft.type is FeatureType.ACTION]

    # init f/t visualizer
    ft_visualizer = None
    if plot_ft:
        ft_visualizer = FTVisualizer(
            maxlen=800,  # No maxlen limit for evaluation
            include_stiffness=include_stiffness,
            force_ylim=(-50, 120),
            arm="left",  # Based on the original code
            figure_size=(7, 4)
        )

    # Setup printing options for numbers
    np.set_printoptions(linewidth=np.inf)
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    max_timesteps = int(max_timesteps * 1)
    num_rollouts = 30

    all_ft = []
    # check from here
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        policy.reset()
        obs = env.reset()

        if ft_visualizer:
            ft_visualizer.clear()  # Clear data for new rollout

        image_list = []  # for visualization
        ee_pos = []
        ee_vel = []

        # with torch.inference_mode():
        for t in range(max_timesteps):
            start_time = timeit.default_timer()

            if onscreen_render:
                env.render()

            # FIXME should this be remove?
            # Offset the first reading
            # if t == 0:
            #     init_ft = np.concatenate([obs['robot0_eef_force_torque'],
            #                               obs['robot1_eef_force_torque']])

            # Prepare observations
            states, images = format_observations(env, obs, camera_names)
            # resize_transform = transforms.Resize((92, 92), antialias=True)
            # images = {k: resize_transform(v) for k, v in images.items()}

            observation = {**states, **images}

            policy_obs = {
                k: v.cuda().float().unsqueeze(0) if isinstance(v, torch.Tensor) else v
                for k, v in observation.items()
            }

            action = policy.select_action(policy_obs)

            action_dict = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in action.items()}
            env_action = get_env_action(env, action_dict, action_keys)

            obs, reward, done, info = env.step(env_action)

            ee_pos.append(obs['robot0_eef_pos'])
            ee_vel.append(obs['robot0_eef_vel_lin'])

            if ft_visualizer:
                forces = observation['observation.ft'].detach().cpu().numpy().ravel()
                stiffness = get_stiffness(policy_config['stiffness_representation'], env_action, arm="left") if include_stiffness else None
                ft_visualizer.add_data(t, forces, stiffness)
                if t % skip_frame == 0:
                    ft_visualizer.render_now()
                    ft_visualizer.set_xlim(xmin=-max_timesteps*0.01, xmax=max_timesteps*1.01)

            # for visualization
            # eef_pos_list.append(eef_pos) if action_space == 'cartesian' else qpos_list.append(qpos_numpy)
            # target_qpos_list.append(target_qpos)
            # rewards.append(reward)
            computation_time = timeit.default_timer() - start_time
            # sleep about the same as the control frequency
            if computation_time < sleep_time:
                time.sleep(sleep_time - computation_time)

        if ft_visualizer:
            fig_path = os.path.join(ckpt_dir, f"rollout_{rollout_id}.png")
            ft_visualizer.save(fig_path)
            _, forces_data, _ = ft_visualizer.get_data()
            all_ft.append(forces_data)

        if False:
            plot_traj(eef_position=ee_pos, eef_velocity=ee_vel, save_dir=cfg.paths.ckpt_dir)

        # if save_episode:
        #     save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    env.close()
    if ft_visualizer:
        ft_visualizer.close()

    force_path = os.path.join(ckpt_dir, f"contact_force.npy")
    np.save(force_path, all_ft)


if __name__ == "__main__":
    global env
    env = None
    try:
        main()
    finally:
        if env:
            env.close()

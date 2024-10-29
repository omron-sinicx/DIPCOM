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
import hydra
from omegaconf import DictConfig, OmegaConf

from pathlib import Path
import shutil
import timeit
import numpy as np
import time
import tqdm

import robosuite as suite
from dipcom.common.datasets.utils import format_observations, flatten_np_dict
from dipcom.common.utils.teleop_utils import build_features, get_device, get_teleop_action, get_controller_config
from dipcom.common.utils.ft_visualizer import FTVisualizer

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    env.close()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


@hydra.main(config_path="../../configs", config_name="single_wipe.yaml", version_base=None)
def main(cfg: DictConfig):

    # ----------------- RoboSuite Config ------------------
    robosuite_config = OmegaConf.to_container(cfg.robosuite, resolve=True)
    robosuite_config['controller_configs'] = get_controller_config(cfg, robosuite_config)

    sleep_time = (1.0/cfg.dataset.dataset.fps)
    camera_names = [cam_name for cam_name in cfg.dataset.cameras]
    num_cam = len(camera_names)
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
        render_camera="closeview",
        ignore_done=True,
        use_camera_obs=True,
        camera_names=camera_names,
        camera_heights=camera_height,
        camera_widths=camera_width,
    )
    env.deterministic_reset = False

    # Prepare teleoperation interface
    device = get_device(env, cfg.teleop.device)

    if cfg.teleop.device == 'keyboard':
        env.viewer.add_keypress_callback(device.on_press)

    # ------------------------------------- LeRobot Config -------------------------------------
    features = build_features(cfg=cfg)

    dataset_dir = Path(cfg.dataset.dataset.dir)
    if cfg.dataset.dataset.overwrite or not dataset_dir.exists():
        if dataset_dir.exists() and dataset_dir.is_dir():
            confirm = input("Dataset directory already exists. Do you want to overwrite it? (y/n): ")
            if confirm == "y":
                shutil.rmtree(dataset_dir)
            else:
                print("Exiting...")
                sys.exit(1)

        dataset = LeRobotDataset.create(
            cfg.dataset.dataset.repo_id,
            cfg.dataset.dataset.fps,
            root=cfg.dataset.dataset.dir,
            robot_type=None,
            use_videos=True,
            image_writer_processes=cfg.dataset.image_writer.num_processes,
            image_writer_threads=cfg.dataset.image_writer.threads_per_camera * 2,
            features=features,
        )
    else:
        dataset = LeRobotDataset(
            cfg.dataset.dataset.repo_id,
            root=cfg.dataset.dataset.dir,
        )

        if num_cam > 0:  # check num_cam is done properly
            dataset.start_image_writer(
                num_processes=cfg.dataset.image_writer.num_processes,
                num_threads=cfg.dataset.image_writer.threads_per_camera * 2,
            )

    ft_visualizer = None
    if cfg.teleop.plot_ft:
        # init f/t visualizer
        ft_visualizer = FTVisualizer(
            maxlen=cfg.teleop.ft_viz_maxlen,
            include_stiffness="stiffness" in cfg.dataset.actions,
            arm=cfg.teleop.arm,
            figure_size=(5, 3),
            force_ylim=(-10, 100),
        )
        ft_visualizer.clear()
        ft_visualizer.render_now()

    stop_recording = False
    num_episodes = cfg.dataset.dataset.num_episodes
    episode_idx = dataset.num_episodes

    if episode_idx > 0:
        for _ in range(episode_idx):
            env.reset()

    # demo (episode) loop
    while True:
        if episode_idx >= num_episodes or stop_recording:
            break
        # Reset the environment
        obs = env.reset()

        # Setup rendering
        num_cam = len(env.sim.model.camera_names)
        env.render()

        # Initialize device control
        device.start_control()
        time.sleep(1)
        all_prev_gripper_actions = [
            {
                f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
                for robot_arm in robot.arms
                if robot.gripper[robot_arm].dof > 0
            }
            for robot in env.robots
        ]
        device.active_robot = 1 if cfg.teleop.arm == "right" else 0

        last_stiffness_switch = 0

        stiffness_values = [500, 3000]
        stiff_idx = 1
        base_stiffness = stiffness_values[stiff_idx]

        skip_episode = True

        if ft_visualizer:
            ft_visualizer.clear()  # Clear any previous data

        t = 0
        # action loop
        for t in tqdm.tqdm(range(cfg.dataset.dataset.episode_len), desc=f"Episode {episode_idx}"):
            start_time = timeit.default_timer()

            # Get the newest action
            input_ac_dict = device.input2action(mirror_actions=device.active_robot == 0)

            # If action is none, then this a reset so we should break
            if input_ac_dict is None:
                tqdm.tqdm.write("Reset saving episode")
                env.deterministic_reset = False
                skip_episode = False  # don't skip
                break

            if cfg.teleop.device.name in ["gamepad", "gello", "keyboard"]:
                device_state = input_ac_dict["state"]
                btn_state = device_state.get('buttons_state', False)  # state of X, Y, A, B buttons
                if btn_state:
                    if btn_state[0] and not last_stiffness_switch:  # X button (or 'x' key for gello)
                        arm = env.robots[device.active_robot].arms[0]
                        if env.robots[device.active_robot].part_controllers[arm].compliance_mode == "variable_stiffness":
                            stiff_idx = (stiff_idx + 1) % len(stiffness_values)
                            base_stiffness = stiffness_values[stiff_idx]
                            tqdm.tqdm.write(f"Switching stiffness to {base_stiffness}")
                    if btn_state[2]:  # A button (or 'a' key for gello)
                        env.deterministic_reset = True
                        tqdm.tqdm.write("Reset skip episode")
                        break
                    if btn_state[1]:  # Y button (or 'y' key for gello)
                        tqdm.tqdm.write("Stopping recording")
                        stop_recording = True
                        break

                last_stiffness_switch = btn_state[0]  # X button (or 'x' key for gello)

            action, env_action_dict = get_teleop_action(env, device, input_ac_dict, all_prev_gripper_actions, base_stiffness)

            # Step through the simulation and render
            obs, reward, done, info = env.step(action)
            observation, images = format_observations(env, obs, camera_names)
            # Filter observations by keys in cfg.states
            observation = {k: v for k, v in observation.items() if k in cfg.dataset.states}

            env.render()

            if ft_visualizer:
                forces = np.concatenate([obs[f'robot{i}_eef_force_torque'] for i in range(len(env.robots))])
                ft_visualizer.add_data(t, forces, base_stiffness)
                if (t+1) % 10 == 0:
                    ft_visualizer.render_now()

            action_le = {k: v.astype(np.float32) for k, v in flatten_np_dict(env_action_dict).items() if k in cfg.dataset.actions}

            # ---------------- Lerobot Dataset -------------------
            # here should add everything to lerobot
            frame = {**observation, **action_le, **images}
            dataset.add_frame(frame, task=cfg.dataset.dataset.task)

            computation_time = timeit.default_timer() - start_time
            # sleep about the same as the control frequency
            if computation_time < sleep_time:
                time.sleep(sleep_time - computation_time)

        if not skip_episode:
            episode_idx += 1
            print(f"Saving episode {episode_idx}")
            dataset.save_episode()
            print(f"Saved episode {episode_idx}")
        else:
            dataset.clear_episode_buffer()

    env.close()
    if ft_visualizer:
        ft_visualizer.close()

    print(f'Saved to {cfg.dataset.dataset.dir}')


if __name__ == "__main__":

    main()

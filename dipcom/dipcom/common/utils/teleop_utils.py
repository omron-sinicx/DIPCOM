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

from copy import deepcopy
import copy
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from omegaconf import DictConfig

import robosuite as suite
from robosuite.controllers.parts.generic.joint_pos import JointPositionController
from robosuite.devices.device import get_arm_action_simple
from robosuite.utils import transform_utils as T


def get_action(env, action_dict, control_delta, active_arm, add_noise, base_stiffness):
    """Get robot action based on device input and control parameters.

    Args:
        env: Robot environment
        action_dict: Dictionary containing device actions
        control_delta: Whether to use delta control
        active_arm: Which arm is active (0 to n-1)
        add_noise: Whether to add noise to actions
        base_stiffness: Base stiffness value for impedance control

    Returns:
        Dictionary containing formatted actions for all robots
    """
    # Get device action and generate noise
    noise = _generate_noise(add_noise)

    # Initialize robot states for all robots
    robot_states = _initialize_robot_states(env, control_delta)

    # Update active arm with device input
    action = np.concatenate([action_dict['right'], action_dict['right_gripper']])
    robot_states[active_arm] = _update_robot_state(action, noise, robot_states[active_arm])

    # Generate stiffness matrices
    stiffness_diag = base_stiffness * np.ones(6)

    # Format and return action dictionary
    return _format_action_dict(robot_states, stiffness_diag, env)


def _generate_noise(add_noise):
    """Generate position and orientation noise."""
    if not add_noise:
        return np.zeros(6)

    position_noise = np.random.uniform(low=-0.0005, high=0.0005, size=3)
    ori_noise = np.random.uniform(low=-np.deg2rad(0.05), high=np.deg2rad(0.05), size=3)
    return np.concatenate([position_noise, ori_noise])


def _initialize_robot_states(env, control_delta):
    """Initialize robot end-effector states for all robots."""
    robot_states = []

    if not control_delta:
        for robot in env.robots:
            pose = robot.part_controllers['right'].delta_to_abs_action(np.zeros(6), goal_update_mode=None)
            robot_states.append({
                'pos': pose[:3],
                'rot': T.axis_angle2ortho6(pose[3:]),
                'rot_vec': pose[3:],
                'gripper': 0.0,
            })
    else:
        for _ in env.robots:
            robot_states.append({
                'pos': np.zeros(3),
                'rot': T.mat2ortho6(np.eye(3)),
                'rot_vec': np.zeros(3),
                'gripper': 0.0,
            })

    return robot_states


def _update_robot_state(device_action, noise, robot_state):
    """Update robot state with device input and noise."""
    robot_state['pos'] = device_action[:3] + noise[:3]
    robot_state['rot'] = T.axis_angle2ortho6(device_action[3:6] + noise[3:])
    robot_state['rot_vec'] = device_action[3:6] + noise[3:]
    robot_state['gripper'] = device_action[-1] * 0.01
    return robot_state


def _format_action_dict(robot_states, stiffness_diag, env):
    """Format action dictionary for environment."""
    action_dict = {
        'action.position': [state['pos'] for state in robot_states],
        'action.rotation_ortho6': [state['rot'] for state in robot_states],
        'action.rotation_axis_angle': [state['rot_vec'] for state in robot_states],
        'action.stiffness_diag': [stiffness_diag for _ in robot_states],
    }

    # Handle gripper actions
    action_dict['action.gripper'] = []
    for i, robot in enumerate(env.robots):
        if len(robot.gripper['right'].current_action) > 0:
            action_dict['action.gripper'].append([robot_states[i]['gripper']])
        else:
            action_dict['action.gripper'].append([])

    return action_dict


def get_device(env, args):
    # initialize device
    if args.name == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    elif args.name == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    elif args.name == "gamepad":
        from robosuite.devices import GamePad
        device = GamePad(env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    else:
        raise Exception("Invalid device choice: choose either 'keyboard', 'gamepad', or 'spacemouse'.")

    return device


def get_controller_config(cfg, robosuite_config):
    ctr_config = None
    controller_config = robosuite_config['controller_configs']
    if isinstance(controller_config, str):
        possible_paths = [Path(os.path.dirname(suite.__file__)), Path(cfg.paths.root) / "dipcom/configs"]
        for path in possible_paths:
            controller_config_path = path / controller_config
            if controller_config_path.exists():
                ctr_config = json.load(open(controller_config_path, "r"))
                break
        if not ctr_config:
            raise FileNotFoundError(f"Controller config file not found in any of the possible paths: {possible_paths}")
    else:
        ctr_config = controller_config
    return ctr_config


def process_action_dict(robot, input_ac_dict):
    action_dict = deepcopy(input_ac_dict)
    extra_action_info = {}
    # set arm actions
    for arm in robot.arms:
        controller_input_type = robot.part_controllers[arm].input_type
        controller_type = robot.part_controllers[arm].__class__.__name__

        if controller_input_type == "absolute":
            input_type = "abs"
        elif controller_input_type == "delta":
            input_type = "delta"
        else:
            raise ValueError

        ctrl_type = ""
        if controller_type == "JointPositionController":
            ctrl_type = "joint_"
            extra_action_info["joint"] = input_ac_dict[f"{arm}_joint_{input_type}"]
            extra_action_info["cartesian"] = input_ac_dict.get(f"{arm}_{input_type}", np.zeros(6))

        action_dict[arm] = input_ac_dict[f"{arm}_{ctrl_type}{input_type}"]

    return action_dict, extra_action_info


def get_teleop_action(env, device, input_ac_dict, all_prev_gripper_actions, base_stiffness):
    active_robot = env.robots[device.active_robot]
    arm = active_robot.arms[0]
    device_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
    # action dictionaries for control
    action_dicts = {}
    joint_actions = {}

    # Maintain gripper state for each robot but only update the active robot with action
    for i in range(len(env.robots)):
        if i == device.active_robot:
            action_dicts[i], joint_actions[i] = process_action_dict(active_robot, input_ac_dict)
            if active_robot.part_controllers[arm].compliance_mode == "variable_stiffness":
                act_dict = copy.deepcopy(action_dicts[i])
                act_dict[arm] = np.concatenate([act_dict[arm], base_stiffness * np.ones(active_robot.part_controllers[arm].control_dim//2)])
                device_action[i] = active_robot.create_action_vector(act_dict)
            else:
                device_action[i] = active_robot.create_action_vector(action_dicts[i])
        else:
            ac_dict = get_arm_action_simple(env.robots[i], env.robots[i].arms[0], np.zeros(6))
            ac_dict[arm + '_gripper'] = np.zeros(env.robots[i].gripper[arm].dof)
            action_dicts[i], joint_actions[i] = process_action_dict(env.robots[i], ac_dict)
            device_action[i] = env.robots[i].create_action_vector(action_dicts[i])
            if env.robots[i].part_controllers[arm].compliance_mode == "variable_stiffness":
                device_action[i] = np.concatenate([device_action[i], base_stiffness * np.ones(6)])

    device_action = np.concatenate(device_action)
    for gripper_ac in all_prev_gripper_actions[device.active_robot]:
        all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dicts[device.active_robot][gripper_ac]

    # Create action dictionaries for dataset
    if isinstance(active_robot.part_controllers[active_robot.arms[0]], JointPositionController):
        env_action_dict = {
            "action.position": [action_dicts[device.active_robot][arm][:3]],
            "action.rotation_axis_angle": [action_dicts[device.active_robot][arm][3:]],
            "action.rotation_ortho6": [T.axis_angle2ortho6(action_dicts[device.active_robot][arm][3:])],
            "action.gripper": [action_dicts[device.active_robot][arm + "_gripper"]],
            "action.joint": [joint_actions[device.active_robot]["joint"]],
        }
    else:
        env_action_dict = {
            'action.position': [],
            'action.rotation_ortho6': [],
            'action.rotation_axis_angle': [],
            'action.stiffness_diag': [],
            'action.gripper': [],
            'action.joint': [],
        }
        for i in range(len(env.robots)):
            env_action_dict['action.position'].append(action_dicts[i][arm][:3])
            env_action_dict['action.rotation_ortho6'].append(T.axis_angle2ortho6(action_dicts[i][arm][3:6]))
            env_action_dict['action.rotation_axis_angle'].append(action_dicts[i][arm][3:6])
            env_action_dict['action.stiffness_diag'].append(base_stiffness * np.ones(6))
            env_action_dict['action.gripper'].append(action_dicts[i][arm + "_gripper"])
            env_action_dict['action.joint'].append(np.zeros(6))

    return device_action, env_action_dict


def build_features(cfg: DictConfig) -> dict:

    features = {}

    # Process cameras
    for cam_name in cfg.dataset.cameras:
        cam_info = cfg.dataset.cameras[cam_name]
        key = f"observation.images.{cam_name}"
        features[key] = {
            "dtype": "video",
            "shape": (cam_info.channels, cam_info.height, cam_info.width),  # CHW
            "names": ["channels", "height", "width"],
            "info": None,
        }

    # Process states
    for state_key in cfg.dataset.states:
        shape_list = cfg.dataset.states[state_key]
        features[state_key] = {
            "dtype": "float32",
            "shape": tuple(shape_list),
            "names": None,
            "info": None,
        }

    # Process action
    for action_key in cfg.dataset.actions:
        shape_list = cfg.dataset.actions[action_key]
        features[action_key] = {
            "dtype": "float32",
            "shape": tuple(shape_list),
            "names": None,
            "info": None,
        }

    return features


def get_wanted_features(cfg: DictConfig) -> list:
    features = [f"observation.images.{cam}" for cam in cfg.cameras]

    if "states" in cfg and cfg.states is not None:
        features += list(cfg.states)

    if "actions" in cfg and cfg.actions is not None:
        features += list(cfg.actions)

    return features

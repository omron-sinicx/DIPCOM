
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

from __future__ import annotations
import copy
import numpy as np
import torch
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
import robosuite.utils.transform_utils as T
from dipcom.common.policies.types import FeatureType, PolicyFeature


def format_observations_old(env: ManipulationEnv, obs, camera_names):

    # store the data as dictionaries
    obs_formatted_dict = {}

    # Gripper actions
    obs_formatted_dict['observation.gripper'] = []
    for robot in env.robots:
        if len(robot.gripper.current_action) > 0:
            obs_formatted_dict['observation.gripper'].append(robot.gripper.current_action)  # TODO: Make this part of the env's obs

    # Joint position and Velocity
    gripper1 = env.robots[1].gripper.current_action.tolist()
    obs_formatted_dict['observation.qpos'] = np.concatenate([obs['robot0_joint_pos'], env.robots[0].gripper.current_action, obs['robot1_joint_pos'], gripper1])
    obs_formatted_dict['observation.qvel'] = np.concatenate([obs['robot0_joint_vel'], env.robots[0].gripper.current_action, obs['robot1_joint_vel'], gripper1])

    # Force/Torque
    obs_formatted_dict['observation.ft'] = np.concatenate([obs['robot0_eef_force_torque'], obs['robot1_eef_force_torque']])

    # End-Effector's Cartesian pose and velocity
    obs_formatted_dict['observation.eef.position'] = [obs['robot0_eef_pos'], obs['robot1_eef_pos']]
    obs_formatted_dict['observation.eef.linear_velocity'] = [obs['robot0_eef_vel_lin'], obs['robot1_eef_vel_lin']]
    obs_formatted_dict['observation.eef.angular_velocity'] = [obs['robot0_eef_vel_ang'], obs['robot1_eef_vel_ang']]
    obs_formatted_dict['observation.eef.rotation_ortho6'] = [T.quat2ortho6(obs['robot0_eef_quat']), T.quat2ortho6(obs['robot1_eef_quat'])]

    for k in obs_formatted_dict:
        obs_formatted_dict[k] = np.array(obs_formatted_dict[k]).flatten()

    # Cameras
    for cam_name in camera_names:
        obs_formatted_dict[f'observation.images.{cam_name}'] = obs[f'{cam_name}_image']

    return obs_formatted_dict


def format_observations(env: ManipulationEnv, obs, camera_names):
    obs_formatted_dict = {}

    # Gripper actions
    obs_formatted_dict['observation.gripper'] = []
    for robot in env.robots:
        if len(robot.gripper['right'].current_action) > 0:
            obs_formatted_dict['observation.gripper'].append(robot.gripper['right'].current_action)

    # Joint position and Velocity
    qpos_parts = []
    qvel_parts = []
    for i, robot in enumerate(env.robots):
        qpos_parts.extend([
            obs[f'robot{i}_joint_pos'],
            robot.gripper['right'].current_action
        ])
        qvel_parts.extend([
            obs[f'robot{i}_joint_vel'],
            robot.gripper['right'].current_action
        ])
    obs_formatted_dict['observation.qpos'] = np.concatenate(qpos_parts)
    obs_formatted_dict['observation.qvel'] = np.concatenate(qvel_parts)

    # Force/Torque
    ft_parts = [obs[f'robot{i}_eef_force_torque'] for i in range(len(env.robots))]
    obs_formatted_dict['observation.ft'] = np.concatenate(ft_parts)

    # End-Effector's Cartesian pose and velocity
    obs_formatted_dict['observation.eef.position'] = [
        obs[f'robot{i}_eef_pos'] for i in range(len(env.robots))
    ]
    obs_formatted_dict['observation.eef.linear_velocity'] = [
        obs[f'robot{i}_eef_vel_lin'] for i in range(len(env.robots))
    ]
    obs_formatted_dict['observation.eef.angular_velocity'] = [
        obs[f'robot{i}_eef_vel_ang'] for i in range(len(env.robots))
    ]
    obs_formatted_dict['observation.eef.rotation_ortho6'] = [
        T.quat2ortho6(obs[f'robot{i}_eef_quat']) for i in range(len(env.robots))
    ]

    # Flatten and convert to torch.float32
    for k in obs_formatted_dict:
        # Skip image keys, will handle separately
        if "images" not in k:
            array = np.array(obs_formatted_dict[k]).flatten()
            obs_formatted_dict[k] = torch.tensor(array, dtype=torch.float32)

    images_dict = {}
    # Cameras (keep as-is, or convert explicitly if needed)
    for cam_name in camera_names:
        image = obs[f'{cam_name}_image']
        image = np.transpose(image, (2, 0, 1))
        # Optionally convert images to float32 tensors as well
        # image_tensor = torch.tensor(image, dtype=torch.float32) / 255.0  # normalize if needed
        images_dict[f'observation.images.{cam_name}'] = torch.tensor(image, dtype=torch.uint8)

    return obs_formatted_dict, images_dict


def flatten_dict(d, parent_key="", sep="/"):
    """Flatten a nested dictionary structure by collapsing nested keys into one key with a separator.

    For example:
    ```
    >>> dct = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}`
    >>> print(flatten_dict(dct))
    {"a/b": 1, "a/c/d": 2, "e": 3}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d, sep="/"):
    outdict = {}
    for key, value in d.items():
        parts = key.split(sep)
        d = outdict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return outdict


def dict_np_to_torch(d):
    res = {}
    for k in d:
        res[k] = torch.from_numpy(d[k])
    return res


def flatten_np_dict(dict):
    res = copy.copy(dict)
    for key in dict:
        res[key] = np.concatenate(dict[key]).flatten()
    return res


def dataset_to_policy_features(features: dict[str, dict]) -> dict[str, PolicyFeature]:
    policy_features = {}
    for key, ft in features.items():
        shape = ft["shape"]
        if ft["dtype"] in ["image", "video"]:
            type = FeatureType.VISUAL
            if len(shape) != 3:
                raise ValueError(f"Number of dimensions of {key} != 3 (shape={shape})")

            names = ft["names"]
            # Backward compatibility for "channel" which is an error introduced in LeRobotDataset v2.0 for ported datasets.
            if names[2] in ["channel", "channels"]:  # (h, w, c) -> (c, h, w)
                shape = (shape[2], shape[0], shape[1])
        elif key == "observation.environment_state":
            type = FeatureType.ENV
        elif key.startswith("observation"):
            type = FeatureType.STATE
        elif key.startswith("action"):
            type = FeatureType.ACTION
        elif key.startswith("next.reward"):
            type = FeatureType.REWARD
        else:
            continue

        policy_features[key] = PolicyFeature(
            type=type,
            shape=shape,
        )

    return policy_features


def tensors_to_numpy(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().squeeze(0).cpu().numpy()
    elif isinstance(obj, dict):
        return {k: tensors_to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensors_to_numpy(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(tensors_to_numpy(v) for v in obj)
    else:
        return obj


def split_actions(data):
    out = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            # If 1D and even, split into two
            if v.ndim == 1 and v.shape[0] % 2 == 0:
                mid = v.shape[0] // 2
                out[k] = [v[:mid], v[mid:]]
            else:
                out[k] = [v]
        else:
            out[k] = [np.array(v)]  # Ensure non-array scalars are wrapped
    return out

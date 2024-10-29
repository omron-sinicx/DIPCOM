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

import numpy as np
import torch
import os
import yaml
import datetime
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import Optional, Dict, Any

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from dipcom.common.policies.types import FeatureType

from dipcom.common.datasets.utils import dataset_to_policy_features
from omegaconf import DictConfig, ListConfig, OmegaConf
import hydra

from robosuite.utils import transform_utils as T


def get_policy_config(name, cfg):
    from dipcom.envs.configuration_dipcom import DipcomConfig
    if name == "dipcom":
        return DipcomConfig(**cfg)
    else:
        raise NotImplementedError(f"Uknown Policy Config {name}")


def get_policy(name, cfg, stats):
    if name == "dipcom":
        from dipcom.envs.modeling_dipcom import DipcomPolicy
        return DipcomPolicy(cfg, dataset_stats=stats)
    elif name == "diffusion":
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        return DiffusionPolicy(cfg, dataset_stats=stats)
    else:
        raise NotImplementedError(f"Uknown Policy {name}")


# FIXME make this independent of the config, from now on cfg.action will contain
# every action in the dataset and we are filtering based on wanted features
def get_env_action(env, action_dict, action_keys: list[str]):
    num_robots = len(env.robots)
    num_grippers = sum([robot.gripper[robot.arms[0]].dof for robot in env.robots])
    if num_robots == 1:
        action = []
        for key in action_keys:
            action_data = action_dict[key]
            if action_data.shape == torch.Size([]):
                action_data = action_data.unsqueeze(0)
            action.append(action_data.cpu().numpy())
    else:
        action = []
        for i, robot in enumerate(env.robots):
            for key in action_keys:
                if key == "action.gripper" and robot.gripper[robot.arms[0]].dof == 0:
                    continue
                divider = num_grippers if key == "action.gripper" else num_robots

                action_data = action_dict[key]
                if action_data.shape == torch.Size([]):
                    action_data = action_data.unsqueeze(0)

                action_len = len(action_data)
                from_idx = i * action_len // divider
                to_idx = (i + 1) * action_len // divider
                _action = action_data[from_idx:to_idx].cpu().numpy()
                if key == "action.rotation_ortho6":
                    _action = T.ortho62axisangle(_action)
                action.append(_action)
    return np.concatenate(action)


def save_training_config(
    cfg: DictConfig,
    ckpt_dir: Path,
    model_name: Optional[str] = None,
    additional_info: Optional[Dict[str, Any]] = None,
    exclude_keys: Optional[list] = None
) -> Dict[str, Any]:
    """
    Save a compact training configuration.
    """

    # Ensure checkpoint directory exists
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Get model name
    if model_name is None:
        model_name = getattr(cfg, 'model_name', 'unknown')

    # Get current timestamp
    current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

    # Create base config to save
    config_to_save = {
        'model_name': model_name,
        'training_timestamp': current_timestamp,
        # 'hydra_config_name': getattr(cfg, '_name', 'unknown'),
    }

    # Standard sections to include
    standard_sections = [
        'model',
        'wanted_features',
        'dataset',
        'policy',
        'paths',
        # 'teleop',
        # 'save_episode'
    ]

    # Add standard sections if they exist
    for section in standard_sections:
        if hasattr(cfg, section):
            value = getattr(cfg, section)
            # Handle both OmegaConf objects and regular values
            if OmegaConf.is_config(value):
                config_to_save[section] = OmegaConf.to_container(value, resolve=True, enum_to_str=True)
            else:
                # For simple values (strings, numbers, lists)
                config_to_save[section] = value

    # Add model-specific configurations if they exist
    if hasattr(cfg, 'model_configs') and OmegaConf.is_config(cfg.model_configs):
        model_configs = OmegaConf.to_container(cfg.model_configs, resolve=True, enum_to_str=True)
        current_model_config = model_configs.get(model_name, {})
        if current_model_config:
            config_to_save['model_specific'] = current_model_config

    # Add any additional information
    if additional_info:
        config_to_save.update(additional_info)

    # Remove excluded keys
    if exclude_keys:
        for key in exclude_keys:
            config_to_save.pop(key, None)

    # Save the configuration in compact style
    config_path = ckpt_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_to_save, f, default_flow_style=None)

    return config_to_save


def load_policy(ckpt_dir):

    # Load saved config and auto-detect model type
    config_path = ckpt_dir / "config.yaml"
    cfg = OmegaConf.load(config_path)

    dataset_repo_dir = Path(cfg.dataset.dataset.dir) if cfg.dataset.dataset.dir is not None else None

    # Get dataset metadata
    dataset_meta = LeRobotDatasetMetadata(
        cfg.dataset.dataset.repo_id,
        root=dataset_repo_dir
    )

    all_features = dataset_to_policy_features(dataset_meta.features)
    features = {
        k: v for k, v in all_features.items()
        if any(partial in k for partial in cfg.wanted_features)
    }

    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    diffusion_config = hydra.utils.instantiate(
        cfg.model.diffusion_config,
        output_features=output_features,
        input_features=input_features
    )

    policy = get_policy(cfg.model.name, diffusion_config, stats=dataset_meta.stats)
    policy.to("cuda")

    # Load checkpoint
    ckpt_path = ckpt_dir / "ema_policy_last.ckpt"
    policy.load_state_dict(torch.load(ckpt_path))
    policy.eval()

    return policy, cfg, features

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


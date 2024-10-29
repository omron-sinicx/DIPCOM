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

from math import ceil
import random
from colorama import Fore, Back, Style, init
import torch
from matplotlib import pyplot as plt

from pathlib import Path

from torch.utils.data import DataLoader, Subset
import numpy as np
import os

import h5py


# Initialize colorama
init(autoreset=True)


def compare_values(original, predicted, loss):
    """
    Compares the values of two dictionaries with the same keys and prints
    the original value, predicted value, and the difference between them with colors.
    """
    from termcolor import colored

    for key in original:
        orig_val = original[key]
        pred_val = predicted[key]
        diff = orig_val - pred_val
        # loss = F.mse_loss

        # Print key name
        print(colored(f"Key: {key}", 'green'))
        print(colored(f"Original Value: {orig_val}", 'blue'))
        print(colored(f"Predicted Value: {pred_val}", 'yellow'))
        print(colored(f"Difference: {diff}", 'red'))
        print("-" * 30)
    if loss:
        print(colored(f"loss: {loss}", 'light_cyan'))
    print("*" * 80)


def print_time(label, data):
    print(Fore.YELLOW, label + Style.RESET_ALL,
          f"Time: mean={np.average(data):0.04f} std={np.std(data):0.04f} min={np.min(data):0.04f} max={np.max(data):0.04f}",
          f"Hz: mean={1./np.average(data):0.04f} std={1./np.std(data):0.04f} min={1./np.min(data):0.04f} max={1./np.max(data):0.04f}")


def print_obs(obs, stiffness_representation, orientation_representation, normalized=False, bimanual=False):
    print(Back.LIGHTRED_EX + Fore.BLACK + f"{'' if not normalized else 'Normalized '}Observations [{stiffness_representation}] [{orientation_representation}]" + Style.RESET_ALL)

    def print_data(obs):
        print(Fore.YELLOW, 'position' + Style.RESET_ALL,
              obs[:3])

        print(Fore.YELLOW, 'rotation' + Style.RESET_ALL,
              obs[3:9])

        print(Fore.YELLOW, 'gripper' + Style.RESET_ALL,
              obs[-1])

    if bimanual:
        obs_idx = int(obs.shape[0]/2)
        print(Back.LIGHTBLUE_EX + Fore.BLACK + "A_BOT:" + Style.RESET_ALL)
        print_data(obs[:obs_idx])
        print(Back.LIGHTMAGENTA_EX + Fore.BLACK + "B_BOT:" + Style.RESET_ALL)
        print_data(obs[obs_idx:])
    else:
        print_data(obs)
    print("======================\n")


def print_comparison(label, predicted_value, expected_value):
    print(Fore.CYAN + label + Style.RESET_ALL)
    print(Fore.GREEN + 'Expected: ' + str(expected_value) + Style.RESET_ALL)
    print(Fore.YELLOW + 'Predicted: ' + str(predicted_value) + Style.RESET_ALL)
    print(Fore.RED + 'Difference: ' + str(expected_value - predicted_value) + Style.RESET_ALL)
    print()


def display_images(images, titles=None, cols=3, figsize=(15, 15), figtitle=""):
    """
    Display multiple images in a grid layout.
    Works for both grayscale and RGB images.

    Parameters:
    - images: List of numpy arrays representing images
    - titles: List of titles for each image (optional)
    - cols: Number of columns in the grid
    - figsize: Figure size (width, height) in inches
    """
    rows = (len(images) - 1) // cols + 1
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for i, img in enumerate(images):
        ax = axes[i]

        # Check if the image is grayscale or RGB
        if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
            # Grayscale image
            ax.imshow(img, cmap='gray')
        else:
            # RGB image
            ax.imshow(img)

        ax.axis('off')
        if titles is not None and i < len(titles):
            ax.set_title(titles[i])

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.suptitle(figtitle)
    plt.tight_layout()
    plt.show()


def get_random_batches(dataloader, num_batches):
    dataset = dataloader.dataset
    batch_size = dataloader.batch_size
    dataset_size = len(dataset)

    # Calculate the total number of samples we need
    total_samples = num_batches * batch_size

    # Ensure we don't request more samples than available
    if total_samples > dataset_size:
        raise ValueError(f"Requested {total_samples} samples, but dataset only has {dataset_size}")

    # Get random indices
    indices = random.sample(range(dataset_size), total_samples)

    # Create a Subset of the dataset
    subset = Subset(dataset, indices)

    # Create a new DataLoader for this subset
    subset_loader = DataLoader(subset, batch_size=batch_size, shuffle=False, pin_memory=dataloader.pin_memory, num_workers=dataloader.num_workers)

    return subset_loader


def get_random_batch(dataloader: DataLoader):
    # Get the dataset from the dataloader
    dataset = dataloader.dataset

    # Calculate the number of samples in a batch
    batch_size = dataloader.batch_size

    # Get a random starting index
    start_index = random.randint(0, len(dataset) - batch_size)

    # Create a Subset of the dataset for just this batch
    batch_subset = Subset(dataset, range(start_index, start_index + batch_size))

    # Create a new DataLoader for just this batch
    batch_loader = DataLoader(batch_subset, batch_size=batch_size, shuffle=False, pin_memory=dataloader.pin_memory, num_workers=dataloader.num_workers)

    # Get the single batch from this DataLoader
    return next(iter(batch_loader))


def load_hf_dataset(dataset_dir):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    # from osx_teleoperation.datasets.lerobot_dataset import LeRobotDataset

    dataset_dir = Path(dataset_dir)
    # Let's take one for this example
    repo_id = f"{dataset_dir.parents[0].name}/{dataset_dir.name}"
    root = dataset_dir.parents[1]

    # Set up the dataset.
    return LeRobotDataset(repo_id, root=root)


def load_hf_dataloader(dataset_dir, batch_size=10):
    dataset = load_hf_dataset(dataset_dir)
    dataloader = DataLoader(
        dataset,
        num_workers=1,
        batch_size=batch_size,
        shuffle=False,
    )
    return dataloader


def format_batch(batch, camera_names):
    dataset = {}
    dataset['eef_pos'] = batch['observation.eef_pos'].numpy()
    dataset['eef_pos_ortho6'] = batch['observation.eef_pos_ortho6'].numpy()
    dataset['rotation_ortho6'] = batch['observation.eef_pos.rotation_ortho6'].numpy()
    dataset['rotation_axis_angle'] = batch['observation.eef_pos.rotation_axis_angle'].numpy()
    dataset['ft'] = batch['observation.ft'].numpy()
    dataset['qpos'] = batch['observation.qpos'].numpy()
    dataset['action'] = batch['action'].numpy()
    dataset['action_diag_ortho6'] = batch['action_diag_ortho6'].numpy()
    dataset['action_cholesky_ortho6'] = batch['action_cholesky_ortho6'].numpy()
    dataset['action.rotation_ortho6'] = batch['action.rotation_ortho6'].numpy()
    dataset['action.cholesky_ortho6'] = batch['action_cholesky_ortho6'].numpy()
    dataset['cameras'] = torch.stack([batch[f'observation.images.{cam_name}'] for cam_name in camera_names], dim=-4)

    return dataset


def load_hf(dataset_dir, camera_names, batch_size=10):
    dataset = load_hf_dataset(dataset_dir)

    dataloader = DataLoader(
        dataset,
        num_workers=1,
        batch_size=batch_size,
        shuffle=False,
    )

    batch = next(iter(dataloader))
    # print(type(batch))
    # print(batch.keys())

    dataset = {}
    dataset['eef_pos'] = batch['observation.eef_pos'].numpy()
    dataset['rotation_ortho6'] = batch['observation.eef_pos.rotation_ortho6'].numpy()
    # dataset['eef_pos_ortho6'] = batch['observation.eef_pos_ortho6'].numpy()
    # dataset['rotation_axis_angle'] = batch['observation.eef_pos.rotation_axis_angle'].numpy()
    dataset['ft'] = batch['observation.ft'].numpy()
    dataset['qpos'] = batch['observation.qpos'].numpy()
    dataset['action'] = batch['action'].numpy()
    # dataset['action_diag_ortho6'] = batch['action_diag_ortho6'].numpy()
    # dataset['action_cholesky_ortho6'] = batch['action_cholesky_ortho6'].numpy()
    dataset['cameras'] = torch.stack([batch[f'observation.images.{cam_name}'] for cam_name in camera_names], dim=-4)

    return dataset


def print_action_diag_ortho6(action):
    # Setup printing options for numbers
    np.set_printoptions(linewidth=np.inf)
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
    print("========= Dataset Sample Action =============")
    print(f"{action.shape=}")
    print('stiffness', action[0, :6])
    print('position', action[0, 6:9])
    print('orientation', action[0, 9:15])
    print('gripper', action[0, 15])
    if action.shape[1] > 16:
        idx = ceil(action.shape[1]/2)
        print("B BOT")
        print('stiffness', action[0, idx:idx+6])
        print('position', action[0, idx+6:idx+9])
        print('orientation', action[0, idx+9:idx+15])
        print('gripper', action[0, -1])
    print("=====================\n")


def load_data_bc(dataset_dir, episode_idx, bimanual):
    data = []
    dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
    with h5py.File(dataset_path, 'r+') as root:
        qpos = root['observations']['qpos'][:]
        ft = root['observations']['ft'][:] if 'ft' in root['observations'] else np.zeros_like(12 if bimanual else 6)
        eef_pos = root['observations']['eef_pos'][:]
        action = root['action'][:]

        image_dict = dict()
        for cam_name in root.attrs['camera_names']:
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][:]

        data = {
            'episode': episode_idx,
            'cameras': image_dict,
            'qpos': qpos,
            'eef_pos': eef_pos,
            'ft': ft,
            'action': action,
        }
    return data

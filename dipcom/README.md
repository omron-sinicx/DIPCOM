# DIPCOM: Diffusion Policies for Compliant Manipulation

## Introduction
This repository contains all the necessary scripts to train and evaluate DIPCOM (Diffusion Policies for Compliant Manipulation), a method designed for training policies that facilitate compliant manipulation tasks using diffusion models. 

## Dataset Formats

we use [LeRobot - HuggingFace](https://github.com/huggingface/lerobot) dataset format.

---

## Configuration

Configurations are managed via YAML files using [Hydra](https://hydra.cc/). All config files are stored in the `configs/` directory with the following structure:

| Directory     | Description                                      |
|---------------|--------------------------------------------------|
| `common/`     | Shared configs (policy training, teleop settings)|
| `datasets/`   | Task dataset configurations                      |
| `model/`      | Model architecture parameters                    |
| `robosuite/`  | Robosuite simulation task configs                |

---

## Training

### Quick Start

```bash
python train.py --config-path <path_to_config> --config-name <config_file_name>
```

Trained models are saved to:
- **`output_dir`** — Training outputs and logs
- **`ckpt_dir`** — Model checkpoints

### Policy Parameters

| Parameter               | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `fps`                   | Control frequency. Match this to your rollout frequency.                    |
| `use_ema`               | Enable exponential moving average during training (recommended).            |
| `update_ema_every_n_steps` | How often to update the EMA model.                                       |

### Diffusion Parameters

| Parameter                | Description                                                                                      |
|--------------------------|--------------------------------------------------------------------------------------------------|
| `n_obs_steps`            | Number of observation history steps including present (T₀). Typically set to `2`.               |
| `horizon`                | Total prediction horizon (T), including `n_obs_steps`. Typically `48–60` at 30Hz.               |
| `n_action_steps`         | Number of actions to execute (Tₐ). Recommended: 1 second worth (e.g., `30` at 30Hz).            |
| `temporal_ensemble_coeff`| Action chunking coefficient to reduce shaking. Typically `0.01`. See [ACT paper](https://arxiv.org/abs/2304.13705). |
| `action_drop`            | Drop first `n` predicted actions to compensate for control lag.                                 |
| `noise_scheduler`        | Type of noise scheduler used during training.                                                   |
| `num_train_steps`        | Number of diffusion timesteps for training.                                                     |
| `num_inference_steps`    | Number of diffusion timesteps for inference. Defaults to `num_train_steps` if unset.            |

---

## Evaluation

To evaluate a trained policy:

```bash
python evaluate_robosuite.py --config-path <path_to_config> --config-name <config_file_name>---
```
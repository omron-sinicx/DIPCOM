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

"""
Training script for diffusion-based imitation learning policies.

This script provides a clean, modular training pipeline with:
- EMA (Exponential Moving Average) weight tracking
- Early stopping with patience
- Checkpoint management with rotation
- Configurable learning rate scheduling
"""

import hashlib
import timeit
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import torch
from aim import Run
from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm.rich import tqdm

from dipcom.common.datasets.utils import dataset_to_policy_features
from dipcom.common.policies.types import FeatureType
from dipcom.common.utils.ema import ExponentialMovingAverage
from dipcom.common.utils.utils import (
    get_policy,
    save_training_config,
    set_seed,
)
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)


# =============================================================================
# Helper Classes
# =============================================================================
def split_dataset(dataset, cfg):
    # Split into train/validation
    assert cfg.policy.train_ratio > 0 and cfg.policy.train_ratio < 1, "train_ratio must be between 0 and 1"
    val_size = int(len(dataset) * (1 - cfg.policy.train_ratio))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    logger.info(f"\nDataset split:")
    logger.info(f"  Training: {len(train_dataset)} samples")
    logger.info(f"  Validation: {len(val_dataset)} samples")

    return train_dataset, val_dataset

@dataclass
class EarlyStopping:
    """Tracks validation loss and triggers early stopping when no improvement."""

    enabled: bool = True
    patience: int = 50
    min_delta: float = 0.0001
    patience_counter: int = field(default=0, init=False)
    best_loss: float = field(default=np.inf, init=False)

    def check(self, val_loss: float) -> bool:
        """
        Check if training should stop.

        Returns:
            True if training should stop, False otherwise.
        """
        if not self.enabled:
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.patience_counter = 0
            return False

        self.patience_counter += 1
        tqdm.write(
            f"Early stopping: no improvement for "
            f"{self.patience_counter}/{self.patience} checks"
        )

        if self.patience_counter >= self.patience:
            tqdm.write(
                f"Early stopping triggered! Val loss hasn't improved "
                f"for {self.patience} validation checks."
            )
            return True

        return False

    def improved(self, val_loss: float) -> bool:
        """Check if validation loss improved (without updating state)."""
        return val_loss < self.best_loss - self.min_delta


class CheckpointManager:
    """Manages saving, loading, and rotating model checkpoints."""

    def __init__(self, ckpt_dir: Path, max_checkpoints: int = 4, seed: int = 42):
        self.ckpt_dir = ckpt_dir
        self.max_checkpoints = max_checkpoints
        self.seed = seed
        self.checkpoint_files: deque = deque()
        self.ema_checkpoint_files: deque = deque()

    def save_checkpoint(
        self,
        policy: torch.nn.Module,
        epoch: int,
        ema_helper: Optional[ExponentialMovingAverage] = None,
    ) -> None:
        """Save policy checkpoint and optionally EMA checkpoint."""
        # Save regular checkpoint
        ckpt_path = self.ckpt_dir / f"policy_epoch_{epoch}_seed_{self.seed}.ckpt"
        torch.save(policy.state_dict(), ckpt_path)
        self.checkpoint_files.append(ckpt_path)

        # Save EMA checkpoint if enabled
        if ema_helper is not None:
            ema_ckpt_path = self.ckpt_dir / f"ema_policy_epoch_{epoch}_seed_{self.seed}.ckpt"
            ema_helper.store(policy.parameters())
            ema_helper.copy_to(policy.parameters())
            torch.save(policy.state_dict(), ema_ckpt_path)
            ema_helper.restore(policy.parameters())
            self.ema_checkpoint_files.append(ema_ckpt_path)

        # Rotate old checkpoints
        self._rotate_checkpoints(ema_helper is not None)

    def _rotate_checkpoints(self, has_ema: bool) -> None:
        """Remove oldest checkpoints if exceeding max_checkpoints."""
        if len(self.checkpoint_files) > self.max_checkpoints:
            oldest = self.checkpoint_files.popleft()
            if oldest.exists():
                oldest.unlink()

            if has_ema:
                oldest_ema = self.ema_checkpoint_files.popleft()
                if oldest_ema.exists():
                    oldest_ema.unlink()

    def save_best(
        self,
        policy_state: dict,
        ema_state: Optional[dict] = None,
    ) -> None:
        """Save the best policy checkpoint."""
        torch.save(policy_state, self.ckpt_dir / "best_policy.ckpt")
        if ema_state is not None:
            torch.save(ema_state, self.ckpt_dir / "best_ema_policy.ckpt")

    def save_final(
        self,
        policy: torch.nn.Module,
        ema_helper: Optional[ExponentialMovingAverage] = None,
    ) -> None:
        """Save the final policy checkpoint."""
        torch.save(policy.state_dict(), self.ckpt_dir / "policy_last.ckpt")

        if ema_helper is not None:
            ema_helper.store(policy.parameters())
            ema_helper.copy_to(policy.parameters())
            torch.save(policy.state_dict(), self.ckpt_dir / "ema_policy_last.ckpt")
            ema_helper.restore(policy.parameters())


class Trainer:
    """
    Main trainer class for diffusion policy training.

    Handles the complete training pipeline including:
    - Model and optimizer setup
    - Training and validation loops
    - EMA weight tracking
    - Checkpoint management
    - Early stopping
    - Metrics logging
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda")

        # Initialize components
        set_seed(cfg.policy.seed)
        self._setup_directories()
        self._setup_logging()
        self._setup_data()
        self._setup_model()
        self._setup_training()

    def _setup_directories(self) -> None:
        """Create checkpoint directory and save config."""
        self.ckpt_dir = Path(self.cfg.paths.ckpt_dir) / "runs" / self.cfg.paths.leaf_dir
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Remove existing config if present
        destfile = self.ckpt_dir / "config.yaml"
        if destfile.exists():
            destfile.unlink()

        save_training_config(
            cfg=self.cfg, ckpt_dir=self.ckpt_dir, model_name=self.cfg.model_name
        )

    def _setup_logging(self) -> None:
        """Initialize Aim experiment tracking."""
        self.aim_run = Run(
            repo=str(self.cfg.paths.output_dir), experiment=self.cfg.model.name
        )
        self.aim_run["config"] = self.cfg

    def _setup_data(self) -> None:
        """Load dataset and create dataloaders."""
        cfg = self.cfg

        # Select dataset class
        if isinstance(cfg.dataset.dataset.repo_id, (list, ListConfig)):
            dataset_class = MultiLeRobotDataset
        else:
            dataset_class = LeRobotDataset

        dataset_repo_dir = (
            Path(cfg.dataset.dataset.dir) if cfg.dataset.dataset.dir else None
        )

        # Load metadata and features
        dataset_meta = LeRobotDatasetMetadata(
            cfg.dataset.dataset.repo_id, root=dataset_repo_dir
        )
        self.dataset_stats = dataset_meta.stats

        all_features = dataset_to_policy_features(dataset_meta.features)
        self.features = {
            k: v
            for k, v in all_features.items()
            if any(partial in k for partial in list(cfg.wanted_features))
        }

        # Create dataset with image transforms
        if "camera_shape" in cfg.model.diffusion_config:
            camera_shape = cfg.model.diffusion_config.camera_shape
            image_scale = transforms.Resize(camera_shape[1:])
            for key in self.features:
                if key.startswith("observation.images"):
                    self.features[key].shape = camera_shape
        else:
            camera_shape = None
            image_scale = None

        logger.debug(f"Features being trained: {self.features}")

        # Separate input/output features
        self.output_features = {
            k: ft for k, ft in self.features.items() if ft.type is FeatureType.ACTION
        }
        self.input_features = {
            k: ft for k, ft in self.features.items() if k not in self.output_features
        }

        # Build delta timestamps for temporal context
        delta_timestamps = self._build_delta_timestamps()

        dataset = dataset_class(
            cfg.dataset.dataset.repo_id,
            root=dataset_repo_dir,
            delta_timestamps=delta_timestamps,
            video_backend="pyav",
            image_transforms=image_scale,
        )

        train_dataset, val_dataset = split_dataset(dataset, cfg)

        self.train_dataloader = DataLoader(
            train_dataset,
            num_workers=cfg.policy.num_workers,
            batch_size=cfg.policy.batch_size,
            shuffle=cfg.policy.shuffle,
            pin_memory=self.device != torch.device("cpu"),
            drop_last=cfg.policy.drop_last,
            persistent_workers=cfg.policy.persistent_workers
        )

        self.val_dataloader = DataLoader(
            val_dataset,
            num_workers=cfg.policy.num_workers,
            batch_size=cfg.policy.batch_size,
            shuffle=cfg.policy.shuffle,
            pin_memory=self.device != torch.device("cpu"),
            drop_last=cfg.policy.drop_last,
            persistent_workers=cfg.policy.persistent_workers
        )

    def _build_delta_timestamps(self) -> dict:
        """Build temporal offset timestamps for observations and actions."""
        cfg = self.cfg
        fps = cfg.policy.fps
        n_obs_steps = cfg.model.diffusion_config.n_obs_steps
        horizon = cfg.model.diffusion_config.horizon

        # Observation timestamps: past context up to current time
        obs_timestamps = {
            k: [i / fps for i in range(1 - n_obs_steps, 1)]
            for k in self.features.keys()
            if not k.startswith("action")
        }

        # Action timestamps: from past context through prediction horizon
        action_timestamps = {
            k: [i / fps for i in range(1 - n_obs_steps, 1 - n_obs_steps + horizon)]
            for k in self.features.keys()
            if k.startswith("action")
        }

        return {**obs_timestamps, **action_timestamps}


    def _setup_model(self) -> None:
        """Initialize policy model and EMA."""
        cfg = self.cfg

        # Create diffusion config with features
        diffusion_config = hydra.utils.instantiate(
            cfg.model.diffusion_config,
            output_features=self.output_features,
            input_features=self.input_features,
        )

        # Create and load policy
        self.policy = get_policy(
            cfg.model.name, diffusion_config, stats=self.dataset_stats
        )
        self.policy.to(self.device)

        # Log model size
        num_params = sum(p.numel() for p in self.policy.parameters())
        num_trainable = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(
            f"Model parameters: {num_params/1e6:.2f}M | "
            f"Trainable: {num_trainable/1e6:.2f}M"
        )

        # Load pretrained weights if specified
        if "pretrained_policy" in cfg.policy:
            self.policy.load_state_dict(torch.load(cfg.policy.pretrained_policy))
            logger.info(f"Loaded pretrained policy from {cfg.policy.pretrained_policy}")

        # Initialize EMA
        self.use_ema = cfg.policy.use_ema
        self.ema_helper = ExponentialMovingAverage(
            self.policy.parameters(), decay=cfg.policy.decay, device=self.device
        )

    def _setup_training(self) -> None:
        """Initialize optimizer, scheduler, and training utilities."""
        cfg = self.cfg

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=cfg.policy.lr,
            weight_decay=1.0e-4,
        )

        # Learning rate scheduler
        self.lr_scheduler = None
        if cfg.policy.lr_scheduler:
            if cfg.policy.lr_scheduler_type == "cosine_annealing":
                self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=self.optimizer,
                    T_max=cfg.policy.epochs,
                    eta_min=cfg.policy.lr_min,
                )
            elif cfg.policy.lr_scheduler_type == "one_cycle":
                self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer=self.optimizer,
                    max_lr=cfg.policy.lr,
                    total_steps=cfg.policy.epochs,
                    pct_start=0.3,
                    anneal_strategy="cos",
                )
            else:
                raise ValueError(f"Invalid learning rate scheduler type: {cfg.policy.lr_scheduler_type}")

        # Mixed precision training (AMP)
        self.use_amp = getattr(cfg.policy, "use_amp", True)
        self.scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            logger.info("Mixed precision training (AMP) enabled")

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            ckpt_dir=self.ckpt_dir,
            max_checkpoints=cfg.policy.max_checkpoints,
            seed=cfg.policy.seed,
        )

        # Early stopping
        self.early_stopping = EarlyStopping(
            enabled=cfg.policy.early_stopping.enabled,
            patience=cfg.policy.early_stopping.patience,
            min_delta=cfg.policy.early_stopping.min_delta,
        )

        # Best checkpoint tracking
        self.best_ckpt_info = None
        self.ema_best_ckpt_info = None

    def train_one_epoch(self, epoch: int) -> int:
        """
        Run one training epoch.

        Returns:
            The final iteration number of the epoch.
        """
        self.policy.train()

        epoch_iter = 0
        mean_loss = 0.0
        for batch_idx, batch in enumerate(
            tqdm(
                self.train_dataloader,
                total=len(self.train_dataloader),
                desc=f"Epoch {epoch + 1}",
                leave=False,
                position=1,
                dynamic_ncols=True,
            )
        ):
            iter_num = epoch * len(self.train_dataloader) + batch_idx

            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            # Forward pass with mixed precision
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                loss, _ = self.policy(batch)
            mean_loss += loss.item()

            # Backward pass with gradient scaling for AMP
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), max_norm=self.cfg.policy.grad_clip_norm
            )

            # Optimizer step with scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            # Update EMA
            if self.use_ema:
                self.ema_helper.update(self.policy.parameters())

            # Log metrics
            self.aim_run.track(
                loss.item(), name="loss", context={"subset": "train"}, step=iter_num
            )

            epoch_iter = iter_num

        mean_loss /= len(self.train_dataloader)
        self.aim_run.track(mean_loss, name="epoch_mean_loss", context={"subset": "train"}, step=epoch)
        self.aim_run.track(
            self.optimizer.param_groups[0]["lr"], name="LR", step=epoch
        )

        # Step LR scheduler after epoch
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return epoch_iter

    @torch.no_grad()
    def validate(self) -> dict:
        """
        Run validation and return metrics.

        Uses EMA weights if enabled.
        """
        # Swap to EMA weights for validation
        if self.use_ema:
            self.ema_helper.store(self.policy.parameters())
            self.ema_helper.copy_to(self.policy.parameters())

        self.policy.eval()
        total_loss = 0.0
        batch_device = "cpu"

        for val_batch in tqdm(
            self.val_dataloader,
            total=len(self.val_dataloader),
            desc="Validation",
            position=1,
            leave=False,
            dynamic_ncols=True,
        ):
            if self.cfg.policy.use_batch_dataset and batch_device != self.device:
                val_batch = {k: v.to(self.device) for k, v in val_batch.items() if isinstance(v, torch.Tensor)}

            if not self.cfg.policy.use_batch_dataset:
                val_batch = {k: v.to(self.device) for k, v in val_batch.items() if isinstance(v, torch.Tensor)}

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                val_loss, _ = self.policy(val_batch)
            total_loss += val_loss.item()

        # Restore original weights
        if self.use_ema:
            self.ema_helper.restore(self.policy.parameters())

        avg_loss = total_loss / len(self.val_dataloader)
        return {"loss": avg_loss}

    def _update_best_checkpoint(self, val_loss: float, epoch: int) -> None:
        """Update best checkpoint if validation improved."""
        self.best_ckpt_info = (epoch, val_loss, deepcopy(self.policy.state_dict()))

        # Save best policy
        self.checkpoint_manager.save_best(self.best_ckpt_info[2])

        # Save EMA version of best policy
        if self.use_ema:
            self.ema_helper.store(self.policy.parameters())
            self.ema_helper.copy_to(self.policy.parameters())
            self.ema_best_ckpt_info = deepcopy(self.policy.state_dict())
            self.ema_helper.restore(self.policy.parameters())
            self.checkpoint_manager.save_best(
                self.best_ckpt_info[2], self.ema_best_ckpt_info
            )

    def run(self) -> tuple:
        """
        Run the complete training loop.

        Returns:
            Tuple of (best_epoch, best_val_loss, best_state_dict)
        """
        cfg = self.cfg
        start_time = timeit.default_timer()
        iter_num = 0
        early_stopped = False

        for epoch in tqdm(
            range(cfg.policy.epochs),
            desc="Training",
            position=0,
            dynamic_ncols=True,
        ):
            # Training
            iter_num = self.train_one_epoch(epoch)

            # Validation
            if (epoch + 1) % cfg.policy.validation_interval == 0:
                val_dict = self.validate()
                val_loss = val_dict["loss"]

                # Log validation metrics
                self.aim_run.track(
                    val_loss, name="epoch_mean_loss", context={"subset": "val"}, step=epoch
                )

                tqdm.write(
                    f"Epoch {epoch + 1}: Val loss: {val_loss:.5f} "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )

                # Check for improvement and update best checkpoint
                if self.early_stopping.improved(val_loss):
                    self._update_best_checkpoint(val_loss, epoch)

                # Check early stopping
                if self.early_stopping.check(val_loss):
                    early_stopped = True

            # Periodic checkpoint saving
            if (epoch + 1) % cfg.policy.save_policy_interval == 0:
                ema = self.ema_helper if self.use_ema else None
                self.checkpoint_manager.save_checkpoint(self.policy, epoch, ema)

            if early_stopped:
                break

        # Training complete
        if early_stopped:
            logger.info(
                f"Training stopped early at iteration {iter_num} "
                "due to early stopping"
            )

        elapsed = round(timeit.default_timer() - start_time, 2)
        logger.info(f"Total Training Time: {elapsed}s")

        # Save final checkpoints
        self._save_final_checkpoints()

    def _save_final_checkpoints(self) -> None:
        """Save final model checkpoints."""
        ema = self.ema_helper if self.use_ema else None
        self.checkpoint_manager.save_final(self.policy, ema)

        if self.best_ckpt_info is not None:
            best_epoch, best_loss, best_state = self.best_ckpt_info
            ckpt_path = (
                self.ckpt_dir
                / f"policy_epoch_{best_epoch}_seed_{self.cfg.policy.seed}.ckpt"
            )
            torch.save(best_state, ckpt_path)

            if self.ema_best_ckpt_info is not None:
                ema_path = (
                    self.ckpt_dir
                    / f"ema_policy_epoch_{best_epoch}_seed_{self.cfg.policy.seed}.ckpt"
                )
                torch.save(self.ema_best_ckpt_info, ema_path)

            logger.info(
                f"Training finished: Seed {self.cfg.policy.seed}, "
                f"val loss {best_loss:.6f} at epoch {best_epoch}"
            )
        else:
            logger.warning("No best checkpoint was saved (validation never improved)")



# =============================================================================
# Main Entry Point
# =============================================================================


@hydra.main(version_base=None, config_path="../../configs", config_name="single_wipe.yaml")
def main(cfg: DictConfig) -> None:
    """Main training entry point."""

    # print(OmegaConf.to_yaml(cfg, resolve=True))
    # exit(1)

    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()

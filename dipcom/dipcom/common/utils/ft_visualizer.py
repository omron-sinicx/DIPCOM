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

import collections
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Union, List

from robosuite.utils.buffers import RingBuffer


class FTVisualizer:
    """A class to handle Force/Torque data visualization with matplotlib."""

    def __init__(
        self,
        maxlen: Optional[int] = None,
        include_stiffness: bool = False,
        arm: str = "right",
        title: Optional[str] = None,
        figure_size: tuple = (6, 4),
        force_ylim: tuple = (-50, 50),
        stiffness_ylim: tuple = (0, 1000)
    ):
        """
        Initialize the FT visualizer.

        Args:
            maxlen: Maximum length of data history to keep (None for unlimited)
            include_stiffness: Whether to include stiffness visualization
            arm: Which arm to visualize ("left", "right", "single")
            title: Custom title for the plot
            figure_size: Figure size as (width, height)
            force_ylim: Y-axis limits for force plot as (min, max)
            stiffness_ylim: Y-axis limits for stiffness plot as (min, max)
        """
        self.maxlen = maxlen
        self.include_stiffness = include_stiffness
        self.arm = arm
        self.figure_size = figure_size
        self.force_ylim = force_ylim
        self.stiffness_ylim = stiffness_ylim

        # Data storage
        self.times = collections.deque(maxlen=maxlen) if maxlen else []
        self.ft_filter = RingBuffer(dim=6, length=5)
        self.forces_hist = collections.deque(maxlen=maxlen) if maxlen else []
        self.ft_filter_hist = collections.deque(maxlen=maxlen) if maxlen else []
        self.stiffness_hist = collections.deque(maxlen=maxlen) if maxlen and include_stiffness else ([] if include_stiffness else None)

        # Plot setup
        self.fig = None
        self.axs = None
        self.axs_twin = None
        self._setup_plot(title)

        # Rendering control
        self._step_count = 0

    def _setup_plot(self, title: Optional[str]):
        """Setup the matplotlib figure and axes."""
        self.fig, self.axs = plt.subplots(figsize=self.figure_size)

        if self.include_stiffness:
            self.axs_twin = self.axs.twinx()
        else:
            self.axs_twin = None

        plot_title = title or f'Force and Torque Readings of the {self.arm.title()} Arm'
        self.fig.suptitle(plot_title, fontsize=14)

        # Set window position - handle different backends
        mngr = self.fig.canvas.manager
        try:
            if hasattr(mngr, 'window'):
                # Try Tkinter backend first
                if hasattr(mngr.window, 'wm_geometry'):
                    mngr.window.wm_geometry("+100+50")
                # Try Qt backend
                elif hasattr(mngr.window, 'move'):
                    mngr.window.move(100, 50)
        except Exception:
            # If window positioning fails, just continue without positioning
            pass

        plt.show(block=False)

    def add_data(
        self,
        time_step: Union[int, float],
        forces: np.ndarray,
        stiffness: Optional[Union[float, np.ndarray]] = None
    ):
        """
        Add a new data point to the visualization.

        Args:
            time_step: Current time step or time value
            forces: Force/torque data as numpy array
            stiffness: Optional stiffness value (only used if include_stiffness=True)
        """
        self.times.append(time_step)
        self.forces_hist.append(forces)
        self.ft_filter.push(forces)
        self.ft_filter_hist.append(np.linalg.norm(self.ft_filter.average))

        if self.include_stiffness and stiffness is not None:
            self.stiffness_hist.append(stiffness)

        self._step_count += 1

    def _visualize_internal(self):
        """Internal method that handles the actual visualization logic."""
        if len(self.times) == 0:
            return

        labels = ['Fx', 'Fy', 'Fz']
        colors = ['r', 'g', 'b']
        # Update the force subplot
        offset = 6 if self.arm in ["right"] else 0

        self.axs.clear()

        forces_array = np.array(self.forces_hist)

        self.axs.plot(self.times, np.linalg.norm(forces_array[:, offset:offset+3], axis=1), label="Normal Force", color=colors[1], alpha=0.4, linewidth=1, linestyle='--')
        self.axs.plot(self.times, self.ft_filter_hist, label='Filtered Force', color=colors[0], alpha=0.8, linewidth=1)
        # for i in range(3):
        #     self.axs.plot(self.times, forces_array[:, i+offset], label=labels[i], color=colors[i], alpha=0.8, linewidth=1)

        self.axs.set_ylabel('force')
        self.axs.set_ylim(self.force_ylim[0], self.force_ylim[1])
        self.axs.legend(loc='upper left')
        self.axs.grid()

        if self.include_stiffness and self.stiffness_hist is not None and len(self.stiffness_hist) > 0:
            self.axs_twin.clear()
            color = 'tab:olive'
            self.axs_twin.yaxis.set_label_position("right")
            self.axs_twin.plot(self.times, self.stiffness_hist, color=color)
            self.axs_twin.set_ylabel('stiffness', color=color)
            self.axs_twin.tick_params(axis='y', color=color)
            self.axs_twin.set_ylim(self.stiffness_ylim[0], self.stiffness_ylim[1])

        self.fig.tight_layout()

        # Keep plot in focus
        plt.figure(self.fig.number)
        plt.show(block=False)
        plt.pause(0.001)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def render(self, skip_frame: int = 10):
        """
        Render the visualization if conditions are met.

        Args:
            skip_frame: Only render every skip_frame steps for performance
        """
        if self._step_count % skip_frame == 0 and len(self.times) > 0:
            self._visualize_internal()

    def render_now(self):
        """Force an immediate render regardless of skip_frame."""
        if len(self.times) > 0:
            self._visualize_internal()

    def clear(self):
        """Clear all stored data."""
        self.times.clear()
        self.forces_hist.clear()
        self.ft_filter.clear()
        self.ft_filter_hist.clear()
        for i in range(self.maxlen):
            self.times.append(-self.maxlen + i)
            self.forces_hist.append(np.zeros(6))
            self.ft_filter_hist.append(0)
        if self.stiffness_hist is not None:
            self.stiffness_hist.clear()
        self._step_count = 0

    def close(self):
        """Close the matplotlib figure."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axs = None
            self.axs_twin = None

    def save(self, filepath: str):
        """
        Save the current plot to a file.

        Args:
            filepath: Path where to save the figure
        """
        if self.fig is not None:
            self.fig.savefig(filepath)

    def set_xlim(self, xmin: Optional[float] = None, xmax: Optional[float] = None):
        """Set x-axis limits for the plot."""
        if self.axs is not None:
            self.axs.set_xlim(xmin=xmin, xmax=xmax)

    def get_data(self):
        """
        Get all stored data.

        Returns:
            tuple: (times, forces_hist, stiffness_hist)
        """
        return list(self.times), list(self.forces_hist), list(self.stiffness_hist) if self.stiffness_hist else None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically close the plot."""
        self.close()

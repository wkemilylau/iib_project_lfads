import colorsys
from glob import glob

import h5py
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.io import loadmat
from scipy.signal import lfilter
from scipy.signal.windows import gaussian
from sklearn.decomposition import PCA

SESSION_ID = "PAG"


def merge_data(train_data, valid_data, train_inds, valid_inds):
    n_neurons = train_data.shape[2]
    n_timepoints = train_data.shape[1]
    n_trials = train_data.shape[0] + valid_data.shape[0]
    merged_data = np.zeros((n_trials, n_timepoints, n_neurons))
    for i, ind in enumerate(train_inds):
        merged_data[ind] = train_data[i]
    for i, ind in enumerate(valid_inds):
        merged_data[ind] = valid_data[i]
    return merged_data


def load_results(data_path):
    rates = {}
    factors = {}
    spikes = {}
    inferred_inputs = {}
    initial_conditions = {}

    session_id = SESSION_ID

    with h5py.File(data_path, "r") as f:
        train_encod_data = f["train_encod_data"][:]
        valid_encod_data = f["valid_encod_data"][:]
        train_factors = f["train_factors"][:]
        valid_factors = f["valid_factors"][:]
        train_rates = f["train_output_params"][:]
        valid_rates = f["valid_output_params"][:]
        train_inferred_inputs = f["train_co_means"][:]
        valid_inferred_inputs = f["valid_co_means"][:]
        train_initial_conditions = f["train_ic_mean"][:]
        valid_initial_conditions = f["valid_ic_mean"][:]
        train_inds = f["train_inds"][:].astype(int)
        valid_inds = f["valid_inds"][:].astype(int)

        spikes[session_id] = merge_data(
            train_encod_data, valid_encod_data, train_inds, valid_inds
        )
        factors[session_id] = merge_data(
            train_factors, valid_factors, train_inds, valid_inds
        )
        rates[session_id] = merge_data(train_rates, valid_rates, train_inds, valid_inds)
        inferred_inputs[session_id] = merge_data(
            train_inferred_inputs, valid_inferred_inputs, train_inds, valid_inds
        )
        initial_conditions[session_id] = np.concatenate(
            (train_initial_conditions, valid_initial_conditions), axis=0
        )

        return spikes, factors, rates, inferred_inputs, initial_conditions


def plot_rates_individual(spikes_data, rates_data, title, axes=None):
    """
    Plots overlay PSTHs for the first 8 neurons.
    Returns the figure and axes objects for further use in subfigures.
    """
    if axes is None:
        # Create a new figure if no axes are provided
        fig, axes = plt.subplots(2, 4, figsize=(4, 8), sharex=True)
    else:
        # Use the figure associated with the passed axes
        fig = axes.flatten()[0].get_figure()

    axes_flat = axes.flatten()

    for i in range(8):
        if i >= len(axes_flat):
            break

        ax = axes_flat[i]

        # Calculate the mean across all trials for the i-th neuron
        # spikes_data shape: (trials, time, neurons)
        spikes_mean_activity = np.mean(spikes_data[:, :, i], axis=0)
        rates_mean_activity = np.mean(rates_data[:, :, i], axis=0)

        ax.plot(
            spikes_mean_activity,
            color="teal",
            linewidth=1,
            alpha=0.8,
            label="Raw" if i == 0 else "",
        )
        ax.plot(
            rates_mean_activity,
            color="red",
            linewidth=1,
            alpha=0.8,
            label="LFADS" if i == 0 else "",
        )
        ax.set_title(f"Neuron {i}", fontsize=10)

        # Clean up visual clutter
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Labels for outer plots
        if i >= 4:
            ax.set_xlabel("Time (bins)")
        if i % 4 == 0:
            ax.set_ylabel("Firing Rate")

    plt.suptitle(title, fontsize=16, y=1.02)

    return fig, axes


def plot_condition_averages(spikes_data, rates_data, title, axes=None):
    if axes is None:
        # Create a new figure if no axes are provided
        fig, axes = plt.subplots(1, 4, figsize=(10, 3), sharex=True, sharey=True)
    else:
        # Use the figure associated with the passed axes
        fig = axes.flatten()[0].get_figure()

    # Flatten the axes array to easily iterate through all 4 subplots
    axes_flat = axes.flatten()
    conditions = ["Danger", "Uncertain shock", "Uncertain no shock", "Safe"]

    mean_spikes_danger = np.mean(np.mean(spikes_data[0:4, :, :], axis=2), axis=0)
    mean_spikes_unc_noshock = np.mean(np.mean(spikes_data[4:6, :, :], axis=2), axis=0)
    mean_spikes_unc_shock = np.mean(np.mean(spikes_data[6:11, :, :], axis=2), axis=0)
    mean_spikes_safety = np.mean(np.mean(spikes_data[11:15, :, :], axis=2), axis=0)

    spikes = [
        mean_spikes_danger,
        mean_spikes_unc_noshock,
        mean_spikes_unc_shock,
        mean_spikes_safety,
    ]

    mean_rates_danger = np.mean(np.mean(rates_data[0:4, :, :], axis=2), axis=0)
    mean_rates_unc_noshock = np.mean(np.mean(rates_data[4:6, :, :], axis=2), axis=0)
    mean_rates_unc_shock = np.mean(np.mean(rates_data[6:11, :, :], axis=2), axis=0)
    mean_rates_safety = np.mean(np.mean(rates_data[11:15, :, :], axis=2), axis=0)

    rates = [
        mean_rates_danger,
        mean_rates_unc_noshock,
        mean_rates_unc_shock,
        mean_rates_safety,
    ]

    for i in range(4):
        ax = axes_flat[i]

        ax.plot(
            spikes[i][1:],
            color="teal",
            linewidth=1,
            alpha=0.8,
            label="Raw mean firing rate",
        )
        ax.plot(
            rates[i][1:],
            color="red",
            linewidth=1,
            alpha=0.8,
            label="LFADS mean firing rate",
        )
        ax.set_title(conditions[i], fontsize=12)

        if i == 0:
            ax.legend()

        # Clean up visual clutter
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add labels only to the bottom-most plots for a cleaner look
        if i >= 0:
            ax.set_xlabel("Timestep")
        if i % 1 == 0:
            ax.set_ylabel("Mean firing rate")

    plt.tight_layout()
    plt.show()

    return fig, axes


def plot_pca(data_array, title, axes=None):
    """
    Performs PCA on reshaped (Trials, Time * Dimensions) data and plots
    on a provided axis for subplot integration.
    """
    # 1. Reshape
    n_trials, n_time, n_dim = data_array.shape
    reshaped_data = data_array.reshape(n_trials, -1)

    # 2. Handle Axis/Figure
    if axes is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        ax = axes  # Connect local 'ax' to passed 'axes'
        fig = ax.get_figure()

    # 3. Define labels and palette
    condition_counts = {
        "Danger": 6,
        "Uncertain Shock": 6,
        "Uncertain No Shock": 10,
        "Safety": 10,
    }
    base_palette = {
        "Danger": "#e74c3c",
        "Uncertain Shock": "#9b59b6",
        "Uncertain No Shock": "#3498db",
        "Safety": "#2ecc71",
    }

    labels, trial_colors = [], []
    for cond, count in condition_counts.items():
        base_rgb = mcolors.to_rgb(base_palette[cond])
        h, s, v = colorsys.rgb_to_hsv(*base_rgb)
        for i in range(count):
            labels.append(cond)
            # Saturation logic
            saturation_level = 1 - (0.1 + (0.9 * ((i + 1) / count)))
            new_rgb = colorsys.hsv_to_rgb(h, saturation_level, v)
            trial_colors.append(new_rgb)

    # 4. Run PCA
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(reshaped_data)
    var_explained = pca.explained_variance_ratio_ * 100

    # 5. Visualization - Plot directly onto 'ax'
    sns.set_style("whitegrid")

    # Plot dots
    for i in range(n_trials):
        plt.scatter(
            pca_results[i, 0],
            pca_results[i, 1],
            color=trial_colors[i],
            s=120,
            edgecolor="black",
            alpha=0.9,
            label=labels[i]
            if labels[i] not in plt.gca().get_legend_handles_labels()[1]
            else "",
        )

    # 6. Manual Legend (Fixes the UserWarning)
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=cond,
            markerfacecolor=color,
            markersize=10,
            markeredgecolor="black",
        )
        for cond, color in base_palette.items()
    ]

    ax.set_title(title, fontsize=15)
    ax.set_xlabel(f"PC 1 ({var_explained[0]:.1f}% Var)")
    ax.set_ylabel(f"PC 2 ({var_explained[1]:.1f}% Var)")

    ax.legend(title="Condition", bbox_to_anchor=(1, 1), loc="upper right")

    return fig, ax


def plot_inputs():
    pass


def plot_input_energy():
    pass

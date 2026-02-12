import importlib

import evaluation as eval
import matplotlib.pyplot as plt
import numpy as np

importlib.reload(eval)

SESSION_ID = eval.SESSION_ID


def create_lfads_summary_plots(results_file):
    """
    Creates summary plots for LFADS results.
    """
    # Setting a clean style
    plt.style.use("seaborn-v0_8-muted")

    # load results from results_file FUNCTION
    spikes, factors, rates, inputs, init_conds = eval.load_results(results_file)

    # 1. FIGURE: Denoised Firing Rates
    # a) denoised firing rate on first 8 neurons
    # b) average denoised firing rate on each condition
    fig1 = plt.figure(figsize=(21, 12))
    fig1_subfigs = fig1.subfigures(2, 1, height_ratios=[2, 1], hspace=0.2)
    axes1a = fig1_subfigs[0].subplots(2, 4, sharex=True)
    eval.plot_rates_individual(
        spikes[SESSION_ID],
        rates[SESSION_ID],
        "Firing rates of first 8 neurons",
        axes=axes1a,
    )
    axes1b = fig1_subfigs[1].subplots(1, 4, sharex=True, sharey=True)
    eval.plot_condition_averages(
        spikes[SESSION_ID],
        rates[SESSION_ID],
        "Condition-averaged firing rates",
        axes=axes1b,
    )

    # 2. FIGURE: Latent Factors
    # a) PCA on latent factors
    fig2 = plt.figure(figsize=(10, 10))
    fig2.suptitle("Figure 2: Latent factors", fontsize=16)
    fig2_subfigs = fig2.subfigures(1, 1)
    axes2a = fig2_subfigs.subplots(1, 1)
    eval.plot_pca(
        factors[SESSION_ID], "PCA projection of latent factors by condition", axes2a
    )

    # 3. FIGURE: Inferred Inputs
    # a) Inferred input trajectories (grouped by dimension)
    # b) PCA on inferred inputs
    # c) Input energy
    fig3 = plt.figure(figsize=(10, 10))
    fig3.suptitle("Figure 3: Inferred inputs", fontsize=16)
    fig3_subfigs = fig3.subfigures(2, 2)
    axes3a = fig3_subfigs[0, 0].subplots(1, 1)
    # eval.plot_inputs(inputs[SESSION_ID], "Inferred inputs by dimension", axes3a)
    axes3b = fig3_subfigs[0, 1].subplots(1, 1)
    eval.plot_pca(
        inputs[SESSION_ID], "PCA projection of inferred inputs by condition", axes3b
    )
    axes3c = fig3_subfigs[1, 0].subplots(1, 1)
    # eval.plot_input_energy(inputs[SESSION_ID], "Input energy by condition", axes3c)

    # 4. FIGURE: Initial Conditions
    # a) PCA on initial conditions
    # fig4 = plt.figure(figsize=(10, 10))
    # fig4.suptitle('Figure 4: Initial conditions', fontsize=16)
    # fig4_subfigs = fig4.subfigures(1, 1)
    # axes4a = fig4_subfigs.subplots(1, 1)
    # eval.plot_pca(init_conds[SESSION_ID], "PCA projection of latent factors by condition", axes4a)

    # # Formatting all figures
    # for fig in [fig1, fig2, fig3, fig4]:
    #     fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig1, fig2, fig3


if __name__ == "__main__":
    # --- Configuration ---
    # Update these paths to your actual .h5 or .pt files
    results_file = "/homes/wkel2/iib_project_lfads/tutorials/multisession/lfads_output_pag_multisession_4_lfads_EAA.h5"

    print(f"Creating summary figures for {results_file}...")

    # Generate the skeletons
    f1, f2, f3 = create_lfads_summary_plots(results_file)

    f1.savefig("summary_neurons.png")
    f2.savefig("summary_factors.png")
    f3.savefig("summary_inputs.png")
    # f4.savefig("summary_ics.png")
    plt.show()

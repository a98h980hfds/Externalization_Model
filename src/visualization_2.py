import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import re

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
})

COLOR_DICT = {
    "alpha": "green",
    "beta": "red",
    "gamma": "dodgerblue",
    "delta": "orange",
    "externalizers": "darkblue",
    "non-externalizers": "purple"
}

def read_data(csvpath="/../data/ABM_base_simulation.csv"):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(file_dir + csvpath, dtype=str)
    df = df.loc[2:len(df)]

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass

    index_colnames = ["simulation", "generation", "learning_step"]
    df.rename(columns=dict(zip(df.columns[:3], index_colnames)), inplace=True)
    df.set_index(index_colnames, inplace=True)

    cols = []
    half = int(len(df.columns) / 2)
    for i in range(half):
        for attr in ["externalization", "behavior"]:
            cols.append((i, attr))
    df.columns = pd.MultiIndex.from_tuples(cols)

    return df

def extract_benchmark_list(group_size, mixed):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    path = file_dir + "/../data/significance_test_results.txt"
    with open(path, "r") as f:
        content = f.read()

    blocks = content.split("Significance Test Results")
    blocks = [b.strip() for b in blocks if b.strip()]

    # Build selector string
    if mixed:
        key = f"population size {group_size} mixed:"
    else:
        key = f"population size {group_size}:"

    # Select block
    target = None
    for b in blocks:
        if key in b:
            target = b
            break
    if target is None:
        raise ValueError(f"Requested block not found for key '{key}'")

    # Extract benchmark list
    match = re.search(r'Benchmark results:\s*\[([^\]]+)\]', target)
    if not match:
        raise ValueError("Benchmark list not found in block")

    nums = match.group(1).split(',')
    return [int(x.strip()) for x in nums]

def compute_behavior_summary(df, behaviors):
    max_sim = int(df.reset_index()["simulation"].max())
    max_lst = int(df.reset_index()["learning_step"].max())

    summary = []
    for step in range(max_lst + 1):
        counts = {b: [] for b in behaviors}
        for s in range(max_sim + 1):
            row = df.loc[(s, 0, step), pd.IndexSlice[:, "behavior"]]
            for b in behaviors:
                counts[b].append((row == b).sum())

        for b in behaviors:
            series = pd.Series(counts[b])
            summary.append([step, b, series.mean(), series.quantile(0.975), series.quantile(0.025)])

    return pd.DataFrame(summary, columns=["learning_step", "behavior", "mean", "97.5", "2.5"])

def compute_final_externalizers(df):
    max_sim = int(df.reset_index()["simulation"].max())
    results = []
    for s in range(max_sim + 1):
        gens = df.loc[pd.IndexSlice[s, :, :]].reset_index()["generation"]
        max_gen = int(gens.max())
        count = df.loc[(s, max_gen, 0), pd.IndexSlice[:, "externalization"]].sum()
        results.append(count)
    return results

def plot_shared(learning_df, final_ext, final_A, max_hist, binwidth, ncols):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, ncols, figsize=(4*ncols, 5))
    axes = axes.flatten()

    behaviors = ["alpha", "beta", "gamma", "delta"]

    axes[0].set_ylabel("Number of Agents")
    axes[0].set_xlabel("Learning Cycle")
    sns.lineplot(data=learning_df, x="learning_step", y="mean",
                 hue="behavior", marker='o', palette=COLOR_DICT,
                 markeredgewidth=0, ax=axes[0])
    sns.lineplot(data=learning_df, x="learning_step", y="97.5",
                 hue="behavior", linestyle='--', palette=COLOR_DICT,
                 markeredgewidth=0, ax=axes[0])
    sns.lineplot(data=learning_df, x="learning_step", y="2.5",
                 hue="behavior", linestyle='--', palette=COLOR_DICT,
                 markeredgewidth=0, ax=axes[0])
    axes[0].set_title("Learning Proccess\n(100 simulations aggregate)")

    axes[1].set_ylabel("Number of Simulations")
    axes[1].set_xlabel("Final Externalizers")
    sns.histplot(x=pd.Series(final_ext), stat="count", binwidth=binwidth, color="gray", kde=False, ax=axes[1])
    axes[1].set_ylim(0, 100)
    axes[1].set_xlim(0, max_hist)
    axes[1].set_title("Externalization\nSimulation Results")

    if final_A:
        axes[2].set_ylabel("Number of Simulations")
        axes[2].set_xlabel("Final Trait A Agents")
        sns.histplot(x=pd.Series(final_A), stat="count", binwidth=binwidth, color="gray", kde=False, ax=axes[2])
        axes[2].set_ylim(0, 100)
        axes[2].set_xlim(0, max_hist)
        axes[2].set_title("Irrelevant Trait A\nSimulation Results")

    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    handles = []
    for b in behaviors:
        handles.append(Line2D([0], [0], color=COLOR_DICT[b], marker='o', linestyle='-', linewidth=2, markersize=6))
        handles.append(Line2D([0], [0], color=COLOR_DICT[b], linestyle='--', linewidth=2))

    labels = []
    for b in behaviors:
        labels.append(f"{b} (mean)")
        labels.append(f"{b} (2.5% and 97.5%)")

    legend_placement = (0.7, 0.3) if ncols==3 else (1.0, 0.3)
    fig.legend(handles=handles, labels=labels, ncol=2, bbox_to_anchor=legend_placement)
    fig.tight_layout(pad=1.0)
    plt.subplots_adjust(bottom=0.4)
    return fig

def visualize_vs_fitness_irrelevant(filename, binwidth, group_size, mixed):
    df = read_data(csvpath="/../data/" + filename)
    learning_df = compute_behavior_summary(df, ["alpha", "beta", "gamma", "delta"])
    final_ext = compute_final_externalizers(df)
    final_A = extract_benchmark_list(group_size=group_size, mixed=mixed)
    return plot_shared(learning_df, final_ext, final_A, group_size, binwidth, ncols=3)

def visualize_two_panels(filename, max_hist, binwidth):
    df = read_data(csvpath="/../data/" + filename)
    learning_df = compute_behavior_summary(df, ["alpha", "beta", "gamma", "delta"])
    final_ext = compute_final_externalizers(df)
    return plot_shared(learning_df, final_ext, None, max_hist, binwidth, ncols=2)


if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.realpath(__file__))

    fig6 = visualize_vs_fitness_irrelevant("ABM_base_simulation.csv", binwidth=10, group_size=100, mixed=False)
    fig6.savefig(file_dir+"/../plots/fig6.png")

    fig7 = visualize_two_panels("ABM_mixed_learning_mechanism_simulation.csv", max_hist=100, binwidth=10)
    fig7.savefig(file_dir+"/../plots/fig7.png")

    fig8 = visualize_vs_fitness_irrelevant("ABM_pop_size_12_simulation.csv", binwidth=1, group_size=12, mixed=False)
    fig8.savefig(file_dir+"/../plots/fig8.png")

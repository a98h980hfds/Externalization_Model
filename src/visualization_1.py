import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os
import model_1

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
    "externalizing": "darkblue",
    "non-externalizing": "purple"
}
BEHAVIORS = ["alpha", "delta", "beta", "gamma"]
EXT_TRAITS = ["externalizing", "non-externalizing"]

# ------------------------------
# Utility: data reading
# ------------------------------
def read_data(csvpath):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    csv_file = file_dir + csvpath
    df = pd.read_csv(csv_file, dtype=str)
    df = df.loc[3:len(df)]
    index_colnames = ["generation", "learning_step", "game_round", "metric"]
    df.rename(columns=dict(zip(df.columns[:4], index_colnames)), inplace=True)
    for col in df.columns:
        if col != "metric":
            df[col] = pd.to_numeric(df[col])
    df.set_index(index_colnames, inplace=True)
    df.columns = model_1.initialize_dataframe().columns
    return df

# ------------------------------
# Panel 1: partnered share development within a learning step
# ------------------------------
def panel_partnered_share(df, generation=0, learning_step=0, ax=None):
    max_round = int(df.index.get_level_values("game_round").max())
    visibility_offset = {"alpha": 0.008, "beta": 0, "gamma": -0.004, "delta": 0.004}
    vis_df = pd.DataFrame(columns=["game_round", "behavior", "matched_share"])
    for behavior in BEHAVIORS:
        total = df.loc[(generation, learning_step, 0, "shares"), pd.IndexSlice[:, behavior, :]].sum()
        for game_round in range(max_round + 1):
            matched_share = 0
            if total > 0:
                matched_share = (
                    df.loc[(generation, learning_step, game_round, "shares"),
                           pd.IndexSlice[:, behavior, "matched"]].sum() / total
                    + visibility_offset[behavior]
                )
            vis_df.loc[len(vis_df)] = [game_round, behavior, matched_share]

    sns.lineplot(data=vis_df, x="game_round", y="matched_share",
                 hue="behavior", marker='o', palette=COLOR_DICT,
                 ax=ax, markeredgewidth=0)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Partnered Share")
    return ax

# ------------------------------
# Panel 2: accumulated payoff across learning steps
# ------------------------------
def panel_accumulated_payoff(df, generation=0, learning_step=0, ax=None):
    max_round = int(df.index.get_level_values("game_round").max())
    vis_df = pd.DataFrame(columns=["game_round", "behavior", "accumulated_payoff"])
    for behavior in BEHAVIORS:
        accumulated_payoff = 0
        for game_round in range(max_round + 1):
            payoff_matched = df.loc[(generation, learning_step, game_round, "payoffs"),
                                    pd.IndexSlice["non-externalizing", behavior, "matched"]]
            payoff_unmatched = df.loc[(generation, learning_step, game_round, "payoffs"),
                                      pd.IndexSlice["non-externalizing", behavior, "unmatched"]]
            matched_share = df.loc[(generation, learning_step, game_round, "shares"),
                                   pd.IndexSlice[:, behavior, "matched"]].sum()
            total = df.loc[(generation, learning_step, 0, "shares"),
                           pd.IndexSlice[:, behavior, :]].sum()
            matched_share = matched_share / total if total > 0 else 0
            payoff = matched_share * payoff_matched + (1 - matched_share) * payoff_unmatched
            accumulated_payoff += payoff
            vis_df.loc[len(vis_df)] = [game_round, behavior, accumulated_payoff]

    sns.lineplot(data=vis_df, x="game_round", y="accumulated_payoff",
                 hue="behavior", marker='o', palette=COLOR_DICT,
                 ax=ax, markeredgewidth=0)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Accumulated Payoff")
    return ax

# ------------------------------
# Panel 3: learning process over learning steps
# ------------------------------
def panel_learning_process(df, generation=0, ax=None):
    max_step = int(df.index.get_level_values("learning_step").max())
    max_gro = int(df.index.get_level_values("game_round").max())
    vis_df = pd.DataFrame(columns=["learning_step", "behavior", "share"])
    for behavior in BEHAVIORS:
        for step in range(max_step + 1):
            share = df.loc[(generation, step, max_gro, "shares"), pd.IndexSlice[:, behavior, :]].sum()
            vis_df.loc[len(vis_df)] = [step, behavior, share]

    sns.lineplot(data=vis_df, x="learning_step", y="share",
                 hue="behavior", marker='o', palette=COLOR_DICT,
                 ax=ax, markeredgewidth=0)
    ax.set_xlabel("Learning Cycle")
    ax.set_ylabel("Share of Population")
    return ax

# ------------------------------
# Panel 4: natural selection process across generations
# ------------------------------
def panel_natural_selection(df, ax=None):
    max_gen = int(df.index.get_level_values("generation").max())
    nat_df = pd.DataFrame(columns=["generation", "ext_trait", "share"])
    for ext in EXT_TRAITS:
        for gen in range(max_gen + 1):
            share = df.loc[(gen, 0, 0, "shares"), pd.IndexSlice[ext, :, :]].sum()
            nat_df.loc[len(nat_df)] = [gen, ext, share]

    sns.lineplot(data=nat_df, x="generation", y="share",
                 hue="ext_trait", marker='o', palette=COLOR_DICT,
                 ax=ax, markeredgewidth=0)
    ax.set_xlabel("Generational Cycle")
    ax.set_ylabel("Share of Population")
    return ax

# ------------------------------
# Panel 5: end-of-learning-step partnered share across learning steps
# ------------------------------
def panel_end_of_lst_partnered(df, generation=0, ax=None):
    max_step = int(df.index.get_level_values("learning_step").max())
    max_gro = int(df.index.get_level_values("game_round").max())
    vis_df = pd.DataFrame(columns=["learning_step", "behavior", "end_of_step_matched"])

    for behavior in BEHAVIORS:
        for step in range(max_step + 1):
            total_beh = df.loc[
                (generation, step, max_gro, "shares"),
                pd.IndexSlice["non-externalizing", behavior, :]
            ].sum()

            if total_beh > 0:
                end_matched = (
                    df.loc[
                        (generation, step, max_gro, "shares"),
                        pd.IndexSlice["non-externalizing", behavior, "matched"]
                    ] / total_beh
                )
            else:
                end_matched = 0

            vis_df.loc[len(vis_df)] = [step, behavior, end_matched]

    sns.lineplot(
        data=vis_df, x="learning_step", y="end_of_step_matched",
        hue="behavior", marker='o', palette=COLOR_DICT, ax=ax, markeredgewidth=0
    )
    ax.set_xlabel("Learning Cycle")
    ax.set_ylabel("End-of-Step Partnered Share")
    return ax

def visualize_master(panel_configs):
    """
    panel_configs: dict of {panel_name: {"type": str, "params": dict}}
    panel types:
        "partnered_share"
        "accumulated_payoff"
        "learning_process"
        "natural_selection"
        "end_of_lst_partnered"
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, len(panel_configs), figsize=(4 * len(panel_configs), 5))
    if len(panel_configs) == 1:
        axes = [axes]

    panel_functions = {
        "partnered_share": panel_partnered_share,
        "accumulated_payoff": panel_accumulated_payoff,
        "learning_process": panel_learning_process,
        "natural_selection": panel_natural_selection,
        "end_of_lst_partnered": panel_end_of_lst_partnered
    }

    for ax, (key, config) in zip(axes, panel_configs.items()):
        ftype = config["type"]
        params = config.get("params", {})
        panel_functions[ftype](ax=ax, **params)
        if not ftype == "accumulated_payoff":
            ax.set_ylim(-0.1, 1.1)
        ax.set_title(key)
        if ax.get_legend():
            ax.get_legend().remove()

    # unified legend
    handles, labels = [], []
    panel_types = [cfg["type"] for cfg in panel_configs.values()]

    # add ext_trait legend only if natural_selection panel is included
    if "natural_selection" in panel_types:
        handles += [
            Line2D([0], [0], color=COLOR_DICT[e], marker='o',
                   linestyle='-', linewidth=2, markersize=6)
            for e in EXT_TRAITS
        ]
        labels += EXT_TRAITS

    # add behavior legend if any behavior-based panel is included
    behavior_panels = {
        "partnered_share",
        "accumulated_payoff",
        "learning_process",
        "end_of_lst_partnered"
    }
    if any(pt in behavior_panels for pt in panel_types):
        handles += [
            Line2D([0], [0], color=COLOR_DICT[b], marker='o',
                   linestyle='-', linewidth=2, markersize=6)
            for b in BEHAVIORS
        ]
        labels += BEHAVIORS

    fig.legend(
        handles, labels,
        ncol=int(len(handles)/2),
        loc="lower left",
        bbox_to_anchor=(0.1, -0.015),
        frameon=True
    )
    fig.tight_layout(pad=1.0)
    plt.subplots_adjust(bottom=0.28) # give room for legend
    return fig

if __name__ == "__main__":
    # FIG 1: Interaction Process
    df = read_data("/../data/base_model_simulation.csv")
    panels = {
        "Partnered Share": {"type": "partnered_share", "params": {"df": df, "generation": 0, "learning_step": 0}},
        "Accumulated Payoff": {"type": "accumulated_payoff", "params": {"df": df, "generation": 0, "learning_step": 0}},
    }
    fig = visualize_master(panels)
    fig.savefig("../plots/fig1.png")

    # FIG 2: Learning process
    df = read_data("/../data/base_model_simulation.csv")
    panels = {
        "Learning Process": {"type": "learning_process", "params": {"df": df, "generation": 0}},
        "Partnered Share At\nEnd of Learning Step ": {"type": "end_of_lst_partnered", "params": {"df": df, "generation": 0}}
    }
    fig = visualize_master(panels)
    fig.savefig("../plots/fig2.png")

    # FIG 3: Natural Selection Process
    df = read_data("/../data/base_model_simulation.csv")
    panels = {
        "Learning Process\nGeneration 12": {"type": "learning_process", "params": {"df": df, "generation": 12}},
        "Natural Selection Process": {"type": "natural_selection", "params": {"df": df}},
    }
    fig = visualize_master(panels)
    fig.savefig("../plots/fig3.png")

    # FIG 4: Externalization necessary with 4 turns per learning cycle
    df1 = read_data("/../data/PD_gro4_externalizing_population.csv")
    df2 = read_data("/../data/PD_gro4_non_externalizing_population.csv")
    panels = {
        "Learning Process\nExternalizing Population": {"type": "learning_process", "params": {"df": df1, "generation": 0}},
        "Learning Process\nNon-Externalizing Population": {"type": "learning_process", "params": {"df": df2, "generation": 0}},
    }
    fig = visualize_master(panels)
    fig.savefig("../plots/fig4.png")

    # FIG 5: Externalization necessary and adaptive for n_E=3
    df = read_data("/../data/lst_3_gro_7_9610_simulation.csv")
    panels = {
        "Learning Process\nGeneration 1": {"type": "learning_process", "params": {"df": df, "generation": 0}},
        "Natural Selection Process": {"type": "natural_selection", "params": {"df": df}},
        "Learning Process\nGeneration 60": {"type": "learning_process", "params": {"df": df, "generation": 59}},
    }
    fig = visualize_master(panels)
    fig.savefig("../plots/fig5.png")
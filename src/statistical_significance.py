import pandas as pd
import os
import numpy as np
from scipy.stats import mannwhitneyu
from random import shuffle, sample
from numpy.random import choice

COLOR_DICT = {
    "alpha": "green",
    "beta": "red",
    "gamma": "dodgerblue",
    "delta": "orange",
    "externalizers": "darkblue",
    "non-externalizers": "purple"
}

def simulate_benchmark(sample_size, pop_size):
    MODEL_PARAMS = {
        'cc': 2,
        'dc': 3,
        'cd': 0,
        'dd': 1,
        'learning_steps': 15,
        'learning_mechanism': 'mixed',
        'game_rounds': 15,
        'pop_size': pop_size,
        'initial_A': 1,
        'generations': 50
    }

    def new_generation_A(n_A, pop_size):
        row_values = []
        for i in range(pop_size):
            row_values.append(1 if i < n_A else 0)
            row_values.append(choice(["alpha", "beta", "gamma", "delta"], p=[0.25]*4))
            row_values.append(0)
            row_values.append(0)
            row_values.append("unpartnered")
        return row_values

    def initialize_dataframe_A():
        pop_size = MODEL_PARAMS['pop_size']
        cols = [("generation", ""), ("learning_step", ""), ("game_round", "")]
        for i in range(pop_size):
            for attr in ["trait_A", "behavior", "payoff_interaction_process",
                         "payoff_learning_process", "current_partner"]:
                cols.append((i, attr))
        multi_index = pd.MultiIndex.from_tuples(cols)
        df = pd.DataFrame(columns=multi_index)
        df = df.set_index(["generation", "learning_step", "game_round"])
        df.loc[(0, 0, 0)] = new_generation_A(MODEL_PARAMS['initial_A'], pop_size)
        return df

    def find_partner(df, g, l, r):
        idx = (g, l, r)
        unpartnered = [i for i in range(MODEL_PARAMS['pop_size'])
                       if df.loc[idx, (i, "current_partner")] == "unpartnered"]
        shuffle(unpartnered)
        for pair in range(0, len(unpartnered), 2):
            df.loc[idx, (unpartnered[pair], "current_partner")] = unpartnered[pair + 1]
            df.loc[idx, (unpartnered[pair + 1], "current_partner")] = unpartnered[pair]

    def play_game(df, g, l, r):
        idx = (g, l, r)
        next_idx = (g, l, r + 1)
        must_play = [i for i in range(MODEL_PARAMS['pop_size'])]
        while must_play:
            a = must_play[0]
            p = df.loc[idx, (a, "current_partner")]
            must_play.remove(a)
            must_play.remove(p)
            ab = df.loc[idx, (a, "behavior")]
            pb = df.loc[idx, (p, "behavior")]
            if ab in ["alpha", "gamma"]:
                if pb in ["alpha", "gamma"]:
                    payoff, ppayoff = MODEL_PARAMS['cc'], MODEL_PARAMS['cc']
                else:
                    payoff, ppayoff = MODEL_PARAMS['cd'], MODEL_PARAMS['dc']
            else:
                if pb in ["alpha", "gamma"]:
                    payoff, ppayoff = MODEL_PARAMS['dc'], MODEL_PARAMS['cd']
                else:
                    payoff, ppayoff = MODEL_PARAMS['dd'], MODEL_PARAMS['dd']
            df.loc[idx, (a, "payoff_interaction_process")] += payoff
            df.loc[idx, (p, "payoff_interaction_process")] += ppayoff
            if r != MODEL_PARAMS['game_rounds'] - 1:
                df.loc[next_idx, pd.IndexSlice[a, :]] = df.loc[idx, pd.IndexSlice[a, :]]
                df.loc[next_idx, pd.IndexSlice[p, :]] = df.loc[idx, pd.IndexSlice[p, :]]
                stable = (
                    (ab == "alpha" and pb == "alpha") or
                    (ab == "beta" and pb == "gamma") or
                    (ab == "gamma" and pb == "beta") or
                    (ab == "delta" and pb == "delta")
                )
                if not stable:
                    df.loc[next_idx, (a, "current_partner")] = "unpartnered"
                    df.loc[next_idx, (p, "current_partner")] = "unpartnered"

    def update_behavior(df, g, l):
        idx = (g, l, MODEL_PARAMS['game_rounds'] - 1)
        next_idx = (g, l + 1, 0)
        pop_size = MODEL_PARAMS['pop_size']
        ranking = [(df.loc[idx, (i, "payoff_interaction_process")],
                    df.loc[idx, (i, "behavior")]) for i in range(pop_size)]
        ranking.sort(reverse=True, key=lambda x: x[0])
        mech = MODEL_PARAMS["learning_mechanism"]
        if mech == "mixed":
            mech = choice(["success", "frequency", "source"])
        if mech == "success":
            probs = [len(ranking) - i for i in range(len(ranking))]
            probs = [i / sum(probs) for i in probs]
        elif mech == "frequency":
            probs = [1 / len(ranking)] * len(ranking)
        else:
            probs = [0] * len(ranking)
            for s in sample(range(len(ranking)), 4):
                probs[s] = 1/4
        for i in range(pop_size):
            df.loc[next_idx, (i, "behavior")] = choice([r[1] for r in ranking], p=probs)
            df.loc[next_idx, (i, "payoff_learning_process")] = (
                df.loc[idx, (i, "payoff_learning_process")] +
                df.loc[idx, (i, "payoff_interaction_process")]
            )
            df.loc[next_idx, (i, "payoff_interaction_process")] = 0
            df.loc[next_idx, (i, "current_partner")] = "unpartnered"
            df.loc[next_idx, (i, "trait_A")] = df.loc[idx, (i, "trait_A")]

    def update_trait_A(df, g):
        idx = (g, MODEL_PARAMS['learning_steps'] - 1, MODEL_PARAMS['game_rounds'] - 1)
        next_idx = (g + 1, 0, 0)
        pop_size = MODEL_PARAMS['pop_size']
        order = list(range(pop_size))
        shuffle(order)
        ranking = [(df.loc[idx, (i, "payoff_learning_process")],
                    df.loc[idx, (i, "trait_A")]) for i in order]
        ranking.sort(reverse=True, key=lambda x: x[0])
        top_half_A = sum(i[1] for i in ranking[: pop_size // 2])
        new_A = top_half_A * 2
        df.loc[next_idx] = new_generation_A(new_A, pop_size)
        converged = df.loc[next_idx, pd.IndexSlice[:, "trait_A"]].sum() in [0, pop_size]
        return converged

    results = []
    for run in range(sample_size):
        print(f"run {run} of {sample_size}")
        df = initialize_dataframe_A()
        for g in range(MODEL_PARAMS['generations']):
            for l in range(MODEL_PARAMS['learning_steps']):
                for r in range(MODEL_PARAMS['game_rounds']):
                    find_partner(df, g, l, r)
                    play_game(df, g, l, r)
                update_behavior(df, g, l)
            conv = update_trait_A(df, g)
            if conv:
                break

        final_A = int(df.loc[(g + 1, 0, 0), pd.IndexSlice[:, "trait_A"]].sum())
        results.append(final_A)
    return results

def read_data(csvpath="/../data/ABM_base_simulation.csv"):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    csv_file = file_dir + csvpath
    df = pd.read_csv(csv_file, dtype=str)
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
    for i in range(int(len(df.columns)/2)):
        for attr in ["externalization", "behavior"]:
            cols.append((i, attr))
    df.columns = pd.MultiIndex.from_tuples(cols)
    return df

def test_for_significance(filename, pop_size, benchmark):
    df = read_data(csvpath="/../data/" + filename)
    max_sim = int(df.reset_index()["simulation"].max())
    simulation_results = []
    for simulation in range(max_sim + 1):
        max_gen = int(df.loc[pd.IndexSlice[simulation, :, :]].reset_index()["generation"].max())
        n_ext = df.loc[(simulation, max_gen, 0), pd.IndexSlice[:, "externalization"]].sum()
        simulation_results.append(int(n_ext))
    export_txt = f"Simulation results: {simulation_results}\n"
    export_txt += f"Mean of simulation results: {np.mean(simulation_results)}\n"
    export_txt += f"Number of 0: {np.sum(np.array(simulation_results) == 0)}\n"
    export_txt += f"Number of {pop_size}: {np.sum(np.array(simulation_results) == pop_size)}\n"
    export_txt += f"Benchmark results: {benchmark}\n"
    export_txt += f"Mean of benchmark: {np.mean(benchmark)}\n"
    export_txt += f"Number of 0: {np.sum(np.array(benchmark) == 0)}\n"
    export_txt += f"Number of {pop_size}: {np.sum(np.array(benchmark) == pop_size)}\n"
    stat, p_val = mannwhitneyu(simulation_results, benchmark, alternative='greater')
    export_txt += f"Mann-Whitney U: {stat}, p-value: {p_val}\n"
    return export_txt


if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.realpath(__file__))

    benchmark100 = simulate_benchmark(sample_size=1000, pop_size=100)
    benchmark12 = simulate_benchmark(sample_size=1000, pop_size=12)
    benchmark8 = simulate_benchmark(sample_size=1000, pop_size=8)
    
    export_txt = "Significance Test Results population size 100:\n"
    export_txt += test_for_significance("ABM_base_simulation.csv", pop_size=100, benchmark=benchmark100)
    
    export_txt += "\nSignificance Test Results population size 100 mixed:\n"
    export_txt += test_for_significance("ABM_mixed_learning_mechanism_simulation.csv", pop_size=100, benchmark=benchmark100)
    
    export_txt += "\nSignificance Test Results population size 12:\n"
    export_txt += test_for_significance("ABM_pop_size_12_simulation.csv", pop_size=12, benchmark=benchmark12)
    
    export_txt += "\nSignificance Test Results population size 12 mixed learning:\n"
    export_txt += test_for_significance("ABM_pop_size_12_mixed_learning_simulation.csv", pop_size=12, benchmark=benchmark12)

    export_txt += "\nSignificance Test Results population size 8\n"
    export_txt += test_for_significance("ABM_pop_size_8_simulation.csv", pop_size=8, benchmark=benchmark8)

    export_txt += "\nSignificance Test Results population size 8 mixed learning\n"
    export_txt += test_for_significance("ABM_pop_size_8_mixed_learning_simulation.csv", pop_size=8, benchmark=benchmark8)

    with open(file_dir+"/../data/significance_test_results.txt", "w") as f:
        f.write(export_txt)

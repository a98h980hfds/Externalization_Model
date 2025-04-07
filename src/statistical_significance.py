
import pandas as pd
import os
import numpy as np
from scipy.stats import mannwhitneyu

COLOR_DICT = {
    "alpha": "green",
    "beta": "red", 
    "gamma": "dodgerblue",
    "delta": "orange",
    "externalizers": "darkblue",
    "non-externalizers": "purple"
}

def simulate_benchmark(sample_size, pop_size):
    results = []
    for i in range(sample_size):
        population = [1]+[0]*(pop_size-1)
        np.random.shuffle(population)
        for gen in range(50):
            population = population[:int(pop_size/2)]*2
            np.random.shuffle(population)
            if sum(population) == 0 or sum(population) == pop_size:
                break
        results.append(sum(population))
    return results

def read_data(csvpath="/../data/ABM_base_simulation.csv"):
    # Read the file
    file_dir = os.path.dirname(os.path.realpath(__file__))
    csv_file = file_dir+csvpath
    df = pd.read_csv(csv_file, dtype=str)

    # Convert to numeric
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
    df = read_data(csvpath="/../data/"+filename)
    
    max_sim = int(df.reset_index()["simulation"].max())
    simulation_results = []

    for simulation in range(max_sim+1):
        max_gen_in_sim = int(df.loc[pd.IndexSlice[simulation, :, :]].reset_index()["generation"].max())
        number_ext = df.loc[(simulation, max_gen_in_sim, 0), pd.IndexSlice[:, "externalization"]].sum()
        simulation_results.append(int(number_ext))
    export_txt = f"Simulation results: {simulation_results}\n"
    export_txt += f"Mean of simulation results: {np.mean(simulation_results)}\n"
    export_txt += f"Number of 0 in simulation results: {np.sum(np.array(simulation_results) == 0)}\n"
    export_txt += f"Number of {pop_size} in simulation results: {np.sum(np.array(simulation_results) == pop_size)}\n"

    export_txt += f"Benchmark results: {benchmark}\n"
    export_txt += f"Mean of benchmark results: {np.mean(benchmark)}\n"
    export_txt += f"Number of 0 in benchmark results: {np.sum(np.array(benchmark) == 0)}\n"
    export_txt += f"Number of {pop_size} in benchmark results: {np.sum(np.array(benchmark) == pop_size)}\n"

    # One-sided Mann-Whitney U test to test if the simulation results are greater than the benchmark
    statistic, p_value = mannwhitneyu(simulation_results, benchmark, alternative='greater')
    export_txt += f"Mann-Whitney U statistic: {statistic}, p-value: {p_value}\n"
    return export_txt


if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.realpath(__file__))

    benchmark100 = simulate_benchmark(sample_size=1000, pop_size=100)
    benchmark12 = simulate_benchmark(sample_size=1000, pop_size=12)
    benchmark10 = simulate_benchmark(sample_size=1000, pop_size=10)
    benchmark8 = simulate_benchmark(sample_size=1000, pop_size=8)
    benchmark6 = simulate_benchmark(sample_size=1000, pop_size=6)

    export_txt = "Significance Test Results Base ABM:\n"
    export_txt += test_for_significance("ABM_base_simulation.csv", pop_size=100, benchmark=benchmark100)
    
    export_txt += "\nSignificance Test Results for mixed learning mechanism:\n"
    export_txt += test_for_significance("ABM_mixed_learning_mechanism_simulation.csv", pop_size=100, benchmark=benchmark100)
    
    export_txt += "\nSignificance Test Results mixed learning mechanism Hawk Dove:\n"
    export_txt += test_for_significance("ABM_mixed_learning_mechanism_simulation_HD.csv", pop_size=100, benchmark=benchmark100)
    
    export_txt += "\nSignificance Test Results mixed learning mechanism Stag Hunt:\n"
    export_txt += test_for_significance("ABM_mixed_learning_mechanism_simulation_SH.csv", pop_size=100, benchmark=benchmark100)
    
    export_txt += "\nSignificance Test Results population size 12:\n"
    export_txt += test_for_significance("ABM_pop_size_12_simulation.csv", pop_size=12, benchmark=benchmark12)
    
    export_txt += "\nSignificance Test Results population size 12 mixed learning:\n"
    export_txt += test_for_significance("ABM_pop_size_12_mixed_learning_simulation.csv", pop_size=12, benchmark=benchmark12)

    export_txt += "\nSignificance Test Results population size 8 Stag Hunt\n"
    export_txt += test_for_significance("ABM_pop_size_8_simulation_SH.csv", pop_size=8, benchmark=benchmark8)

    export_txt += "\nSignificance Test Results population size 8 Stag Hunt mixed learning\n"
    export_txt += test_for_significance("ABM_pop_size_8_mixed_learning_simulation_SH.csv", pop_size=8, benchmark=benchmark8)

    export_txt += "\nSignificance Test Results population size 12 Hawk Dove\n"
    export_txt += test_for_significance("ABM_pop_size_12_simulation_HD.csv", pop_size=12, benchmark=benchmark12)

    export_txt += "\nSignificance Test Results population size 12 Hawk Dove mixed learning\n"
    export_txt += test_for_significance("ABM_pop_size_12_mixed_learning_simulation_HD.csv", pop_size=12, benchmark=benchmark12)

    export_txt += "\nSignificance Test Results population size 8\n"
    export_txt += test_for_significance("ABM_pop_size_8_simulation.csv", pop_size=8, benchmark=benchmark8)

    with open(file_dir+"/../data/significance_test_results.txt", "w") as f:
        f.write(export_txt)

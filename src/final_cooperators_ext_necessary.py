import pandas as pd
import os
import numpy as np

def read_data(csvpath):
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

def compare_final_coop_ext_vs_non(dfs, pop_size):
    max_lst = int(dfs["ext"].reset_index()["learning_step"].max())

    counts = {"ext": {s: [0]*100 for s in {"cooperative", "defective"}},
              "non": {s: [0]*100 for s in {"cooperative", "defective"}}}
    averages = {"ext": {s: 0 for s in {"cooperative", "defective"}},
              "non": {s: 0 for s in {"cooperative", "defective"}}}
    for x in ["ext", "non"]:
        for sim in range(100):
            row = dfs[x].loc[(sim, 0, max_lst), pd.IndexSlice[:, "behavior"]]
            for agent in range(pop_size):
                behavior = row[(agent, "behavior")]
                if behavior in ["alpha", "gamma"]:
                    counts[x]["cooperative"][sim] += 1
                else:
                    counts[x]["defective"][sim]+=1
        averages[x]["cooperative"] = float(np.mean(counts[x]["cooperative"]))
        averages[x]["defective"] = float(np.mean(counts[x]["defective"]))
    return averages

def main():
    export_txt = ""

    dfs = {"ext": read_data("/../data/ABM_lst_3_gro_7_9610_ext.csv"),
           "non": read_data("/../data/ABM_lst_3_gro_7_9610_non.csv")}
    avg = compare_final_coop_ext_vs_non(dfs, 100)
    export_txt += "Comparison for 3 learning cycles, 7 turns, PD with 9 6 1 0\n"
    export_txt += str(avg)
    export_txt += "\n\n"
    
    dfs = {"ext": read_data("/../data/ABM_mixed_lst_3_gro_7_9610_ext.csv"),
           "non": read_data("/../data/ABM_mixed_lst_3_gro_7_9610_non.csv")}
    avg = compare_final_coop_ext_vs_non(dfs, 100)
    export_txt += "Comparison for mixed learning, 3 learning cycles, 7 turns, PD with 9 6 1 0\n"
    export_txt += str(avg)
    export_txt += "\n\n"

    dfs = {"ext": read_data("/../data/ABM_pop_size_12_lst_3_gro_7_9610_ext.csv"),
           "non": read_data("/../data/ABM_pop_size_12_lst_3_gro_7_9610_non.csv")}
    avg = compare_final_coop_ext_vs_non(dfs, 12)
    export_txt += "Comparison for population size 12, 3 learning cycles, 7 turns, PD with 9 6 1 0\n"
    export_txt += str(avg)
    export_txt += "\n\n"

    file_dir = os.path.dirname(os.path.realpath(__file__))
    with open(file_dir+"/../data/ext_vs_non_coop_ABM_results.txt", "w") as f:
        f.write(export_txt)

if __name__ == "__main__":
    main()
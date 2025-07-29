import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
plt.rcParams["font.family"] = "Hiragino Sans"

result_files = glob.glob("output/*/result.csv")

print(result_files)

def read_result(path):
    df = pd.read_csv(path)
    return df

def analyze_process(df, target):
    df[f"{target}_round"] = df[f"{target}"].round()
    summary_df = df.groupby(f"{target}_round")["逆転"].mean().reset_index()
    summary_df["逆転"] = summary_df["逆転"] * 100
    return summary_df

def single_plot_process(summary_df, target, path):
    fig, ax = plt.subplots()
    ax.plot(summary_df[f"{target}_round"], summary_df["逆転"])
    ax.set_title(f"{target}とB勝率の関係")
    ax.set_xlabel(f"{target}(%)")
    ax.set_ylabel(f"B勝率(%)")
    plt.savefig(f"{os.path.dirname(path)}/{target}_curve.png")

def main():
    for path in sorted (result_files):
        df = read_result(path)
        for target in df.columns[ : 5]:
            summary_df = analyze_process(df, target)
            single_plot_process(summary_df, target, path)

main()
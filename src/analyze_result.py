import os
import glob
from typing import List

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Hiragino Sans"


def read_result(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def analyze_process(df: pd.DataFrame, target: str) -> pd.DataFrame:
    df[f"{target}_round"] = df[f"{target}"].round()
    summary_df = df.groupby(f"{target}_round")["逆転"].mean().reset_index()
    summary_df["逆転"] = summary_df["逆転"] * 100
    return summary_df


def single_plot_process(summary_df: pd.DataFrame, target: str, path: str) -> None:
    fig, ax = plt.subplots()
    ax.plot(summary_df[f"{target}_round"], summary_df["逆転"])
    ax.set_title(f"{target}とB勝率の関係")
    ax.set_xlabel(f"{target}(%)")
    ax.set_ylabel("B勝率(%)")
    plt.savefig(f"{os.path.dirname(path)}/{target}_curve.png")


def main() -> None:
    result_files: List[str] = sorted(glob.glob("output/*/result.csv"))
    if not result_files:
        print("結果ファイルが見つかりませんでした。先にシミュレーションを実行してください。")
        return

    for path in result_files:
        df = read_result(path)
        # 先頭5列はパラメータ列（voteratio, NtoA_ratio, ...）を想定
        for target in df.columns[:5]:
            summary_df = analyze_process(df, target)
            single_plot_process(summary_df, target, path)


if __name__ == "__main__":
    main()

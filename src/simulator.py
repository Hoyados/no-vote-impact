import os
import time
import numpy as np
import pandas as pd
from logging import getLogger, StreamHandler, FileHandler, INFO, ERROR, DEBUG, Formatter
import argparse
import yaml
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
from tqdm import trange
from pathlib import Path
plt.rcParams["font.family"] = "Hiragino Sans"

logger = getLogger(__name__)
logger.setLevel(DEBUG)  # ロガー全体の出力レベルを設定

# 出力先1: コンソール
console_handler = StreamHandler()
console_handler.setLevel(ERROR)
console_handler.setFormatter(Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

# 出力先2: ファイル
file_handler = FileHandler("log.txt", encoding="utf-8")
file_handler.setLevel(INFO)
file_handler.setFormatter(Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

# ハンドラ追加
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ルートロガーへの伝播を防ぐ
logger.propagate = False

parser = argparse.ArgumentParser(description = "簡易的な投票シミュレータです")

parser.add_argument("--config", default = "input/default_conditions.yml", help = "設定ファイル（yml形式、含拡張子、デフォルト: input/default_conditions.yml）")
parser.add_argument("--loop", action = "store_true")
parser.add_argument("-l", "--loop_number", default = 1000000, help = "試行回数（デフォルト: 1000000）")

args = parser.parse_args()

logger.info(f"処理開始(ファイル名: {os.path.basename(args.config)})")

logger.info(f"読み込もうとしているファイルパス: {args.config}")
logger.info(f"ベクトル演算？: {args.loop}")
logger.debug(f"現在のディレクトリ: {os.getcwd()}")
logger.debug(f"ファイルの存在チェック: {os.path.exists(args.config)}")

output_dir = Path(f"output/{os.path.splitext(os.path.basename(args.config))[0]}")
output_dir.mkdir(parents=True, exist_ok=True)

start_time = time.perf_counter()

def fileload():
    '''
    ファイル読み込みの関数。
    引数: filename(ファイル名称、拡張子抜き)
    戻り値: dataframe
    '''   
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error("読み込み失敗", e)
        exit()

def initial_condition(config):
    '''
    条件指定csvから初期条件を抽出する関数
    引数: dataframe
    戻り値: 変換後の各パラメータ
    '''
    initialratio_A = config["投票率"] * config["A得票率"] / 10000
    initialratio_B = config["投票率"] * config["B得票率"] / 10000
    initialratio_N = (100 - config["投票率"]) / 100 + (config["投票率"] / 100 - (initialratio_A + initialratio_B))
    if initialratio_A + initialratio_B + initialratio_N > 1.001 or initialratio_A + initialratio_B + initialratio_N < 0.999:
        logger.error("A党、B党、他党or無投票の割合の合計が1になりません")
        exit()
    elif initialratio_A < 0 or initialratio_B < 0 or initialratio_N < 0:
        logger.error("A党、B党、他党or無投票の割合が0未満です")
        exit()
    else:
        logger.info("A党得票率（%）, B党得票率（%）, 他政党or無投票率（%）, 合計（%）")
        logger.info(f"{initialratio_A * 100, initialratio_B * 100, initialratio_N * 100, (initialratio_A + initialratio_B + initialratio_N) * 100}")
        return initialratio_N, initialratio_A, initialratio_B
    
def random_ranges(config):
    '''
    条件指定csvからランダム範囲を作成する関数
    引数: dataframe
    戻り値: 変換後の各パラメータ
    '''
    voteratio_min = config["voteratio"]["min"] / 100
    NtoA_ratio_min = config["NtoA_ratio"]["min"] / 100
    AtoB_ratio_min = config["AtoB_ratio"]["min"] / 100
    BtoA_ratio_min = config["BtoA_ratio"]["min"] / 100
    voteratio_max = config["voteratio"]["max"] / 100
    NtoA_ratio_max = config["NtoA_ratio"]["max"] / 100
    AtoB_ratio_max = config["AtoB_ratio"]["max"] / 100
    BtoA_ratio_max = config["BtoA_ratio"]["max"] / 100

    if not (0 <= voteratio_min <= voteratio_max <= 1):
        logger.error("voteratio の最小値と最大値の設定に誤りがあります")
        exit()
    if not (0 <= NtoA_ratio_min <= NtoA_ratio_max <= 1):
        logger.error("NtoA_ratio の最小値と最大値の設定に誤りがあります")
        exit()
    if not (0 <= AtoB_ratio_min <= AtoB_ratio_max <= 1):
        logger.error("AtoB_ratio の最小値と最大値の設定に誤りがあります")
        exit()
    if not (0 <= BtoA_ratio_min <= BtoA_ratio_max <= 1):
        logger.error("BtoA_ratio の最小値と最大値の設定に誤りがあります")
        exit()

    logger.info("ランダムの取りうる幅（％）")
    logger.info(f"voteratio:, {voteratio_min * 100}, -, {voteratio_max * 100}")
    logger.info(f"NtoA_ratio:, {NtoA_ratio_min * 100}, -, {NtoA_ratio_max * 100}")
    logger.info(f"AtoB_ratio:, {AtoB_ratio_min * 100}, -, {AtoB_ratio_max * 100}")
    logger.info(f"BtoA_ratio:, {BtoA_ratio_min * 100}, -, {BtoA_ratio_max * 100}")
    
    return voteratio_min, voteratio_max, NtoA_ratio_min, NtoA_ratio_max, AtoB_ratio_min, AtoB_ratio_max, BtoA_ratio_min, BtoA_ratio_max

# ループ回数の設定
def loop_setting():
    loop_number = args.loop_number
    try:
        loop_number = int(loop_number)
        if loop_number <= 0:
            logger.error("正の整数で入力して下さい")
            exit()
        else:
            return loop_number
    except:    
        logger.error("半角数字で入力して下さい")
        exit()

# 乱数生成
def randomized_vector (min, max, loop_number):
    value = np.random.uniform(min, max, size = loop_number)
    return value

def randomized (min, max):
    value = np.random.uniform(min, max)
    return value

# シミュレーション
def simulate_once (initialratio_N, initialratio_A, initialratio_B, voteratio, NtoA_ratio, AtoB_ratio, BtoA_ratio):
    new_voter_NtoA = initialratio_N * voteratio * NtoA_ratio
    new_voter_AtoB = initialratio_A * AtoB_ratio
    new_voter_NtoB = initialratio_N * voteratio * (1 - NtoA_ratio)
    new_voter_BtoA = initialratio_B * BtoA_ratio

    new_A_ratio = initialratio_A - new_voter_AtoB + new_voter_NtoA + new_voter_BtoA
    new_B_ratio = initialratio_B - new_voter_BtoA + new_voter_NtoB + new_voter_AtoB

    winner_flag = 1 if new_B_ratio > new_A_ratio else 0

    return {
        "voteratio": round(voteratio * 100, 2),
        "NtoA_ratio": round(NtoA_ratio * 100, 2),
        "NtoB_ratio": round((1 - NtoA_ratio) * 100, 2),
        "AtoB_ratio": round(AtoB_ratio * 100, 2),
        "BtoA_ratio": round(BtoA_ratio * 100, 2),
        "A得票率": round(new_A_ratio / (new_A_ratio + new_B_ratio) * 100, 2),
        "B得票率": round(new_B_ratio / (new_A_ratio + new_B_ratio) * 100, 2),
        "逆転": winner_flag,
        "投票率": round((new_A_ratio + new_B_ratio) * 100, 2)
    }

def sim_loop (loop_number, ranges, initial):
    results = []
    for i in trange(loop_number, desc="シミュレーション中", mininterval = 0.1):
        voteratio = randomized(*ranges["voteratio"])
        NtoA_ratio = randomized(*ranges["NtoA_ratio"])
        AtoB_ratio = randomized(*ranges["AtoB_ratio"])
        BtoA_ratio = randomized(*ranges["BtoA_ratio"])
        result = simulate_once(*initial, voteratio, NtoA_ratio, AtoB_ratio, BtoA_ratio)
        results.append(result)
    return results

def simulate_vectorized(loop_number, ranges, initialratio_N, initialratio_A, initialratio_B):
    voteratio = randomized_vector(*ranges["voteratio"], loop_number)
    NtoA_ratio = randomized_vector(*ranges["NtoA_ratio"], loop_number)
    AtoB_ratio = randomized_vector(*ranges["AtoB_ratio"], loop_number)
    BtoA_ratio = randomized_vector(*ranges["BtoA_ratio"], loop_number)

    new_voter_NtoA = initialratio_N * voteratio * NtoA_ratio
    new_voter_AtoB = initialratio_A * AtoB_ratio
    new_voter_NtoB = initialratio_N * voteratio * (1 - NtoA_ratio)
    new_voter_BtoA = initialratio_B * BtoA_ratio

    new_A_ratio = initialratio_A - new_voter_AtoB + new_voter_NtoA + new_voter_BtoA
    new_B_ratio = initialratio_B - new_voter_BtoA + new_voter_NtoB + new_voter_AtoB

    winner_flag = (new_B_ratio > new_A_ratio).astype(int)  # True→1, False→0

    result = {
        "voteratio": np.round(voteratio * 100, 2),
        "NtoA_ratio": np.round(NtoA_ratio * 100, 2),
        "NtoB_ratio": np.round((1 - NtoA_ratio) * 100, 2),
        "AtoB_ratio": np.round(AtoB_ratio * 100, 2),
        "BtoA_ratio": np.round(BtoA_ratio * 100, 2),
        "A得票率": np.round(new_A_ratio / (new_A_ratio + new_B_ratio) * 100, 2),
        "B得票率": np.round(new_B_ratio / (new_A_ratio + new_B_ratio) * 100, 2),
        "逆転": winner_flag,
        "投票率": np.round((new_A_ratio + new_B_ratio) * 100, 2)
    }
    
    return result

def summarize_result (df, initial):
    summary_df = pd.DataFrame(columns=[
        "B勝率(%)", "平均投票率(%)", "最高投票率(%)", "最低投票率(%)",
        "平均A得票率(%)", "最高A得票率(%)", "最低A得票率(%)",
        "平均B得票率(%)", "最高B得票率(%)", "最低B得票率(%)"
        ])
    summary_df.loc[0] = [
    round(df["逆転"].mean() * 100, 2),
    round(df["投票率"].mean(), 2),
    round(df["投票率"].max(), 2),
    round(df["投票率"].min(), 2),
    round(df["A得票率"].mean(), 2),
    round(df["A得票率"].max(), 2),
    round(df["A得票率"].min(), 2),
    round(df["B得票率"].mean(), 2),
    round(df["B得票率"].max(), 2),
    round(df["B得票率"].min(), 2)
    ]
    summary_df.to_csv(f"output/{os.path.splitext(os.path.basename(args.config))[0]}/summary.csv", index = False)
    logger.info("\n" + str(summary_df.iloc[0]))

    fig, axs = plt.subplots(1,2)
    axs[0].errorbar(x = "投票率(%)", y = summary_df.loc[0]["平均投票率(%)"],
                    yerr = [[
                            summary_df.loc[0]["平均投票率(%)"] - summary_df.loc[0]["最低投票率(%)"]
                            ],
                            [
                            summary_df.loc[0]["最高投票率(%)"] - summary_df.loc[0]["平均投票率(%)"]
                            ]],
                            fmt = "o", capsize = 10)
    axs[0].plot(["投票率(%)"], [100 - initial[0] * 100], marker = "o")
    axs[0].set_ylim(0, 100)
    axs[1].errorbar(x = ["A得票率(%)", "B得票率(%)"], y = [summary_df.loc[0]["平均A得票率(%)"], summary_df.loc[0]["平均B得票率(%)"]],
                    yerr = [[
                            summary_df.loc[0]["平均A得票率(%)"] - summary_df.loc[0]["最低A得票率(%)"],
                            summary_df.loc[0]["平均B得票率(%)"] - summary_df.loc[0]["最低B得票率(%)"]
                            ],
                            [
                            summary_df.loc[0]["最高A得票率(%)"] - summary_df.loc[0]["平均A得票率(%)"],
                            summary_df.loc[0]["最高B得票率(%)"] - summary_df.loc[0]["平均B得票率(%)"]
                            ]],
                            fmt = "o", capsize = 10)
    axs[1].plot(["A得票率(%)"], [initial[1] * 100 / (initial[1] + initial[2])], marker = "o")
    axs[1].plot(["B得票率(%)"], [initial[2] * 100 / (initial[1] + initial[2])], marker = "o")
    axs[1].set_ylim(0, 100)
    plt.savefig(f"output/{os.path.splitext(os.path.basename(args.config))[0]}/summary.png")
    # plt.show()

def draw_convergence (df, loop_number, reversed_rate):
    '''
    収束関数を描画する関数。
    引数: 結果dataframe, loop_number, reverse_rate
    戻り値: なし
    出力: 描画したグラフ
    '''
    cumulative_winrate = []
    win_count = 0
    for i, flag in enumerate(df["逆転"], start=1):
        win_count += flag
        cumulative_winrate.append(win_count / i)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, loop_number + 1), [x * 100 for x in cumulative_winrate])
    plt.axhline(reversed_rate * 100, color='red', linestyle='--', linewidth=1)
    plt.text(loop_number * 0.9, reversed_rate * 100 * 1.1, f"{round(reversed_rate * 100, 2)}%", color = 'red', va = "center")
    plt.xlabel("試行回数")
    plt.ylabel("B勝率（%）")
    plt.title("収束曲線：B勝率の推移")
    plt.grid(True)
    plt.savefig(f"output/{os.path.splitext(os.path.basename(args.config))[0]}/convergence_curve.png")
    # plt.show()

def draw_convergence_for_vector (winner_flag, loop_number, reversed_rate):
    cumulative_winrate = np.cumsum(winner_flag) / np.arange(1, loop_number + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, loop_number + 1), [x * 100 for x in cumulative_winrate])
    plt.axhline(reversed_rate * 100, color='red', linestyle='--', linewidth=1)
    plt.text(loop_number * 0.9, reversed_rate * 100 * 1.1, f"{round(reversed_rate * 100, 2)}%", color = 'red', va = "center")
    plt.xlabel("試行回数")
    plt.ylabel("B勝率（%）")
    plt.title("収束曲線：B勝率の推移")
    plt.grid(True)
    plt.savefig(f"output/{os.path.splitext(os.path.basename(args.config))[0]}/convergence_curve.png")

def main():
    config = fileload()
    initialratio_N, initialratio_A, initialratio_B = initial_condition(config)
    initial = (initialratio_N, initialratio_A, initialratio_B)
    voteratio_min, voteratio_max, NtoA_ratio_min, NtoA_ratio_max, AtoB_ratio_min, AtoB_ratio_max, BtoA_ratio_min, BtoA_ratio_max = random_ranges(config)
    ranges = {
        "voteratio": (voteratio_min, voteratio_max),
        "NtoA_ratio": (NtoA_ratio_min, NtoA_ratio_max),
        "AtoB_ratio": (AtoB_ratio_min, AtoB_ratio_max),
        "BtoA_ratio": (BtoA_ratio_min, BtoA_ratio_max),
    }

    loop_number = loop_setting()

    result_df = pd.DataFrame(columns=[
        "voteratio", "NtoA_ratio", "NtoB_ratio",
        "AtoB_ratio", "BtoA_ratio", "A得票率",
        "B得票率", "逆転", "投票率"
    ])
    
    if args.loop == False:
        result = simulate_vectorized(loop_number, ranges, initialratio_N, initialratio_A, initialratio_B)
    else:
        result = sim_loop(loop_number, ranges, initial)
  
    result_df = pd.DataFrame(result)
    result_df.to_csv(f"output/{os.path.splitext(os.path.basename(args.config))[0]}/result.csv", index = False)
    reversed_rate = result_df["逆転"].mean()
    logger.info(f"B党の勝率: {round(reversed_rate * 100, 2)}%")

    summarize_result (result_df, initial)

    if args.loop == False:
        draw_convergence_for_vector (result_df["逆転"], loop_number, reversed_rate)
    else:
        draw_convergence (result_df, loop_number, reversed_rate)
    
    
main()

end_time = time.perf_counter()
logger.info(f"処理時間: {round(end_time - start_time, 2)}")

logger.info(f"処理終了(ファイル名: {os.path.basename(args.config)})")
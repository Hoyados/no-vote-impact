import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
from tqdm import trange
plt.rcParams["font.family"] = "Hiragino Sans"

def fileload(filename):
    '''
    ファイル読み込みの関数。
    引数: filename(ファイル名称、拡張子抜き)
    戻り値: dataframe
    '''   
    try:
        df = pd.read_csv("input/" + filename + ".csv")
        return df
    except:
        print("ファイルがありません")
        exit()

def initial_condition(df):
    '''
    initial_condition.csvの中身を適切な形式に変換する関数
    引数: dataframe
    戻り値: 変換後の各パラメータ
    '''
    initialratio_A = df.loc[df["種別"] == "投票率", "値（%）"].values[0] * df.loc[df["種別"] == "A得票率", "値（%）"].values[0] / 10000
    initialratio_B = df.loc[df["種別"] == "投票率", "値（%）"].values[0] * df.loc[df["種別"] == "B得票率", "値（%）"].values[0] / 10000
    initialratio_N = (100 - df.loc[df["種別"] == "投票率", "値（%）"].values[0]) / 100 + (df.loc[df["種別"] == "投票率", "値（%）"].values[0] / 100 - (initialratio_A + initialratio_B))
    if initialratio_A + initialratio_B + initialratio_N > 1.001 or initialratio_A + initialratio_B + initialratio_N < 0.999:
        print("A党、B党、他党or無投票の割合の合計が1になりません")
        exit()
    elif initialratio_A < 0 or initialratio_B < 0 or initialratio_N < 0:
        print("A党、B党、他党or無投票の割合が0未満です")
        exit()
    else:
        print("A党得票率（%）, B党得票率（%）, 他政党or無投票率（%）, 合計（%）")
        print(initialratio_A * 100, initialratio_B * 100, initialratio_N * 100, (initialratio_A + initialratio_B + initialratio_N) * 100)
        return initialratio_N, initialratio_A, initialratio_B
    
def random_ranges(df):
    '''
    random_ranges.csvの中身を適切な形式に変換する関数
    引数: dataframe
    戻り値: 変換後の各パラメータ
    '''
    voteratio_min = df.loc[df["パラメータ名"] == "voteratio", "最小（％）"].values[0] / 100
    NtoA_ratio_min = df.loc[df["パラメータ名"] == "NtoA_ratio", "最小（％）"].values[0] / 100
    AtoB_ratio_min = df.loc[df["パラメータ名"] == "AtoB_ratio", "最小（％）"].values[0] / 100
    BtoA_ratio_min = df.loc[df["パラメータ名"] == "BtoA_ratio", "最小（％）"].values[0] / 100
    voteratio_max = df.loc[df["パラメータ名"] == "voteratio", "最大（％）"].values[0] / 100
    NtoA_ratio_max = df.loc[df["パラメータ名"] == "NtoA_ratio", "最大（％）"].values[0] / 100
    AtoB_ratio_max = df.loc[df["パラメータ名"] == "AtoB_ratio", "最大（％）"].values[0] / 100
    BtoA_ratio_max = df.loc[df["パラメータ名"] == "BtoA_ratio", "最大（％）"].values[0] / 100

    if not (0 <= voteratio_min <= voteratio_max <= 1):
        print("voteratio の最小値と最大値の設定に誤りがあります")
        exit()
    if not (0 <= NtoA_ratio_min <= NtoA_ratio_max <= 1):
        print("NtoA_ratio の最小値と最大値の設定に誤りがあります")
        exit()
    if not (0 <= AtoB_ratio_min <= AtoB_ratio_max <= 1):
        print("AtoB_ratio の最小値と最大値の設定に誤りがあります")
        exit()
    if not (0 <= BtoA_ratio_min <= BtoA_ratio_max <= 1):
        print("BtoA_ratio の最小値と最大値の設定に誤りがあります")
        exit()

    print("ランダムの取りうる幅（％）")
    print("voteratio:", voteratio_min * 100, "-", voteratio_max * 100)
    print("NtoA_ratio:", NtoA_ratio_min * 100, "-", NtoA_ratio_max * 100)
    print("AtoB_ratio:", AtoB_ratio_min * 100, "-", AtoB_ratio_max * 100)
    print("BtoA_ratio:", BtoA_ratio_min * 100, "-", BtoA_ratio_max * 100)
    
    return voteratio_min, voteratio_max, NtoA_ratio_min, NtoA_ratio_max, AtoB_ratio_min, AtoB_ratio_max, BtoA_ratio_min, BtoA_ratio_max

# ループ回数の設定
def loop_setting():
    loop_number = input ("試行回数を入力して下さい（半角数字、正の整数）")
    try:
        loop_number = int(loop_number)
        if loop_number <= 0:
            print("正の整数で入力して下さい")
            exit()
        else:
            return loop_number
    except:    
        print("半角数字で入力して下さい")
        exit()

# 乱数生成
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

def main():
    df = fileload("initial_condition")
    initialratio_N, initialratio_A, initialratio_B = initial_condition(df)
    df = fileload("random_ranges")
    voteratio_min, voteratio_max, NtoA_ratio_min, NtoA_ratio_max, AtoB_ratio_min, AtoB_ratio_max, BtoA_ratio_min, BtoA_ratio_max = random_ranges(df)
    loop_number = loop_setting()
    result_df = pd.DataFrame(columns=[
        "voteratio", "NtoA_ratio", "NtoB_ratio",
        "AtoB_ratio", "BtoA_ratio", "A得票率",
        "B得票率", "逆転", "投票率"
    ])
    results = []
    for i in trange(loop_number, desc="シミュレーション中", mininterval = 0.1):
        voteratio = randomized(voteratio_min, voteratio_max)
        NtoA_ratio = randomized(NtoA_ratio_min, NtoA_ratio_max)
        AtoB_ratio = randomized(AtoB_ratio_min, AtoB_ratio_max)
        BtoA_ratio = randomized(BtoA_ratio_min, BtoA_ratio_max)
        result = simulate_once(initialratio_N, initialratio_A, initialratio_B, voteratio, NtoA_ratio, AtoB_ratio, BtoA_ratio)
        results.append(result)
    result_df = pd.DataFrame(results)
    result_df.to_csv("output/result.csv", index = False)
    reversed_rate = result_df["逆転"].mean()
    print(f"B党の勝率: {round(reversed_rate * 100, 2)}%")

    cumulative_winrate = []
    win_count = 0
    for i, flag in enumerate(result_df["逆転"], start=1):
        win_count += flag
        cumulative_winrate.append(win_count / i)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, loop_number + 1), [x * 100 for x in cumulative_winrate])
    plt.xlabel("試行回数")
    plt.ylabel("B党勝率（%）")
    plt.title("収束曲線：B党勝率の推移")
    plt.grid(True)
    plt.savefig("output/convergence_curve.png")
    plt.show()
main()
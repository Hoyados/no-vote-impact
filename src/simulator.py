import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import argparse
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from logging import (
    getLogger,
    StreamHandler,
    FileHandler,
    INFO,
    ERROR,
    DEBUG,
    Formatter,
    Logger,
)

plt.rcParams["font.family"] = "Hiragino Sans"


def setup_logger() -> Logger:
    logger = getLogger("simulator")
    logger.setLevel(DEBUG)

    # clear duplicated handlers if re-run in same process
    logger.handlers.clear()

    console_handler = StreamHandler()
    console_handler.setLevel(ERROR)
    console_handler.setFormatter(
        Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    file_handler = FileHandler("log.txt", encoding="utf-8")
    file_handler.setLevel(INFO)
    file_handler.setFormatter(
        Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="簡易的な投票シミュレータです")
    parser.add_argument(
        "--config",
        default="input/default_conditions.yml",
        help="設定ファイル（yml形式、含拡張子、デフォルト: input/default_conditions.yml）",
    )
    parser.add_argument(
        "-l",
        "--loop_number",
        default=1000000,
        help="試行回数（デフォルト: 1000000）",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="乱数のシード値（デフォルト: 42)"
    )
    return parser.parse_args()

def fileload(config_path: str, logger: Logger) -> Dict:
    '''
    設定ファイル（YAML）を読み込む関数。
    引数: config_path
    戻り値: dict (設定内容)
    '''
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except Exception:
        logger.exception("設定ファイルの読み込みに失敗しました: %s", config_path)
        sys.exit(1)

def initial_condition(config: Dict, logger: Logger) -> Tuple[float, float, float]:
    '''
    YAMLから初期条件を抽出する関数
    戻り値: (N, A, B) の比率
    '''
    initialratio_A = config["投票率"] * config["A得票率"] / 10000
    initialratio_B = config["投票率"] * config["B得票率"] / 10000
    initialratio_N = (
        (100 - config["投票率"]) / 100
        + (config["投票率"] / 100 - (initialratio_A + initialratio_B))
    )
    total = initialratio_A + initialratio_B + initialratio_N
    if not (0.999 <= total <= 1.001):
        logger.error("A党、B党、他党or無投票の割合の合計が1になりません (合計: %.4f)", total)
        sys.exit(1)
    if initialratio_A < 0 or initialratio_B < 0 or initialratio_N < 0:
        logger.error("A党、B党、他党or無投票の割合が0未満です")
        sys.exit(1)

    logger.info(
        "A党得票率（%）, B党得票率（%）, 他政党or無投票率（%）, 合計（%）"
    )
    logger.info(
        "%s",
        (
            initialratio_A * 100,
            initialratio_B * 100,
            initialratio_N * 100,
            total * 100,
        ),
    )
    return initialratio_N, initialratio_A, initialratio_B
    
def random_ranges(config: Dict, logger: Logger) -> Tuple[float, ...]:
    '''
    YAMLからランダム範囲を作成する関数
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
        sys.exit(1)
    if not (0 <= NtoA_ratio_min <= NtoA_ratio_max <= 1):
        logger.error("NtoA_ratio の最小値と最大値の設定に誤りがあります")
        sys.exit(1)
    if not (0 <= AtoB_ratio_min <= AtoB_ratio_max <= 1):
        logger.error("AtoB_ratio の最小値と最大値の設定に誤りがあります")
        sys.exit(1)
    if not (0 <= BtoA_ratio_min <= BtoA_ratio_max <= 1):
        logger.error("BtoA_ratio の最小値と最大値の設定に誤りがあります")
        sys.exit(1)

    logger.info("ランダムの取りうる幅（％）")
    logger.info(f"voteratio:, {voteratio_min * 100}, -, {voteratio_max * 100}")
    logger.info(f"NtoA_ratio:, {NtoA_ratio_min * 100}, -, {NtoA_ratio_max * 100}")
    logger.info(f"AtoB_ratio:, {AtoB_ratio_min * 100}, -, {AtoB_ratio_max * 100}")
    logger.info(f"BtoA_ratio:, {BtoA_ratio_min * 100}, -, {BtoA_ratio_max * 100}")
    
    return (
        voteratio_min,
        voteratio_max,
        NtoA_ratio_min,
        NtoA_ratio_max,
        AtoB_ratio_min,
        AtoB_ratio_max,
        BtoA_ratio_min,
        BtoA_ratio_max,
    )

def loop_setting(raw_value: str, logger: Logger) -> int:
    """
    ループ回数の設定
    """
    try:
        loop_number = int(raw_value)
        if loop_number <= 0:
            logger.error("正の整数で入力して下さい")
            sys.exit(1)
        return loop_number
    except Exception:
        logger.error("半角数字で入力して下さい")
        sys.exit(1)

def randomized_vector(min_val: float, max_val: float, loop_number: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(min_val, max_val, size=loop_number)

def simulate_vectorized(
    loop_number: int,
    ranges: Dict[str, Tuple[float, float]],
    initialratio_N: float,
    initialratio_A: float,
    initialratio_B: float,
    rng: np.random.Generator,
):
    voteratio = randomized_vector(*ranges["voteratio"], loop_number, rng)
    NtoA_ratio = randomized_vector(*ranges["NtoA_ratio"], loop_number, rng)
    AtoB_ratio = randomized_vector(*ranges["AtoB_ratio"], loop_number, rng)
    BtoA_ratio = randomized_vector(*ranges["BtoA_ratio"], loop_number, rng)

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

def summarize_result(df: pd.DataFrame, initial: Tuple[float, float, float], output_dir: Path, logger: Logger) -> None:
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
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    logger.info("%s", summary_df.iloc[0].to_dict())

    fig, axs = plt.subplots(1,2)
    axs[0].errorbar(x = ["投票率(%)"], y = [summary_df.loc[0]["平均投票率(%)"]],
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
    plt.savefig(output_dir / "summary.png")
    # plt.show()

def draw_convergence_for_vector(winner_flag: pd.Series, loop_number: int, reversed_rate: float, output_dir: Path) -> None:
    cumulative_winrate = np.cumsum(winner_flag) / np.arange(1, loop_number + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, loop_number + 1), [x * 100 for x in cumulative_winrate])
    plt.axhline(reversed_rate * 100, color='red', linestyle='--', linewidth=1)
    plt.text(loop_number * 0.9, reversed_rate * 100 * 1.1, f"{round(reversed_rate * 100, 2)}%", color = 'red', va = "center")
    plt.xlabel("試行回数")
    plt.ylabel("B勝率（%）")
    plt.title("収束曲線：B勝率の推移")
    plt.grid(True)
    plt.savefig(output_dir / "convergence_curve.png")


def main() -> None:
    args = parse_args()
    logger = setup_logger()
    start_time = time.perf_counter()

    logger.info("処理開始(ファイル名: %s)", os.path.basename(args.config))
    logger.info("読み込もうとしているファイルパス: %s", args.config)

    output_dir = Path(f"output/{os.path.splitext(os.path.basename(args.config))[0]}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = fileload(args.config, logger)
    initialratio_N, initialratio_A, initialratio_B = initial_condition(config, logger)
    initial = (initialratio_N, initialratio_A, initialratio_B)
    voteratio_min, voteratio_max, NtoA_ratio_min, NtoA_ratio_max, AtoB_ratio_min, AtoB_ratio_max, BtoA_ratio_min, BtoA_ratio_max = random_ranges(config, logger)
    ranges = {
        "voteratio": (voteratio_min, voteratio_max),
        "NtoA_ratio": (NtoA_ratio_min, NtoA_ratio_max),
        "AtoB_ratio": (AtoB_ratio_min, AtoB_ratio_max),
        "BtoA_ratio": (BtoA_ratio_min, BtoA_ratio_max),
    }

    loop_number = loop_setting(args.loop_number, logger)

    rng = np.random.default_rng(args.seed)
    result = simulate_vectorized(loop_number, ranges, initialratio_N, initialratio_A, initialratio_B, rng)
    result_df = pd.DataFrame(result)
    result_df.to_csv(output_dir / "result.csv", index=False)
    reversed_rate = result_df["逆転"].mean()
    logger.info("B党の勝率: %.2f%%", round(reversed_rate * 100, 2))

    summarize_result(result_df, initial, output_dir, logger)
    draw_convergence_for_vector(result_df["逆転"], loop_number, reversed_rate, output_dir)

    end_time = time.perf_counter()
    logger.info("処理時間: %.2f", round(end_time - start_time, 2))
    logger.info("処理終了(ファイル名: %s)", os.path.basename(args.config))


if __name__ == "__main__":
    main()

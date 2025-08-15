import os
import subprocess
import glob
import argparse

folder_path = "input/batch/"
yaml_files = glob.glob(os.path.join(folder_path, "*.yml"))

parser = argparse.ArgumentParser(description = "簡易的な投票シミュレータです")
parser.add_argument("-l", "--loop_number", default = 1000000, help = "試行回数（デフォルト: 1000000）")
parser.add_argument("--seed", default = 42, help = "シード値（デフォルト: 42）")
args = parser.parse_args()

for path in sorted (yaml_files):
    filename = os.path.basename(path)
    print(f"▶︎ 実行中: {filename}")

    subprocess.run([
        "python", "src/simulator.py",
        "--config", path,
        "-l", f"{args.loop_number}",
        "--seed", f"{args.seed}"
    ])
    
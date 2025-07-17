import os
import subprocess
import glob

folder_path = "input/batch/"
yaml_files = glob.glob(os.path.join(folder_path, "*.yml"))

for path in sorted (yaml_files):
    filename = os.path.basename(path)
    print(f"▶︎ 実行中: {filename}")

    subprocess.run([
        "python", "src/simulator.py",
        "--config", path,
        "-l", "1000000"
    ])
    
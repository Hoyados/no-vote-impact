import io
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import yaml
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# 内部ロジックは既存の simulator と整合を取る
try:
    # ケース1: streamlit run src/app.py（sys.path に src が入る）
    from simulator import simulate_vectorized
except Exception:
    # ケース2: ルートが sys.path（src がパッケージ）
    from src.simulator import simulate_vectorized


# ------------------------------
# ユーティリティ（入力検証など）
# ------------------------------
def list_yaml_candidates() -> List[str]:
    paths = []
    paths += [str(p) for p in Path("input").glob("*.yml")]
    paths += [str(p) for p in Path("input/batch").glob("*.yml")]
    return sorted(paths)


def validate_config(config: Dict) -> Optional[str]:
    required_top = [
        "投票率",
        "A得票率",
        "B得票率",
        "voteratio",
        "NtoA_ratio",
        "AtoB_ratio",
        "BtoA_ratio",
    ]
    for k in required_top:
        if k not in config:
            return f"設定に '{k}' が不足しています"

    def _valid_percent(v):
        return isinstance(v, (int, float)) and 0 <= v <= 100

    if not _valid_percent(config["投票率"]):
        return "投票率は0〜100の数値で入力してください"
    if not _valid_percent(config["A得票率"]):
        return "A得票率は0〜100の数値で入力してください"
    if not _valid_percent(config["B得票率"]):
        return "B得票率は0〜100の数値で入力してください"
    if config["A得票率"] + config["B得票率"] > 100:
        return "A得票率とB得票率の合計は100以下にしてください"

    for key in ["voteratio", "NtoA_ratio", "AtoB_ratio", "BtoA_ratio"]:
        sec = config.get(key, {})
        if not isinstance(sec, dict) or "min" not in sec or "max" not in sec:
            return f"{key} は min/max を持つ辞書で指定してください"
        vmin, vmax = sec["min"], sec["max"]
        if not (isinstance(vmin, (int, float)) and isinstance(vmax, (int, float))):
            return f"{key} の min/max は数値で指定してください"
        if not (0 <= vmin <= 100 and 0 <= vmax <= 100):
            return f"{key} の min/max は0〜100の範囲で指定してください"
        if vmin > vmax:
            return f"{key} の min は max 以下にしてください"

    return None


def initial_condition_simple(config: Dict) -> Tuple[float, float, float]:
    a = config["投票率"] * config["A得票率"] / 10000
    b = config["投票率"] * config["B得票率"] / 10000
    n = (100 - config["投票率"]) / 100 + (config["投票率"]) / 100 - (a + b)
    return n, a, b


def random_ranges_simple(config: Dict) -> Dict[str, Tuple[float, float]]:
    return {
        "voteratio": (config["voteratio"]["min"] / 100, config["voteratio"]["max"] / 100),
        "NtoA_ratio": (config["NtoA_ratio"]["min"] / 100, config["NtoA_ratio"]["max"] / 100),
        "AtoB_ratio": (config["AtoB_ratio"]["min"] / 100, config["AtoB_ratio"]["max"] / 100),
        "BtoA_ratio": (config["BtoA_ratio"]["min"] / 100, config["BtoA_ratio"]["max"] / 100),
    }


def summarize_df(df: pd.DataFrame, initial: Tuple[float, float, float]) -> pd.DataFrame:
    out = pd.DataFrame(
        columns=[
            "B勝率(%)",
            "平均投票率(%)",
            "最高投票率(%)",
            "最低投票率(%)",
            "平均A得票率(%)",
            "最高A得票率(%)",
            "最低A得票率(%)",
            "平均B得票率(%)",
            "最高B得票率(%)",
            "最低B得票率(%)",
        ]
    )
    out.loc[0] = [
        round(df["逆転"].mean() * 100, 2),
        round(df["投票率"].mean(), 2),
        round(df["投票率"].max(), 2),
        round(df["投票率"].min(), 2),
        round(df["A得票率"].mean(), 2),
        round(df["A得票率"].max(), 2),
        round(df["A得票率"].min(), 2),
        round(df["B得票率"].mean(), 2),
        round(df["B得票率"].max(), 2),
        round(df["B得票率"].min(), 2),
    ]
    return out


def fig_summary(summary: pd.DataFrame, initial: Tuple[float, float, float]):
    # フォントは複数候補を指定（ブラウザ側で利用可能なものを選択）
    font_family = "Noto Sans JP, Hiragino Sans, Yu Gothic, Meiryo, sans-serif"

    fig1 = go.Figure()
    # 投票率の平均と誤差
    mean_v = summary.loc[0, "平均投票率(%)"]
    ymin = summary.loc[0, "最低投票率(%)"]
    ymax = summary.loc[0, "最高投票率(%)"]
    fig1.add_trace(
        go.Scatter(
            x=["投票率(%)"],
            y=[mean_v],
            mode="markers",
            error_y=dict(
                type="data",
                symmetric=False,
                array=[ymax - mean_v],
                arrayminus=[mean_v - ymin],
                visible=True,
            ),
            name="平均±範囲",
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=["投票率(%)"],
            y=[100 - initial[0] * 100],
            mode="markers",
            name="初期値",
        )
    )
    fig1.update_layout(
        title="投票率の要約",
        yaxis=dict(range=[0, 100]),
        template="plotly_white",
        font=dict(family=font_family),
    )

    # A/B得票率の平均と誤差
    fig2 = go.Figure()
    mean_A = summary.loc[0, "平均A得票率(%)"]
    min_A = summary.loc[0, "最低A得票率(%)"]
    max_A = summary.loc[0, "最高A得票率(%)"]
    mean_B = summary.loc[0, "平均B得票率(%)"]
    min_B = summary.loc[0, "最低B得票率(%)"]
    max_B = summary.loc[0, "最高B得票率(%)"]
    fig2.add_trace(
        go.Scatter(
            x=["A得票率(%)"],
            y=[mean_A],
            mode="markers",
            error_y=dict(
                type="data",
                symmetric=False,
                array=[max_A - mean_A],
                arrayminus=[mean_A - min_A],
                visible=True,
            ),
            name="A 平均±範囲",
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=["B得票率(%)"],
            y=[mean_B],
            mode="markers",
            error_y=dict(
                type="data",
                symmetric=False,
                array=[max_B - mean_B],
                arrayminus=[mean_B - min_B],
                visible=True,
            ),
            name="B 平均±範囲",
        )
    )
    ab_sum = initial[1] + initial[2]
    if ab_sum > 0:
        fig2.add_trace(
            go.Scatter(
                x=["A得票率(%)"],
                y=[initial[1] * 100 / ab_sum],
                mode="markers",
                name="A 初期値",
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=["B得票率(%)"],
                y=[initial[2] * 100 / ab_sum],
                mode="markers",
                name="B 初期値",
            )
        )
    fig2.update_layout(
        title="A/B得票率の要約",
        yaxis=dict(range=[0, 100]),
        template="plotly_white",
        font=dict(family=font_family),
    )

    return fig1, fig2


def fig_convergence(winner_flag: pd.Series, reversed_rate: float):
    font_family = "Noto Sans JP, Hiragino Sans, Yu Gothic, Meiryo, sans-serif"
    loop_number = len(winner_flag)
    cumulative = np.cumsum(winner_flag.values) / np.arange(1, loop_number + 1)
    x = np.arange(1, loop_number + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=cumulative * 100, mode="lines", name="B勝率"))
    fig.add_hline(y=reversed_rate * 100, line_dash="dash", line_color="red")
    fig.update_layout(
        title="収束曲線：B勝率の推移",
        xaxis_title="試行回数",
        yaxis_title="B勝率（%）",
        template="plotly_white",
        font=dict(family=font_family),
    )
    return fig


def make_output_dir(label: str) -> Path:
    safe = "".join(c for c in label if c.isalnum() or c in ("_", "-")) or "result"
    out = Path("output") / safe
    out.mkdir(parents=True, exist_ok=True)
    return out


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="投票シミュレータ", layout="wide")
st.title("簡易投票シミュレータ（Streamlit版）")

with st.sidebar:
    st.header("設定入力")
    input_mode = st.radio(
        "設定の取得方法",
        ["既存YAMLを選択", "YAMLをアップロード", "値を手入力"],
        index=0,
    )

    loop_number = st.number_input("試行回数", min_value=1, max_value=10_000_000, value=100_000, step=1)
    seed = st.number_input("乱数シード", min_value=0, max_value=1_000_000_000, value=42, step=1)
    save_results = st.checkbox("結果（CSV/概要）を保存する", value=True)

    run_button = st.button("シミュレーション実行", type="primary")


def get_config_from_ui(mode: str) -> Tuple[Optional[Dict], Optional[str]]:
    if mode == "既存YAMLを選択":
        candidates = list_yaml_candidates()
        if not candidates:
            st.info("input/ または input/batch/ にYAMLがありません。アップロードや手入力をご利用ください。")
        default_idx = 0
        if "input/default_conditions.yml" in candidates:
            default_idx = candidates.index("input/default_conditions.yml")
        selected = st.selectbox("設定ファイルを選択", options=candidates, index=default_idx if candidates else 0)
        path = selected if candidates else None
        if path and Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return config, Path(path).stem
        return None, None

    elif mode == "YAMLをアップロード":
        up = st.file_uploader("YAMLファイルを選択", type=["yml", "yaml"])
        if up is not None:
            try:
                config = yaml.safe_load(up.getvalue().decode("utf-8"))
                return config, Path(up.name).stem
            except Exception as e:
                st.error(f"YAMLの解析に失敗しました: {e}")
        return None, None

    else:  # 値を手入力
        st.subheader("基本パラメータ（%）")
        col1, col2, col3 = st.columns(3)
        with col1:
            投票率 = st.number_input("投票率", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
        with col2:
            A得票率 = st.number_input("A得票率", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
        with col3:
            B得票率 = st.number_input("B得票率", min_value=0.0, max_value=100.0, value=30.0, step=1.0)

        st.subheader("ランダム範囲（%）")
        c1, c2 = st.columns(2)
        with c1:
            voteratio_min = st.number_input("voteratio min", min_value=0.0, max_value=100.0, value=40.0, step=0.5)
            NtoA_ratio_min = st.number_input("NtoA_ratio min", min_value=0.0, max_value=100.0, value=15.0, step=0.5)
            AtoB_ratio_min = st.number_input("AtoB_ratio min", min_value=0.0, max_value=100.0, value=0.0, step=0.5)
            BtoA_ratio_min = st.number_input("BtoA_ratio min", min_value=0.0, max_value=100.0, value=0.0, step=0.5)
        with c2:
            voteratio_max = st.number_input("voteratio max", min_value=0.0, max_value=100.0, value=60.0, step=0.5)
            NtoA_ratio_max = st.number_input("NtoA_ratio max", min_value=0.0, max_value=100.0, value=25.0, step=0.5)
            AtoB_ratio_max = st.number_input("AtoB_ratio max", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
            BtoA_ratio_max = st.number_input("BtoA_ratio max", min_value=0.0, max_value=100.0, value=5.0, step=0.5)

        config = {
            "投票率": 投票率,
            "A得票率": A得票率,
            "B得票率": B得票率,
            "voteratio": {"min": voteratio_min, "max": voteratio_max},
            "NtoA_ratio": {"min": NtoA_ratio_min, "max": NtoA_ratio_max},
            "AtoB_ratio": {"min": AtoB_ratio_min, "max": AtoB_ratio_max},
            "BtoA_ratio": {"min": BtoA_ratio_min, "max": BtoA_ratio_max},
        }
        return config, "manual"


config, label = get_config_from_ui(input_mode)

if run_button:
    if config is None:
        st.error("設定が正しく取得できていません。入力内容を確認してください。")
        st.stop()

    # バリデーション
    msg = validate_config(config)
    if msg:
        st.error(msg)
        st.stop()

    # ループ回数の安全確認（メモリ・時間用の軽い注意喚起）
    if loop_number > 2_000_000:
        st.warning("大きな試行回数は時間がかかる可能性があります。")

    # 実行
    with st.spinner("シミュレーション実行中..."):
        initial = initial_condition_simple(config)
        ranges = random_ranges_simple(config)
        rng = np.random.default_rng(int(seed))
        result = simulate_vectorized(
            int(loop_number),
            ranges,
            initial[0],
            initial[1],
            initial[2],
            rng,
        )
        df = pd.DataFrame(result)
        reversed_rate = df["逆転"].mean()
        summary = summarize_df(df, initial)

    st.success(f"完了: B党の勝率 = {reversed_rate*100:.2f}%")

    # 表示
    st.subheader("要約")
    st.dataframe(summary)

    fig_v, fig_ab = fig_summary(summary, initial)
    st.plotly_chart(fig_v, use_container_width=True)
    st.plotly_chart(fig_ab, use_container_width=True)

    st.subheader("収束曲線")
    fig_conv = fig_convergence(df["逆転"], reversed_rate)
    st.plotly_chart(fig_conv, use_container_width=True)

    # 保存オプション
    if save_results:
        outdir = make_output_dir(label)
        df.to_csv(outdir / "result.csv", index=False)
        summary.to_csv(outdir / "summary.csv", index=False)
        st.info(f"結果を保存しました: {outdir}")

    # ダウンロードボタン（任意）
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button(
        label="結果CSVをダウンロード",
        data=csv_buf.getvalue(),
        file_name=f"{label}_result.csv",
        mime="text/csv",
    )

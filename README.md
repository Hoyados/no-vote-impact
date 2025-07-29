## 1. プロジェクト概要
- 無党派層の行動が選挙結果に与える影響を検証するシミュレータです。
- 簡易化のため、A(前回勝者)とBの二者の戦いのみを対象とします。
- 有権者を前回投票結果から「A投票者」「B投票者」「それ以外」に分類します。
- 投票行動の変化をランダムで決定し、逆転可能性を算出します。

## 2. フォルダ構成
```
no-vote-impact/
├── input/
│   └── default_condition.yml
├── output/
│   ├── result.csv
│   ├── summary.csv
│   ├── summary.png
│   └── convergence_curve.png
├── src/
│   ├── analyzze_result.csv # 結果の可視化
│   ├── batch.csv # バッチ実行
│   └── simulator.py
├── Condition_Design.md
├── README.md
└── log.txt
```


## 3. 必要な環境
- Python 3.10以上
- matplotlib
- pandas
- numpy
- tqdm
- PyYaml

## 4. 実行方法
### 一条件実行
```bash
python src/simulator.py --config input/default_conditions.yml --vector --loop_number 1000000
```
```bash
--config: 設定ファイルのパス(default = "input/default_conditions.yml")
--loop: 指定ありの場合,ループ処理にて実施（指定なしの場合、ベクトル演算）
--loop_number: 試行回数を指定(default = 1000000)
```
### バッチ実行
```bash
python src/batch.py -l 1000000
```
上記コマンドを実行することで、```input/batch```内部の.ymlファイルに対して連続して実行します。<br>
パラメータを変えた時の変化を見る際に便利です。

### 分析実行
```bash
python src/analyze.py
```
```output/*/result.csv```を読み取り、各種パラメータとB勝率のグラフを描画します。

## 5. 設定ファイル
- default_condition.yml: シミュレーション条件の設定
  - 投票率(%): 投票率を入力
  - A得票率(%): A(前回勝者)の得票率を入力
  - B得票率(%): Bの得票率を入力
  - voteratio(%): 前回「それ以外」のうち、「A投票者」「B投票者」のどちらかになる割合
  - NtoA_ratio(%): voteratioのうち、「A投票者」になる割合
  - AtoB_ratio(%): 前回「A投票者」のうち、「B投票者」になる割合（離反）
  - BtoA_ratio(%): 前回「B投票者」のうち、「A投票者」になる割合（離反）
  
## 6. 出力結果（/output/{yml名称}フォルダ）
- result.csv: 全試行のリスト
- summary.csv: 結果の要約
  - 投票率及びA, B得票率の最大、最小、平均を算出、記載しています。
- summary.png: 計算後の投票率の取りうる幅をグラフ化
- convergence_curve.png: 試行を重ねるうちのB勝率の推移（確率の収束度合いを図示）

## 7. 補足
- 本シミュレーションでは簡易化のためA, Bの二者間の構造に単純化しています。
- 実際の選挙では複数の候補者が存在し、過半数を取らずに当選する可能性もあります。
- A, B以外の得票は、「それ以外」に一括りにしています。
- 乱数を使っており、かつシード値の指定機能はないため、毎回結果が異なる点に留意してください。

## 8. 免責事項
- このシミュレーションモデルは選挙結果の予測ではなく、統計的挙動の観察を目的としています。
- データや計算結果を用いて発生した事象について、作者は一切の責任は負いません。
- 本モデルによる予測や推論が、実際の選挙動向や政策決定を意味するものではありません。
- 当プロジェクトの利用によって生じた損害や誤用に関して、いかなる保証も行われません。
- 現実の政治や選挙活動における意思決定には、あくまで参考情報としてください。
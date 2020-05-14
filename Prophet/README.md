# 時系列データで未来予測


Facebookが開発した「Prophet」というツールを利用する。

以下はイメージ図
![Prophet](..\DocImage\prediction_and_actual.png)
黒点が実測。青帯が予測。


## 事前準備

Googleドライブにログインし、新規　―> その他のアプリ -> アプリを追加
Colaboratory　と検索してインストール
再び　新規 -> その他のアプリ ->　Google Colaboratory

Jupyter　Notebook環境なので、ショートカットキーはだいたい同じ

より発展的な時系列データ解析を行うために「statsmodels」をアップグレードする。アップグレードには数分かかる。
`!pip install --upgrade git+https://github.com/statsmodels/statsmodels`
メニューバーのランタイムからランタイムを再起動する。


ライブラリのインポート
```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings("ignore") # warnings を表示させないようにする
```
データをアップロード
```python
from google.colab import files
upload = files.upload()
```

### Prophetでの時系列予測
Prophetの基本は「基本的構造時系列モデル」と呼ばれる枠組み。  
`時系列　＝　トレンド成分　＋　周期成分　＋　ノイズ`
また、次の特徴がある。
- 元の系列を解釈可能な系列に分解するため、解釈性が高く、現場で活用しやすい
- イベント効果（祝日による効果など）の導入など、ドメイン知識を反映することができる。
- 欠損値があっても解析できる(欠損値も合わせて推定してくれる)
- 実行が簡潔
- 可視化が充実している

#### トレンド成分
トレンドが変化する点を妥当なものにする必要があり、デフォルトで25個の均等な候補点を用意し、傾きの変化に0が多くなるように（スパース）に推定擦ることによって過学習を避けている。

#### 周期成分 
いくつかの三角関数（sin関数とcos関数）を足し合わせることで周期成分を表現している。
関数のセットの数も指定できる。デフォルトでは年周期で10、週周期で3となっているが、三角関数の数が多すぎると過学習になるため注意が必要。
さらに和の効果「トレンド＋周期成分」だけでなく、積の効果「トレンド✕周期成分」を推定することもできる。「夏は売上が1.5倍になり、冬は0.8倍になる」のような現象を表せる。


#### Prophetの利用
Prophetは、３ステップの手順・関数で使用する。
1. Prophet() ：　モデルオブジェクトの作成・モデルの詳細の決定
2. make_future_dataframe()　：　予測期間の指定、推定された値を入れるデータフレームの用意
3. predict()　：　予測

seasonality_modeについて、
「トレンド ✕ 周期成分」の場合は"multiplicative"
「トレンド ＋ 周期成分」の場合は"multiplicative"

```python
# 三角関数の数はデフォルト通りの10を指定
m = Prophet(growth = "linear", # トレンドでつなぎ合わせる関数。ロジスティック曲線にする場合は"logistic"に変更する
            yearly_seaconality = 10,
            weekly_seasonality = False,
            daily_seasonality = False, 
            seasonality_mode='multiplicative').fit(train)
```

データのない月などで大きくハズレた結果が推定された場合、妥当な周期を推定するためには、使用する三角関数の数を減らすなどの工夫をする必要がある。


## 定常性
平均・分散・自己共分散が時点tに関係なく一定である時系列を「弱定常」という。具体的に以下3つの条件を満たす場合、その時系列は弱定常であるという。
1. 平均がtに依存せず一定である。
2. 分散がtに依存せず一定である。
3. 自己共分散がtに依存せず、ラグｋによってのみ定まる
上記の条件3でk = 0とすれば条件2になる。
世の中にある時系列データのほとんどは非定常なもの。
